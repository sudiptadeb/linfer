package bench

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

// A server that answers every completion with max_tokens tokens and records
// what it was asked, when.
type recorded struct {
	maxTokens int
	at        time.Time
}

func recordingServer(t *testing.T) (*httptest.Server, func() []recorded) {
	var mu sync.Mutex
	var got []recorded
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			MaxTokens int `json:"max_tokens"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		mu.Lock()
		got = append(got, recorded{body.MaxTokens, time.Now()})
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		for i := 0; i < body.MaxTokens; i++ {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"tok \"}}]}\n\n")
		}
		fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2000,\"completion_tokens\":%d}}\n\ndata: [DONE]\n\n", body.MaxTokens)
	}))
	t.Cleanup(srv.Close)
	return srv, func() []recorded { mu.Lock(); defer mu.Unlock(); return append([]recorded(nil), got...) }
}

func TestSpeedRunsAColdPassThenAWarmPassAfterTheSettle(t *testing.T) {
	srv, got := recordingServer(t)
	cl := newClient(srv.URL, "m")
	rows := runSpeed(context.Background(), cl, speedPlan{contexts: []int{256}, concurrency: []int{3}, runs: 1,
		decodeTokens: 40, charsPerToken: 4, maxContext: 8192, settle: 300 * time.Millisecond}, func(string) {})

	reqs := got()
	if len(reqs) != 6 {
		t.Fatalf("%d requests, want 3 cold and 3 warm", len(reqs))
	}
	for i, r := range reqs {
		want := 1
		if i >= 3 {
			want = 40
		}
		if r.maxTokens != want {
			t.Errorf("request %d asked for %d tokens, want %d", i, r.maxTokens, want)
		}
	}
	if gap := reqs[3].at.Sub(reqs[2].at); gap < 300*time.Millisecond {
		t.Errorf("warm pass began %v after the cold one, before the settle", gap)
	}
	// Combined counts the warm pass's tokens only: 3 streams × 40.
	if c := rows[0].CombinedTPS; c.N != 1 || c.Median <= 0 {
		t.Fatalf("combined %+v", c)
	}
	if rows[0].TTFT.N != 3 || rows[0].DecodeTPS.N != 3 {
		t.Errorf("ttft %d and decode %d samples, want 3 each", rows[0].TTFT.N, rows[0].DecodeTPS.N)
	}
}
