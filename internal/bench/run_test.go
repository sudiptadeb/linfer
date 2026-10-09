package bench

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	lb "github.com/sudiptadeb/linfer-bench/bench"
	"github.com/sudiptadeb/linfer/internal/linfer"
)

// The test binary doubles as a fake OpenAI-compatible backend when
// LINFER_FAKE_BACKEND is set: /health, /v1/models, and chat completions
// that stream a few tokens or answer the tool suite well enough to score.
func TestMain(m *testing.M) {
	if os.Getenv("LINFER_FAKE_BACKEND") == "1" {
		fakeBackend()
		return
	}
	os.Exit(m.Run())
}

func fakeBackend() {
	port := ""
	for i, a := range os.Args {
		if a == "--port" && i+1 < len(os.Args) {
			port = os.Args[i+1]
		}
	}
	ln, err := net.Listen("tcp", "127.0.0.1:"+port)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) })
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"data":[{"id":"fake","owned_by":"llamacpp"}]}`)
	})
	mux.HandleFunc("/v1/chat/completions", fakeChat)
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM)
	defer stop()
	srv := &http.Server{Handler: mux}
	go srv.Serve(ln)
	<-ctx.Done()
	srv.Close()
}

func call(id, name, args string) lb.ToolCall {
	return lb.ToolCall{ID: id, Type: "function", Function: lb.FunctionCall{Name: name, Arguments: args}}
}

func fakeChat(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Messages  []lb.Message `json:"messages"`
		Stream    bool         `json:"stream"`
		MaxTokens int          `json:"max_tokens"`
	}
	body, _ := io.ReadAll(r.Body)
	json.Unmarshal(body, &req)
	last := req.Messages[len(req.Messages)-1]
	promptChars := 0
	for _, m := range req.Messages {
		promptChars += len(m.Content)
	}
	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		words := []string{"The", " log", " continues", " with", " more", " entries", "."}
		if strings.Contains(last.Content, "vault passphrase") {
			words = []string{"amber", "-falcon", "-7291"}
		}
		n := min(req.MaxTokens, len(words))
		time.Sleep(time.Duration(promptChars/200) * time.Millisecond) // a prefill that grows with the prompt
		for i := 0; i < n; i++ {
			chunk, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"content": words[i]}}}})
			fmt.Fprintf(w, "data: %s\n\n", chunk)
			fl.Flush()
			time.Sleep(5 * time.Millisecond)
		}
		final, _ := json.Marshal(map[string]any{
			"choices": []any{},
			"usage":   map[string]any{"prompt_tokens": promptChars/4 + 20, "completion_tokens": n},
			"timings": map[string]any{"predicted_per_second": 123.0},
		})
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", final)
		return
	}
	// The tool suite, answered correctly, with one deliberate miss: the
	// calendar event's location as a string, so a failure shows in the report.
	msg := map[string]any{"role": "assistant", "content": nil}
	calls := func(cs ...lb.ToolCall) { msg["tool_calls"] = cs }
	text := strings.ToLower(last.Content)
	switch {
	case last.Role == "tool":
		msg["content"] = "It is 7°C in Oslo with light rain."
	case strings.Contains(text, "tokyo") && strings.Contains(text, "berlin"):
		calls(call("c1", "get_weather", `{"location":"Tokyo"}`), call("c2", "get_weather", `{"location":"Berlin"}`))
	case strings.Contains(text, "weather"):
		calls(call("c1", "get_weather", `{"location":"Paris","unit":"celsius"}`))
	case strings.Contains(text, "calculator"):
		calls(call("c1", "calculate", `{"expression":"1234 * 5678"}`))
	case strings.Contains(text, "poem"):
		msg["content"] = "Waves fold the light\nand carry it home."
	case strings.Contains(text, "look up that account"):
		calls(call("c1", "lookup_account", `{"account_id":"4471029385"}`))
	case strings.Contains(text, "calendar event"):
		calls(call("c1", "create_calendar_event", `{"title":"Design review","start":"2026-10-12T10:00","end":"2026-10-12T11:00","attendees":["alice@example.com","bob@example.com"],"location":"the Hub"}`))
	case strings.Contains(text, "search the web"):
		calls(call("c1", "web_search", `{"query":"latest Go release notes","max_results":3}`))
	case strings.Contains(text, "save the text"):
		calls(call("c1", "write_file", `{"path":"notes.txt","content":"hello world"}`))
	default:
		msg["content"] = "hello"
	}
	out, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"message": msg}}})
	w.Header().Set("Content-Type", "application/json")
	w.Write(out)
}

func freePort(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	_, port, _ := net.SplitHostPort(ln.Addr().String())
	return port
}

func fakeCommand(port string) linfer.Command {
	return linfer.Command{Path: os.Args[0], Args: []string{"--port", port}, Env: []string{"LINFER_FAKE_BACKEND=1"},
		Health: "http://127.0.0.1:" + port + "/health"}
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

// testSetup: a config whose files exist (so the llama variant is available),
// a serve daemon on its own port with the fake backend, and a runner that
// launches the fake backend on the config's listen port through
// linfer.Start, as production does. The runner's output is returned for
// the tests to read.
func testSetup(t *testing.T) (linfer.Config, *linfer.Supervisor, *Runner, *bool, *bytes.Buffer) {
	t.Helper()
	// A short directory: a unix socket path is limited to 104 bytes on
	// macOS, and t.TempDir() carries the test's name.
	dir, err := os.MkdirTemp("", "lb")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	for _, f := range []string{"model.gguf", "llama-server"} {
		os.WriteFile(filepath.Join(dir, f), []byte("x"), 0o755)
	}
	listen := "127.0.0.1:" + freePort(t)
	cfg, err := linfer.Parse([]byte(fmt.Sprintf("listen: %s\nmodel:\n  id: fake\n  gguf: %s\nllama_bin: %s\ndir: %s\n",
		listen, filepath.Join(dir, "model.gguf"), filepath.Join(dir, "llama-server"), dir)))
	if err != nil {
		t.Fatal(err)
	}
	hw := linfer.Hardware{OS: "linux", Arch: "amd64", RAM: 64 * linfer.GiB}

	// The daemon, on another port, as the production one would be while a
	// bench runs.
	daemonPort := freePort(t)
	s := linfer.NewSupervisor(cfg, hw, slog.New(slog.NewTextHandler(io.Discard, nil)))
	s.Backoff = 50 * time.Millisecond
	s.Prepare = func() (linfer.Plan, linfer.Command, error) {
		return linfer.Plan{Backend: "llama", Slots: 1, Context: 4096}, fakeCommand(daemonPort), nil
	}
	dctx, dcancel := context.WithCancel(context.Background())
	t.Cleanup(dcancel)
	go s.Run(dctx)
	go linfer.ServeControl(dctx, cfg.Paths().Socket, s)
	waitFor(t, "daemon healthy", func() bool { return s.Status().Healthy })

	pausedDuring := false
	var out bytes.Buffer
	opts := Options{Out: filepath.Join(dir, "out"), Store: filepath.Join(dir, "store"), Tags: []string{"test"},
		Bench: lb.Options{Profile: lb.ProfileQuick, Contexts: lb.Sizes{128}, Concurrency: []int{1, 2}, ReplyTokens: 7, Settle: lb.Duration(10 * time.Millisecond)}}
	r := New(cfg, hw, opts, &out)
	r.Plan = func(v Variant) (linfer.Plan, string, error) {
		return linfer.Plan{Backend: v.Backend, Slots: 2, Context: 4096, CacheRAMMB: 256, Weights: 100 * linfer.MiB}, cfg.GGUFPath(), nil
	}
	r.FreeMemory = func() uint64 { return 1 << 40 }
	r.Start = func(ctx context.Context, v Variant, plan linfer.Plan) (*Running, error) {
		// Seen from the launch: the daemon must be paused by now.
		if st, err := (linfer.Client{Socket: cfg.Paths().Socket}).Status(ctx); err == nil && st.Paused && st.PID == 0 {
			pausedDuring = true
		}
		_, port, _ := net.SplitHostPort(listen)
		p, err := linfer.Start(fakeCommand(port), filepath.Join(dir, "bench-backend.log"))
		if err != nil {
			return nil, err
		}
		if err := p.WaitHealthy(ctx, 10*time.Second); err != nil {
			return nil, err
		}
		return &Running{Plan: plan, RSS: p.RSS, Exited: p.Exited, Stop: func() { p.Stop(5 * time.Second) }}, nil
	}
	return cfg, s, r, &pausedDuring, &out
}

// A full quick run against the fake backend: the daemon is paused while the
// bench's own backend runs and resumed after; every suite produces rows;
// the row carries the launch details and the backend tag; the report files
// are written and the run is stored; the deliberate tool miss is listed.
func TestRunPausesDaemonAndReports(t *testing.T) {
	cfg, s, r, pausedDuring, out := testSetup(t)
	res, err := r.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !*pausedDuring {
		t.Error("the daemon was not paused when the bench backend launched")
	}
	waitFor(t, "daemon resumed and healthy", func() bool {
		st := s.Status()
		return !st.Paused && st.Healthy
	})
	if len(res.Rows) != 1 || res.Rows[0].Label != "llama" || res.Rows[0].Error != "" || res.Title != "fake" || res.ID == "" || res.Tags[0] != "test" {
		t.Fatalf("results %+v", res)
	}
	b := res.Rows[0]
	if b.Server != "llama.cpp" || b.Tags[0] != "backend:llama" || len(b.Info) != 3 || b.Info[0].Value != "0.1 GiB" || b.Info[1].Value != "2×4096, cache 256 MB" || !strings.HasPrefix(b.Info[2].Key, "load") {
		t.Errorf("row details %+v %+v", b.Info, b.Tags)
	}
	if len(b.Speed) != 2 || b.Speed[0].DecodeTPS.N == 0 || b.Speed[0].WarmTTFT.N == 0 || b.Speed[1].Concurrency != 2 || b.Speed[1].CombinedTPS.N == 0 {
		t.Errorf("speed rows %+v", b.Speed)
	}
	if b.Speed[0].Server["timings.predicted_per_second"] != 123 {
		t.Errorf("server timings not kept: %v", b.Speed[0].Server)
	}
	if b.Tools == nil || b.Tools.Cases != 9 || b.Tools.Passed != 8 || len(b.Tools.Failures) != 1 ||
		!strings.Contains(b.Tools.Failures[0], "location: is a string") {
		t.Errorf("tools %+v", b.Tools)
	}
	if len(b.Context) != 1 || !b.Context[0].Found {
		t.Errorf("context %+v", b.Context)
	}
	for _, f := range []string{"results.json", "report.md", "report.html"} {
		if _, err := os.Stat(filepath.Join(r.Opts.Out, f)); err != nil {
			t.Error(err)
		}
	}
	if _, err := os.Stat(filepath.Join(r.Opts.Store, res.ID+".json")); err != nil {
		t.Error("the run is not in the store:", err)
	}
	if len(res.Notes) == 0 || !strings.Contains(res.Notes[0], "paused for the run") {
		t.Errorf("notes %v", res.Notes)
	}
	if _, err := os.Stat(cfg.Paths().Socket); err != nil {
		t.Error("the daemon's socket is gone")
	}
	for _, want := range []string{"bench: fake on llama", "quick profile", "9 tool cases (builtin)", "pausing the linfer daemon", "== llama", "loading", "up in", "llama.cpp; timings", "128 × 1:", "ok   calculator", "FAIL nested object", "metric", "launch", "tools accuracy", "89% (8/9)", "stored as run", "linfer daemon resumed"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output lacks %q:\n%s", want, out.String())
		}
	}
}

// When the bench backend fails to start, or the run is cancelled, the
// daemon is still resumed.
func TestRunResumesDaemonOnFailure(t *testing.T) {
	_, s, r, _, _ := testSetup(t)
	r.Start = func(context.Context, Variant, linfer.Plan) (*Running, error) {
		return nil, fmt.Errorf("the backend died loading")
	}
	res, err := r.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if res.Rows[0].Error != "the backend died loading" {
		t.Errorf("error not recorded: %+v", res.Rows[0])
	}
	waitFor(t, "daemon resumed after failure", func() bool { return !s.Status().Paused && s.Status().Healthy })

	// Cancelled before anything launches: still resumed, and the error is the cancellation.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := r.Run(ctx); err != context.Canceled {
		t.Errorf("cancelled run: %v", err)
	}
	waitFor(t, "daemon resumed after cancel", func() bool { return !s.Status().Paused && s.Status().Healthy })
}

// An already-paused daemon is left alone: whoever paused it wants it so.
func TestRunLeavesPausedDaemonPaused(t *testing.T) {
	_, s, r, _, _ := testSetup(t)
	if err := s.Pause(); err != nil {
		t.Fatal(err)
	}
	r.Start = func(context.Context, Variant, linfer.Plan) (*Running, error) {
		return nil, fmt.Errorf("skip")
	}
	res, err := r.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(200 * time.Millisecond)
	if !s.Status().Paused {
		t.Error("the bench resumed a daemon it did not pause")
	}
	if len(res.Notes) == 0 || !strings.Contains(res.Notes[0], "already paused") {
		t.Errorf("notes %v", res.Notes)
	}
}

// The launch waits for memory that is still being handed back.
func TestRunWaitsForMemoryToComeFree(t *testing.T) {
	_, _, r, _, _ := testSetup(t)
	calls := 0
	r.FreeMemory = func() uint64 {
		calls++
		if calls < 3 {
			return linfer.MiB // the previous model is still letting go
		}
		return 1 << 40
	}
	r.MemoryWait = 30 * time.Second
	res, err := r.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Rows) != 1 || res.Rows[0].Error != "" {
		t.Fatalf("rows %+v", res.Rows)
	}
}

// A config pinned to llama still compares against mlx when its weights and
// omlx exist.
func TestVariantsComparesMLXEvenWhenPinnedToLlama(t *testing.T) {
	dir := t.TempDir()
	for _, f := range []string{"model.gguf", "llama-server", "omlx", "mlx/config.json"} {
		os.MkdirAll(filepath.Dir(filepath.Join(dir, f)), 0o755)
		os.WriteFile(filepath.Join(dir, f), []byte("x"), 0o755)
	}
	cfg := linfer.Config{Backend: linfer.BackendLlama, LlamaBin: filepath.Join(dir, "llama-server"), OMLXBin: filepath.Join(dir, "omlx"),
		Model: linfer.Model{ID: "m", GGUF: filepath.Join(dir, "model.gguf"), MLX: filepath.Join(dir, "mlx")}}
	got := Variants(cfg, linfer.Hardware{OS: "darwin", Arch: "arm64"})
	if len(got) != 2 || got[0].Name != "llama" || got[1].Name != "mlx" {
		t.Fatalf("variants %+v", got)
	}
}

// The machine linfer detected carries over with its GPU detail.
func TestMachine(t *testing.T) {
	m := Machine(linfer.Hardware{OS: "darwin", Arch: "arm64", Chip: "Apple M3 Ultra", CPUs: 32, RAM: 256 * linfer.GiB, GPU: "metal", GPUMem: 222 * linfer.GiB, GPUMemSource: "measured"})
	if m.String() != "darwin/arm64, Apple M3 Ultra, 32 CPUs, 256.0 GiB RAM, GPU metal 222.0 GiB (measured)" {
		t.Errorf("machine %q", m.String())
	}
}
