package bench

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

// filler is the haystack and the long-prompt material: plain prose that no
// backend has a shortcut for.
var filler = []string{
	"The harbour master logged the tide at a quarter past six and noted the wind had turned west.",
	"A delivery of copper fittings arrived on the morning train and was signed for by the foreman.",
	"The committee agreed to meet again after the audit, once the ledgers had been reconciled.",
	"Rain held off until the afternoon, by which time the orchard had been pruned back to the third row.",
	"The library extended its opening hours for the exam season and added a second reading room.",
	"Freight on the northern line was delayed by a signal fault near the junction for forty minutes.",
	"The bakery switched to a slower ferment and found the crust kept better through the week.",
	"An inventory of the stores showed eleven crates unaccounted for, later found in the annex.",
	"The surveyor marked the boundary with iron pins and filed the plan with the county office.",
	"Lanterns were lit along the quay as the last of the fishing boats came in before dark.",
}

// prose returns about n characters of filler, with a nonce woven in so no
// two prompts share a prefix the backend could serve from cache.
func prose(n int, nonce string) string {
	var b strings.Builder
	b.WriteString("Reference ")
	b.WriteString(nonce)
	b.WriteString(". ")
	for i := 0; b.Len() < n; i++ {
		b.WriteString(filler[i%len(filler)])
		b.WriteByte(' ')
	}
	return b.String()
}

// SpeedRow is one (context, concurrency) cell: medians over the runs.
type SpeedRow struct {
	Context       string `json:"context"`
	ContextTokens int    `json:"context_tokens"` // as the backend counted the prompt
	Concurrency   int    `json:"concurrency"`
	Runs          int    `json:"runs"`
	// Per stream, median across streams and runs. TTFT and Prefill come
	// from the cold pass: every stream's new prompt arriving at once.
	// DecodeTPS comes from the warm pass.
	DecodeTPS Stat `json:"decode_tps"`
	TTFT      Stat `json:"ttft_s"`
	Prefill   Stat `json:"prefill_tps"`
	// Combined output of the warm pass: all streams' tokens over the wall
	// time from the first request sent to the last stream's end.
	CombinedTPS Stat `json:"combined_tps"`
	// WarmTTFT is the warm pass's time to first token (concurrency 1 only):
	// the prompt cache at work, as on an agent's next turn.
	WarmTTFT Stat `json:"warm_ttft_s,omitempty"`
	// The backend's own decode and prefill rates, where it reports them.
	ServerDecode  Stat `json:"server_decode_tps,omitempty"`
	ServerPrefill Stat `json:"server_prefill_tps,omitempty"`
	// TokensPerChunk is the stream's granularity: 1 is token by token;
	// more means the backend sends bursts, which a client sees as a wait and
	// then a jump.
	TokensPerChunk Stat `json:"tokens_per_chunk,omitempty"`
	// Server-reported timings, median, keyed as the backend named them.
	Server   map[string]float64 `json:"server,omitempty"`
	Failures []string           `json:"failures,omitempty"`
}

// speedPlan is what the speed suite does for one backend.
type speedPlan struct {
	contexts      []int // target prompt tokens
	concurrency   []int
	runs          int
	decodeTokens  int
	charsPerToken float64
	maxContext    int // the slot's; larger contexts are skipped
	// settle is the pause between the cold and the warm pass. oMLX commits
	// a prompt to its cache a moment after the request ends: a repeat sent
	// at once was measured missing it (1.84 s to first token on a 2k
	// prompt), one sent a second later hitting it (0.12 s).
	settle time.Duration
}

// calibrate measures how many characters make a token on this backend, so
// prompts come out at the size asked for. The ratio is text-dependent, so it
// is measured on the suite's own prose.
func calibrate(ctx context.Context, cl *Client) (float64, error) {
	text := prose(4000, "calibration")
	st := cl.Complete(ctx, []Message{{Role: "user", Content: text + "\nReply with one word."}}, 4)
	if st.Err != "" {
		return 0, fmt.Errorf("calibration request: %s", st.Err)
	}
	if st.PromptTokens < 50 {
		// No usage from this backend: 3.8 characters per token is where
		// current tokenizers land on English prose.
		return 3.8, nil
	}
	return float64(len(text)) / float64(st.PromptTokens-20), nil // ~20 tokens of template
}

// runSpeed measures each cell. A run of a cell is two passes over
// `concurrency` new prompts of the target size, all sent at once each time:
//
//   - cold: one token each, for time to first token and prefill with every
//     prompt arriving together;
//   - warm, after settle: decodeTokens each on the now-cached prompts, for
//     per-stream decode and combined output.
//
// The warm pass is the shape of an agent's turn: a cached history, a little
// new input, a long reply. Measuring combined output on cold prompts with a
// short reply instead measures how the backend queues prefill: oMLX takes
// new prompts one at a time, so 4 cold 2k prompts with 64-token replies
// read 31 tok/s combined where the same model decodes 112 on warm ones.
func runSpeed(ctx context.Context, cl *Client, p speedPlan, progress func(string)) []SpeedRow {
	var rows []SpeedRow
	for _, ctxTokens := range p.contexts {
		for _, conc := range p.concurrency {
			row := SpeedRow{Context: tokensLabel(ctxTokens), Concurrency: conc, Runs: p.runs, Server: map[string]float64{}}
			if ctxTokens+p.decodeTokens+64 > p.maxContext {
				row.Failures = append(row.Failures, fmt.Sprintf("skipped: %d tokens do not fit the %d-token slot", ctxTokens, p.maxContext))
				rows = append(rows, row)
				continue
			}
			var decode, ttft, prefill, combined, warm, sdecode, sprefill, granularity []float64
			server := map[string][]float64{}
			for run := 1; run <= p.runs; run++ {
				if ctx.Err() != nil {
					break
				}
				nonce := fmt.Sprintf("%d-%d-%d-%d", ctxTokens, conc, run, time.Now().UnixNano()%100000)
				prompts := make([]string, conc)
				for i := range prompts {
					prompts[i] = prose(int(float64(ctxTokens)*p.charsPerToken), fmt.Sprintf("%s-s%d", nonce, i)) +
						"\nContinue the log in the same style for several more entries."
				}
				cold, _ := together(ctx, cl, prompts, 1)
				ok := true
				for i, st := range cold {
					if st.Err != "" {
						row.Failures = append(row.Failures, fmt.Sprintf("run %d stream %d (cold): %s", run, i, st.Err))
						ok = false
						continue
					}
					if row.ContextTokens == 0 {
						row.ContextTokens = st.PromptTokens
					}
					ttft = append(ttft, st.TTFT.Seconds())
					prefill = append(prefill, st.PrefillTPS())
					sprefill = append(sprefill, st.ServerPrefillTPS())
				}
				if !ok {
					continue
				}
				select {
				case <-ctx.Done():
				case <-time.After(p.settle):
				}
				streams, wall := together(ctx, cl, prompts, p.decodeTokens)
				var total int
				for i, st := range streams {
					if st.Err != "" {
						row.Failures = append(row.Failures, fmt.Sprintf("run %d stream %d: %s", run, i, st.Err))
						continue
					}
					total += st.CompletionTokens
					decode = append(decode, st.DecodeTPS(p.charsPerToken))
					sdecode = append(sdecode, st.ServerDecodeTPS())
					granularity = append(granularity, st.TokensPerChunk())
					if conc == 1 {
						warm = append(warm, st.TTFT.Seconds())
					}
					for k, v := range st.Server {
						server[k] = append(server[k], v)
					}
				}
				if total > 0 && wall > 0 {
					combined = append(combined, float64(total)/wall.Seconds())
				}
			}
			row.DecodeTPS = Summarise(decode)
			row.TTFT = Summarise(ttft)
			row.Prefill = Summarise(prefill)
			row.CombinedTPS = Summarise(combined)
			row.WarmTTFT = Summarise(warm)
			row.ServerDecode = Summarise(sdecode)
			row.ServerPrefill = Summarise(sprefill)
			row.TokensPerChunk = Summarise(granularity)
			for k, v := range server {
				row.Server[k] = Median(v)
			}
			progress(fmt.Sprintf("    %s × %d: %.1f tok/s per stream, %.1f combined, TTFT %.2fs cold, prefill %.0f tok/s%s",
				row.Context, conc, row.DecodeTPS.Median, row.CombinedTPS.Median, row.TTFT.Median, row.Prefill.Median,
				ifs(len(row.Failures) > 0, fmt.Sprintf(", %d failures", len(row.Failures)), "")))
			rows = append(rows, row)
		}
	}
	return rows
}

// together sends every prompt at once and returns the streams and the wall
// time from the first request sent to the last stream's end.
func together(ctx context.Context, cl *Client, prompts []string, maxTokens int) ([]Stream, time.Duration) {
	streams := make([]Stream, len(prompts))
	t0 := time.Now()
	var wg sync.WaitGroup
	for i := range prompts {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			streams[i] = cl.Complete(ctx, []Message{{Role: "user", Content: prompts[i]}}, maxTokens)
		}(i)
	}
	wg.Wait()
	return streams, time.Since(t0)
}

func tokensLabel(n int) string {
	if n >= 1024 && n%1024 == 0 {
		return fmt.Sprintf("%dk", n/1024)
	}
	return fmt.Sprint(n)
}
