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
	// Per stream, median across streams and runs.
	DecodeTPS Stat `json:"decode_tps"`
	TTFT      Stat `json:"ttft_s"`
	Prefill   Stat `json:"prefill_tps"`
	// Combined decode of all streams: total tokens over the window from the
	// first stream's first token to the last stream's last.
	CombinedTPS Stat `json:"combined_tps"`
	// WarmTTFT is the time to first token when the same prompt is sent
	// again at once (concurrency 1 only): the prompt cache at work.
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

// runSpeed measures each cell. Each run of a cell sends `concurrency`
// streams at once, each with its own cold prompt of the target size, and
// takes the per-stream numbers from every stream.
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
				streams := make([]Stream, conc)
				starts := make([]time.Time, conc)
				var wg sync.WaitGroup
				for i := range prompts {
					wg.Add(1)
					go func(i int) {
						defer wg.Done()
						starts[i] = time.Now()
						streams[i] = cl.Complete(ctx, []Message{{Role: "user", Content: prompts[i]}}, p.decodeTokens)
					}(i)
				}
				wg.Wait()
				var total int // decode tokens after each stream's first chunk
				var firstTok, lastTok time.Time
				for i, st := range streams {
					if st.Err != "" {
						row.Failures = append(row.Failures, fmt.Sprintf("run %d stream %d: %s", run, i, st.Err))
						continue
					}
					total += st.CompletionTokens - st.FirstChunkTokens(p.charsPerToken)
					if row.ContextTokens == 0 {
						row.ContextTokens = st.PromptTokens
					}
					decode = append(decode, st.DecodeTPS(p.charsPerToken))
					ttft = append(ttft, st.TTFT.Seconds())
					prefill = append(prefill, st.PrefillTPS())
					sdecode = append(sdecode, st.ServerDecodeTPS())
					sprefill = append(sprefill, st.ServerPrefillTPS())
					granularity = append(granularity, st.TokensPerChunk())
					for k, v := range st.Server {
						server[k] = append(server[k], v)
					}
					ft := starts[i].Add(st.TTFT)
					lt := starts[i].Add(st.Total)
					if firstTok.IsZero() || ft.Before(firstTok) {
						firstTok = ft
					}
					if lt.After(lastTok) {
						lastTok = lt
					}
				}
				if total > 0 && lastTok.After(firstTok) {
					combined = append(combined, float64(total)/lastTok.Sub(firstTok).Seconds())
				}
				// The warm check: the same prompt again, at once.
				if conc == 1 && streams[0].Err == "" {
					w := cl.Complete(ctx, []Message{{Role: "user", Content: prompts[0]}}, 8)
					if w.Err == "" {
						warm = append(warm, w.TTFT.Seconds())
					}
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
			progress(fmt.Sprintf("    %s × %d: %.1f tok/s per stream, %.1f combined, TTFT %.2fs, prefill %.0f tok/s%s",
				row.Context, conc, row.DecodeTPS.Median, row.CombinedTPS.Median, row.TTFT.Median, row.Prefill.Median,
				ifs(len(row.Failures) > 0, fmt.Sprintf(", %d failures", len(row.Failures)), "")))
			rows = append(rows, row)
		}
	}
	return rows
}

func tokensLabel(n int) string {
	if n >= 1024 && n%1024 == 0 {
		return fmt.Sprintf("%dk", n/1024)
	}
	return fmt.Sprint(n)
}
