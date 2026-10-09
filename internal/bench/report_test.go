package bench

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/sudiptadeb/linfer/internal/linfer"
)

func sampleResults() Results {
	tools := ToolsScore{Cases: 9, Passed: 8, CallCases: 7, ValidCalls: 7, RightTool: 7, SchemaValid: 6, NoCallCases: 2, NoCallCorrect: 2,
		Failures: []string{"nested object and array args (run 1): create_calendar_event: location: is a string, want an object"}}
	mlxTools := ToolsScore{Cases: 9, Passed: 6, CallCases: 7, ValidCalls: 6, RightTool: 6, SchemaValid: 5, NoCallCases: 2, NoCallCorrect: 1, HTTPErrors: 1, Leaks: 1}
	return Results{
		Started: time.Date(2026, 10, 9, 14, 0, 0, 0, time.UTC), Duration: 600, Model: "m:q8",
		Machine: linfer.Hardware{OS: "darwin", Arch: "arm64", Chip: "Apple M3 Ultra", CPUs: 32, RAM: 256 * linfer.GiB, GPU: "metal", GPUMem: 222 * linfer.GiB, GPUMemSource: "measured"},
		Options: Options{Runs: 3, Suites: []string{"speed", "tools", "context"}},
		Backends: []BackendResult{
			{Name: "llama", Backend: "llama", Weights: "/w/model.gguf", WeightGB: 176, Plan: linfer.Plan{Backend: "llama", Slots: 8, Context: 131072, CacheRAMMB: 16384}, LoadS: 90, RSS: 180 * linfer.GiB,
				Speed: []SpeedRow{
					{Context: "2k", Concurrency: 1, Runs: 3, DecodeTPS: Stat{Median: 32, Min: 31, Max: 33, N: 3}, CombinedTPS: Stat{Median: 32, Min: 31, Max: 33, N: 3}, TTFT: Stat{Median: 2.1, Min: 2, Max: 2.3, N: 3}, Prefill: Stat{Median: 950, Min: 900, Max: 1000, N: 3}, WarmTTFT: Stat{Median: 0.4, Min: 0.3, Max: 0.5, N: 3}},
					{Context: "2k", Concurrency: 8, Runs: 3, DecodeTPS: Stat{Median: 13, Min: 12, Max: 14, N: 3}, CombinedTPS: Stat{Median: 104, Min: 100, Max: 108, N: 3}, TTFT: Stat{Median: 4, Min: 3.8, Max: 4.2, N: 3}, Prefill: Stat{Median: 500, Min: 480, Max: 520, N: 3}},
				},
				Tools:   &tools,
				Context: []ContextRow{{Context: "2k", Depth: 10, Found: true}, {Context: "2k", Depth: 50, Found: true}, {Context: "2k", Depth: 90, Found: false, Answer: "amber-falcon-7219"}},
			},
			{Name: "mlx", Backend: "mlx", Weights: "/w/mlx", WeightGB: 182, Plan: linfer.Plan{Backend: "mlx", MaxConcurrent: 6, ContextWindow: 262144}, LoadS: 60, RSS: 190 * linfer.GiB,
				Speed: []SpeedRow{
					{Context: "2k", Concurrency: 1, Runs: 3, DecodeTPS: Stat{Median: 66, Min: 64, Max: 68, N: 3}, CombinedTPS: Stat{Median: 66, Min: 64, Max: 68, N: 3}, TTFT: Stat{Median: 1.5, Min: 1.4, Max: 1.6, N: 3}, Prefill: Stat{Median: 1300, Min: 1250, Max: 1350, N: 3}, WarmTTFT: Stat{Median: 0.3, Min: 0.3, Max: 0.4, N: 3},
						ServerDecode: Stat{Median: 48, Min: 47, Max: 49, N: 3}, ServerPrefill: Stat{Median: 1400, Min: 1380, Max: 1420, N: 3}, TokensPerChunk: Stat{Median: 34.6, Min: 30, Max: 40, N: 3}},
					{Context: "2k", Concurrency: 8, Runs: 3, DecodeTPS: Stat{Median: 13.5, Min: 12, Max: 15, N: 3}, CombinedTPS: Stat{Median: 101, Min: 95, Max: 110, N: 3}, TTFT: Stat{Median: 5, Min: 4.5, Max: 5.5, N: 3}, Prefill: Stat{Median: 450, Min: 400, Max: 500, N: 3},
						Failures: []string{"run 2 stream 7: HTTP 400: prefill exceeds memory"}},
				},
				Tools:   &mlxTools,
				Context: []ContextRow{{Context: "2k", Depth: 10, Found: true}, {Context: "2k", Depth: 50, Found: true}, {Context: "2k", Depth: 90, Found: true}},
			},
			{Name: "spec", Backend: "llama", Error: "no draft model configured"},
		},
	}
}

// The table has a column per backend, a row per cell metric, the tool and
// needle tallies, and the backend that did not run says why.
func TestWriteTable(t *testing.T) {
	var b bytes.Buffer
	WriteTable(&b, sampleResults())
	out := b.String()
	for _, want := range []string{
		"metric", "llama", "mlx", "spec",
		"decode tok/s/stream 2k×1", "32.0 (31.0–33.0)", "66.0 (64.0–68.0)",
		"combined tok/s 2k×8", "104.0 (100.0–108.0)",
		"warm ttft s 2k", "0.40 (0.30–0.50)",
		"  server-reported", "48.0 (47.0–49.0)", "1400 (1380–1420)",
		"  tokens per stream chunk", "34.6 (30.0–40.0)",
		"tools accuracy", "89% (8/9)", "67% (6/9)",
		"leaks / http errors", "1 / 1",
		"needle found", "2 of 3", "3 of 3",
		"failures", "spec did not run: no draft model configured",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("table lacks %q:\n%s", want, out)
		}
	}
}

// The verdict names the fastest single stream with the ratio, calls the
// top-concurrency result within noise when the ranges overlap, gives tool
// accuracy per backend, and lists failures.
func TestVerdict(t *testing.T) {
	v := strings.Join(Verdict(sampleResults()), "\n")
	for _, want := range []string{
		"fastest single agent (2k context, 1 stream): mlx at 66.0 tok/s, 2.06× llama (32.0)",
		"most combined output (2k context, 8 streams): llama at 104.0 tok/s; mlx at 101.0 is within noise",
		"tool accuracy: llama 89% (8/9), mlx 67% (6/9)",
		"needle retrieved: llama 2/3, mlx 3/3",
		"mlx: client-side decode 66.0 vs server-reported 48.0 tok/s at 2k×1 (34.6 tokens per stream chunk)",
		"mlx: 2 failure(s), first: speed 2k×8: run 2 stream 7: HTTP 400",
		"spec did not run",
	} {
		if !strings.Contains(v, want) {
			t.Errorf("verdict lacks %q:\n%s", want, v)
		}
	}
	if strings.Contains(v, "single run per cell") {
		t.Error("three runs flagged as a single run")
	}
	one := sampleResults()
	one.Options.Runs = 1
	if !strings.Contains(strings.Join(Verdict(one), "\n"), "single run per cell") {
		t.Error("a single run must carry the noise caveat")
	}
}

// report.md carries the machine, the weights each backend used, the table,
// the tool failures and the errors.
func TestMarkdown(t *testing.T) {
	md := Markdown(sampleResults())
	for _, want := range []string{
		"# linfer bench: m:q8", "Apple M3 Ultra", "222.0 GiB usable",
		"**llama** (llama): `/w/model.gguf`, 176.0 GiB", "8 slots × 131072 tokens",
		"**mlx** (mlx): `/w/mlx`, 182.0 GiB", "6 concurrent requests",
		"## Verdict", "## Comparison", "decode tok/s/stream 2k×1",
		"### llama: tool failures", "location: is a string, want an object",
		"### mlx: errors", "HTTP 400: prefill exceeds memory",
		"**spec**: did not run",
	} {
		if !strings.Contains(md, want) {
			t.Errorf("markdown lacks %q", want)
		}
	}
}
