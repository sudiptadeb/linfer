package bench

import (
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/sudiptadeb/linfer/internal/linfer"
)

// Results is the whole run, what results.json holds.
type Results struct {
	Started  time.Time       `json:"started"`
	Duration float64         `json:"duration_s"`
	Machine  linfer.Hardware `json:"machine"`
	Model    string          `json:"model"`
	Options  Options         `json:"options"`
	Backends []BackendResult `json:"backends"`
	Notes    []string        `json:"notes,omitempty"`
}

// BackendResult is one backend's row set.
type BackendResult struct {
	Name     string       `json:"name"`
	Backend  string       `json:"backend"`
	Weights  string       `json:"weights"`
	WeightGB float64      `json:"weight_gib"`
	Plan     linfer.Plan  `json:"plan"`
	LoadS    float64      `json:"load_s"`
	RSS      uint64       `json:"rss_bytes"`
	Speed    []SpeedRow   `json:"speed,omitempty"`
	Tools    *ToolsScore  `json:"tools,omitempty"`
	Context  []ContextRow `json:"context,omitempty"`
	Failures []string     `json:"failures,omitempty"`
	Error    string       `json:"error,omitempty"` // the backend did not run at all
}

func (b BackendResult) speedAt(ctxLabel string, conc int) *SpeedRow {
	for i := range b.Speed {
		if b.Speed[i].Context == ctxLabel && b.Speed[i].Concurrency == conc {
			return &b.Speed[i]
		}
	}
	return nil
}

func (b BackendResult) contextFound() (found, total int) {
	for _, r := range b.Context {
		if r.Err != "" && strings.HasPrefix(r.Err, "skipped") {
			continue
		}
		total++
		if r.Found {
			found++
		}
	}
	return
}

func (b BackendResult) allFailures() []string {
	out := append([]string(nil), b.Failures...)
	for _, s := range b.Speed {
		for _, f := range s.Failures {
			if !strings.HasPrefix(f, "skipped") {
				out = append(out, fmt.Sprintf("speed %s×%d: %s", s.Context, s.Concurrency, f))
			}
		}
	}
	if b.Tools != nil && b.Tools.HTTPErrors > 0 {
		out = append(out, fmt.Sprintf("tools: %d HTTP errors", b.Tools.HTTPErrors))
	}
	for _, c := range b.Context {
		if c.Err != "" && !strings.HasPrefix(c.Err, "skipped") {
			out = append(out, fmt.Sprintf("context %s at %d%%: %s", c.Context, c.Depth, c.Err))
		}
	}
	return out
}

// WriteTable prints the terminal comparison: one column per backend.
func WriteTable(w io.Writer, r Results) {
	tw := tabwriter.NewWriter(w, 2, 8, 2, ' ', 0)
	head := "metric"
	for _, b := range r.Backends {
		head += "\t" + b.Name
	}
	fmt.Fprintln(tw, head)
	row := func(label string, f func(b BackendResult) string) {
		line := label
		for _, b := range r.Backends {
			if b.Error != "" {
				line += "\t-"
				continue
			}
			line += "\t" + f(b)
		}
		fmt.Fprintln(tw, line)
	}
	row("weights", func(b BackendResult) string { return fmt.Sprintf("%.1f GiB", b.WeightGB) })
	row("load", func(b BackendResult) string { return fmt.Sprintf("%.0fs, rss %s", b.LoadS, linfer.HumanBytes(b.RSS)) })
	row("launch", func(b BackendResult) string {
		if b.Backend == linfer.BackendMLX {
			return fmt.Sprintf("%d concurrent, window %d", b.Plan.MaxConcurrent, b.Plan.ContextWindow)
		}
		return fmt.Sprintf("%d×%d, cache %d MB", b.Plan.Slots, b.Plan.Context, b.Plan.CacheRAMMB)
	})
	for _, cell := range cells(r) {
		ctxLabel, conc := cell.ctx, cell.conc
		row(fmt.Sprintf("decode tok/s/stream %s×%d", ctxLabel, conc), func(b BackendResult) string {
			return statStr(b.speedAt(ctxLabel, conc), func(s *SpeedRow) Stat { return s.DecodeTPS }, "%.1f")
		})
		row("  server-reported", func(b BackendResult) string {
			return statStr(b.speedAt(ctxLabel, conc), func(s *SpeedRow) Stat { return s.ServerDecode }, "%.1f")
		})
		row("  tokens per stream chunk", func(b BackendResult) string {
			return statStr(b.speedAt(ctxLabel, conc), func(s *SpeedRow) Stat { return s.TokensPerChunk }, "%.1f")
		})
		row(fmt.Sprintf("combined tok/s %s×%d", ctxLabel, conc), func(b BackendResult) string {
			return statStr(b.speedAt(ctxLabel, conc), func(s *SpeedRow) Stat { return s.CombinedTPS }, "%.1f")
		})
		row(fmt.Sprintf("cold ttft s %s×%d", ctxLabel, conc), func(b BackendResult) string {
			return statStr(b.speedAt(ctxLabel, conc), func(s *SpeedRow) Stat { return s.TTFT }, "%.2f")
		})
		row(fmt.Sprintf("prefill tok/s %s×%d", ctxLabel, conc), func(b BackendResult) string {
			return statStr(b.speedAt(ctxLabel, conc), func(s *SpeedRow) Stat { return s.Prefill }, "%.0f")
		})
		row("  server-reported", func(b BackendResult) string {
			return statStr(b.speedAt(ctxLabel, conc), func(s *SpeedRow) Stat { return s.ServerPrefill }, "%.0f")
		})
		if conc == 1 {
			row(fmt.Sprintf("warm ttft s %s", ctxLabel), func(b BackendResult) string {
				return statStr(b.speedAt(ctxLabel, conc), func(s *SpeedRow) Stat { return s.WarmTTFT }, "%.2f")
			})
		}
	}
	if anyTools(r) {
		row("tools accuracy", func(b BackendResult) string {
			if b.Tools == nil {
				return "-"
			}
			return fmt.Sprintf("%.0f%% (%d/%d)", b.Tools.Accuracy(), b.Tools.Passed, b.Tools.Cases)
		})
		row("  valid / right tool / schema", func(b BackendResult) string {
			if b.Tools == nil {
				return "-"
			}
			return fmt.Sprintf("%d / %d / %d of %d", b.Tools.ValidCalls, b.Tools.RightTool, b.Tools.SchemaValid, b.Tools.CallCases)
		})
		row("  no-call correct", func(b BackendResult) string {
			if b.Tools == nil {
				return "-"
			}
			return fmt.Sprintf("%d of %d", b.Tools.NoCallCorrect, b.Tools.NoCallCases)
		})
		row("  leaks / http errors", func(b BackendResult) string {
			if b.Tools == nil {
				return "-"
			}
			return fmt.Sprintf("%d / %d", b.Tools.Leaks, b.Tools.HTTPErrors)
		})
	}
	if anyContext(r) {
		row("needle found", func(b BackendResult) string {
			f, t := b.contextFound()
			if t == 0 {
				return "-"
			}
			return fmt.Sprintf("%d of %d", f, t)
		})
	}
	row("failures", func(b BackendResult) string { return fmt.Sprint(len(b.allFailures())) })
	tw.Flush()
	for _, b := range r.Backends {
		if b.Error != "" {
			fmt.Fprintf(w, "\n%s did not run: %s\n", b.Name, b.Error)
		}
	}
}

type cell struct {
	ctx  string
	conc int
}

// cells is every (context, concurrency) any backend measured, in order.
func cells(r Results) []cell {
	seen := map[cell]bool{}
	var out []cell
	for _, b := range r.Backends {
		for _, s := range b.Speed {
			c := cell{s.Context, s.Concurrency}
			if !seen[c] {
				seen[c] = true
				out = append(out, c)
			}
		}
	}
	return out
}

func statStr(s *SpeedRow, pick func(*SpeedRow) Stat, format string) string {
	if s == nil {
		return "-"
	}
	st := pick(s)
	if st.N == 0 {
		if len(s.Failures) > 0 {
			return "skipped"
		}
		return "-"
	}
	out := fmt.Sprintf(format, st.Median)
	if st.N > 1 {
		out += fmt.Sprintf(" ("+format+"–"+format+")", st.Min, st.Max)
	}
	return out
}

func anyTools(r Results) bool {
	for _, b := range r.Backends {
		if b.Tools != nil {
			return true
		}
	}
	return false
}

func anyContext(r Results) bool {
	for _, b := range r.Backends {
		if len(b.Context) > 0 {
			return true
		}
	}
	return false
}

// Verdict is the short reading of the numbers, careful not to call a
// difference that is inside the runs' own spread.
func Verdict(r Results) []string {
	var ran []BackendResult
	for _, b := range r.Backends {
		if b.Error == "" {
			ran = append(ran, b)
		}
	}
	var out []string
	if len(ran) == 0 {
		return []string{"no backend ran"}
	}
	cs := cells(r)
	if len(cs) > 0 {
		// Fastest single agent: concurrency 1 at the smallest context.
		small := cs[0].ctx
		best, second := rank(ran, func(b BackendResult) Stat {
			if s := b.speedAt(small, 1); s != nil {
				return s.DecodeTPS
			}
			return Stat{}
		})
		if best != nil {
			line := fmt.Sprintf("fastest single agent (%s context, 1 stream): %s at %.1f tok/s", small, best.name, best.stat.Median)
			if second != nil {
				if WithinNoise(best.stat, second.stat) {
					line += fmt.Sprintf("; %s at %.1f is within noise", second.name, second.stat.Median)
				} else {
					line += fmt.Sprintf(", %.2f× %s (%.1f)", best.stat.Median/second.stat.Median, second.name, second.stat.Median)
				}
			}
			out = append(out, line)
		}
		// Most combined output at the top concurrency, largest context measured.
		top := cs[len(cs)-1]
		best, second = rank(ran, func(b BackendResult) Stat {
			if s := b.speedAt(top.ctx, top.conc); s != nil {
				return s.CombinedTPS
			}
			return Stat{}
		})
		if best != nil && top.conc > 1 {
			line := fmt.Sprintf("most combined output (%s context, %d streams): %s at %.1f tok/s", top.ctx, top.conc, best.name, best.stat.Median)
			if second != nil {
				if WithinNoise(best.stat, second.stat) {
					line += fmt.Sprintf("; %s at %.1f is within noise", second.name, second.stat.Median)
				} else {
					line += fmt.Sprintf(", %.2f× %s (%.1f)", best.stat.Median/second.stat.Median, second.name, second.stat.Median)
				}
			}
			out = append(out, line)
		}
	}
	if anyTools(r) {
		var parts []string
		for _, b := range ran {
			if b.Tools != nil {
				parts = append(parts, fmt.Sprintf("%s %.0f%% (%d/%d)", b.Name, b.Tools.Accuracy(), b.Tools.Passed, b.Tools.Cases))
			}
		}
		out = append(out, "tool accuracy: "+strings.Join(parts, ", "))
	}
	if anyContext(r) {
		var parts []string
		for _, b := range ran {
			f, t := b.contextFound()
			if t > 0 {
				parts = append(parts, fmt.Sprintf("%s %d/%d", b.Name, f, t))
			}
		}
		if len(parts) > 0 {
			out = append(out, "needle retrieved: "+strings.Join(parts, ", "))
		}
	}
	// A backend whose own decode figure disagrees with the client's by a
	// fifth or more streams in bursts: say so, since both numbers are in the
	// table and the reader has to pick.
	for _, b := range ran {
		for _, s := range b.Speed {
			if s.Concurrency != 1 || s.DecodeTPS.N == 0 || s.ServerDecode.N == 0 {
				continue
			}
			c, sv := s.DecodeTPS.Median, s.ServerDecode.Median
			if math.Abs(c-sv) > 0.2*math.Max(c, sv) {
				out = append(out, fmt.Sprintf("%s: client-side decode %.1f vs server-reported %.1f tok/s at %s×1 (%.1f tokens per stream chunk); the server's figure is the backend's own, the client's is what a stream consumer sees",
					b.Name, c, sv, s.Context, s.TokensPerChunk.Median))
			}
			break
		}
	}
	for _, b := range r.Backends {
		if b.Error != "" {
			out = append(out, fmt.Sprintf("%s did not run: %s", b.Name, b.Error))
		} else if f := b.allFailures(); len(f) > 0 {
			out = append(out, fmt.Sprintf("%s: %d failure(s), first: %s", b.Name, len(f), f[0]))
		}
	}
	if r.Options.Runs < 2 {
		out = append(out, "single run per cell: no spread to judge noise by, so treat differences under ~10% as unproven")
	}
	return out
}

type ranked struct {
	name string
	stat Stat
}

// rank orders backends by a stat's median, returning the best and second.
func rank(bs []BackendResult, pick func(BackendResult) Stat) (*ranked, *ranked) {
	var rs []ranked
	for _, b := range bs {
		if s := pick(b); s.N > 0 {
			rs = append(rs, ranked{b.Name, s})
		}
	}
	sort.Slice(rs, func(i, j int) bool { return rs[i].stat.Median > rs[j].stat.Median })
	switch len(rs) {
	case 0:
		return nil, nil
	case 1:
		return &rs[0], nil
	}
	return &rs[0], &rs[1]
}

// Markdown is report.md.
func Markdown(r Results) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# linfer bench: %s\n\n", r.Model)
	h := r.Machine
	fmt.Fprintf(&b, "%s · %s/%s, %s, %d CPUs, %s RAM", r.Started.Format("2006-01-02 15:04"), h.OS, h.Arch, h.Chip, h.CPUs, linfer.HumanBytes(h.RAM))
	if h.GPUMem > 0 {
		fmt.Fprintf(&b, ", GPU %s %s usable (%s)", h.GPU, linfer.HumanBytes(h.GPUMem), h.GPUMemSource)
	}
	fmt.Fprintf(&b, " · %s\n\n", time.Duration(r.Duration*float64(time.Second)).Round(time.Second))
	fmt.Fprintln(&b, "## Verdict")
	fmt.Fprintln(&b)
	for _, v := range Verdict(r) {
		fmt.Fprintf(&b, "- %s\n", v)
	}
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "## Backends")
	fmt.Fprintln(&b)
	for _, bk := range r.Backends {
		if bk.Error != "" {
			fmt.Fprintf(&b, "- **%s**: did not run: %s\n", bk.Name, bk.Error)
			continue
		}
		fmt.Fprintf(&b, "- **%s** (%s): `%s`, %.1f GiB; loaded in %.0f s, RSS %s; ", bk.Name, bk.Backend, bk.Weights, bk.WeightGB, bk.LoadS, linfer.HumanBytes(bk.RSS))
		if bk.Backend == linfer.BackendMLX {
			fmt.Fprintf(&b, "%d concurrent requests, window %d\n", bk.Plan.MaxConcurrent, bk.Plan.ContextWindow)
		} else {
			fmt.Fprintf(&b, "%d slots × %d tokens, cache-ram %d MB\n", bk.Plan.Slots, bk.Plan.Context, bk.Plan.CacheRAMMB)
		}
	}
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "## Comparison")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "```")
	WriteTable(&b, r)
	fmt.Fprintln(&b, "```")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "Each cell is two passes over new prompts sent all at once: a cold pass (one token each) gives time to first token and prefill with every prompt arriving together; after a short settle, a warm pass on the now-cached prompts gives decode, combined tok/s (all streams' tokens over the wall time of the pass) and, for one stream, the warm time to first token. The warm pass is the shape of an agent's turn: a cached history and a long reply. Decode and prefill are measured on the client from the stream. Medians, with min–max in brackets when there was more than one run.")
	for _, bk := range r.Backends {
		if bk.Tools != nil && len(bk.Tools.Failures) > 0 {
			fmt.Fprintf(&b, "\n### %s: tool failures\n\n", bk.Name)
			for _, f := range bk.Tools.Failures {
				fmt.Fprintf(&b, "- %s\n", f)
			}
		}
		if f := bk.allFailures(); len(f) > 0 {
			fmt.Fprintf(&b, "\n### %s: errors\n\n", bk.Name)
			for _, x := range f {
				fmt.Fprintf(&b, "- %s\n", x)
			}
		}
		if n := serverTimings(bk); n != "" {
			fmt.Fprintf(&b, "\n### %s: server-reported timings\n\n%s", bk.Name, n)
		}
	}
	if len(r.Notes) > 0 {
		fmt.Fprintln(&b, "\n## Notes")
		fmt.Fprintln(&b)
		for _, n := range r.Notes {
			fmt.Fprintf(&b, "- %s\n", n)
		}
	}
	return b.String()
}

func serverTimings(b BackendResult) string {
	var out strings.Builder
	for _, s := range b.Speed {
		if len(s.Server) == 0 {
			continue
		}
		var ks []string
		for k := range s.Server {
			ks = append(ks, k)
		}
		sort.Strings(ks)
		fmt.Fprintf(&out, "- %s×%d:", s.Context, s.Concurrency)
		for _, k := range ks {
			fmt.Fprintf(&out, " %s=%.2f", k, s.Server[k])
		}
		fmt.Fprintln(&out)
	}
	return out.String()
}
