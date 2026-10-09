package bench

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/sudiptadeb/linfer/internal/linfer"
)

// Options is what `linfer bench` was asked for.
type Options struct {
	Backends    []string `json:"backends"` // variant names, or "all"
	Suites      []string `json:"suites"`   // speed, tools, context
	Concurrency []int    `json:"concurrency"`
	Contexts    []int    `json:"contexts"` // prompt tokens
	Quick       bool     `json:"quick"`
	Out         string   `json:"out"`
	// Filled from Quick when zero.
	Runs         int   `json:"runs"`          // per speed cell
	DecodeTokens int   `json:"decode_tokens"` // per stream
	ToolRuns     int   `json:"tool_runs"`     // passes over the tool suite
	Depths       []int `json:"depths"`        // needle depths, percent
}

// Defaults fills what was not given. Quick is a few minutes on a big model:
// one run, two concurrency levels, one context, one pass of the tools.
func (o *Options) Defaults() {
	if len(o.Backends) == 0 {
		o.Backends = []string{"all"}
	}
	if len(o.Suites) == 0 {
		o.Suites = []string{"speed", "tools", "context"}
	}
	if o.Quick {
		setIf(&o.Concurrency, []int{1, 4})
		setIf(&o.Contexts, []int{2048})
		setInt(&o.Runs, 1)
		setInt(&o.DecodeTokens, 256)
		setInt(&o.ToolRuns, 1)
		setIf(&o.Depths, []int{50})
	} else {
		setIf(&o.Concurrency, []int{1, 2, 4, 8})
		setIf(&o.Contexts, []int{2048, 32768})
		setInt(&o.Runs, 3)
		setInt(&o.DecodeTokens, 512)
		setInt(&o.ToolRuns, 3)
		setIf(&o.Depths, []int{10, 50, 90})
	}
}

func setIf(p *[]int, v []int) {
	if len(*p) == 0 {
		*p = v
	}
}

func setInt(p *int, v int) {
	if *p == 0 {
		*p = v
	}
}

func (o Options) has(suite string) bool {
	for _, s := range o.Suites {
		if s == suite {
			return true
		}
	}
	return false
}

// ParseList reads "1,2,4,8" or "2k,32k" (k is 1024).
func ParseList(s string) ([]int, error) {
	if strings.TrimSpace(s) == "" {
		return nil, nil
	}
	var out []int
	for _, f := range strings.Split(s, ",") {
		f = strings.TrimSpace(strings.ToLower(f))
		mult := 1
		if strings.HasSuffix(f, "k") {
			mult, f = 1024, strings.TrimSuffix(f, "k")
		}
		n, err := strconv.Atoi(f)
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("%q: want positive numbers such as 1,2,4 or 2k,32k", s)
		}
		out = append(out, n*mult)
	}
	return out, nil
}

// Variant is one row of the comparison: a backend, and later perhaps a
// backend with a twist (a speculative decoder, another quantisation), which
// is why it is a name plus a change to the config rather than a backend
// string.
type Variant struct {
	Name    string
	Backend string
	// Tweak adjusts the config for this variant; nil for the plain backend.
	Tweak func(linfer.Config) linfer.Config
}

func (v Variant) config(cfg linfer.Config) linfer.Config {
	cfg.Backend = v.Backend
	if v.Tweak != nil {
		cfg = v.Tweak(cfg)
	}
	return cfg
}

// Variants is every backend this machine can run for the configured model:
// llama when there are GGUF weights, mlx on Apple Silicon when the MLX
// weights and omlx exist. The configured backend does not narrow it: a
// config pinned to llama is still compared against mlx.
func Variants(cfg linfer.Config, hw linfer.Hardware) []Variant {
	var out []Variant
	if cfg.Model.GGUF != "" && linfer.FileExists(cfg.GGUFPath()) && linfer.FileExists(cfg.LlamaBinPath()) {
		out = append(out, Variant{Name: "llama", Backend: linfer.BackendLlama})
	}
	unpinned := cfg
	unpinned.Backend = linfer.BackendAuto
	if linfer.WantsMLX(unpinned, hw) && linfer.FileExists(filepath.Join(cfg.MLXPath(), "config.json")) && linfer.FileExists(cfg.OMLXBinPath()) {
		out = append(out, Variant{Name: "mlx", Backend: linfer.BackendMLX})
	}
	return out
}

// Running is a launched backend as the runner sees it.
type Running struct {
	Plan   linfer.Plan
	RSS    func() uint64
	Exited func() (bool, string)
	Stop   func()
}

// Runner runs the comparison. The launch is behind three functions so the
// tests can run a fake backend; in production they go through
// linfer.PlanFor and linfer.Start, exactly what serve uses.
type Runner struct {
	Cfg  linfer.Config
	HW   linfer.Hardware
	Opts Options
	Out  io.Writer
	// Plan sizes the variant: its plan and which weights it will load.
	Plan func(v Variant) (linfer.Plan, string, error)
	// Start launches it and waits until it answers.
	Start func(ctx context.Context, v Variant, p linfer.Plan) (*Running, error)
	// Daemon is the serve daemon for this config, paused for the duration
	// when it is running; nil means do not look.
	Daemon *linfer.Client
	// FreeMemory is the memory free now, for the check before each launch;
	// zero means unknown.
	FreeMemory func() uint64
	// HealthTimeout bounds a backend's load.
	HealthTimeout time.Duration
	// MemoryWait is how long to wait for memory to come free before a
	// launch. macOS hands back a stopped model's memory over some seconds,
	// so a check made just after pausing the daemon or stopping the
	// previous backend reads low.
	MemoryWait time.Duration
	// Settle is the speed suite's pause between its cold and warm passes,
	// for the backend to commit the prompts to its cache; zero means 2 s.
	Settle time.Duration
}

func (r *Runner) settle() time.Duration {
	if r.Settle > 0 {
		return r.Settle
	}
	return 2 * time.Second
}

// New is the production runner.
func New(cfg linfer.Config, hw linfer.Hardware, opts Options, out io.Writer) *Runner {
	r := &Runner{Cfg: cfg, HW: hw, Opts: opts, Out: out, HealthTimeout: 20 * time.Minute, MemoryWait: 2 * time.Minute}
	r.Plan = func(v Variant) (linfer.Plan, string, error) {
		c := v.config(cfg)
		plan, _, err := linfer.PlanFor(c, hw, v.Backend)
		weights := c.GGUFPath()
		if v.Backend == linfer.BackendMLX {
			weights = c.MLXPath()
		}
		return plan, weights, err
	}
	r.Start = func(ctx context.Context, v Variant, plan linfer.Plan) (*Running, error) {
		c := v.config(cfg)
		_, cmd, err := linfer.PlanFor(c, hw, v.Backend)
		if err != nil {
			return nil, err
		}
		p, err := linfer.Start(cmd, filepath.Join(c.Paths().Root, "bench-backend.log"))
		if err != nil {
			return nil, err
		}
		if err := p.WaitHealthy(ctx, r.HealthTimeout); err != nil {
			p.Stop(10 * time.Second)
			return nil, err
		}
		return &Running{Plan: plan, RSS: p.RSS, Exited: p.Exited, Stop: func() { p.Stop(30 * time.Second) }}, nil
	}
	r.Daemon = &linfer.Client{Socket: cfg.Paths().Socket}
	probe := linfer.SystemProbe()
	r.FreeMemory = func() uint64 { return linfer.FreeMemory(probe) }
	return r
}

func (r *Runner) say(format string, args ...any) {
	fmt.Fprintf(r.Out, format+"\n", args...)
}

// Run does the whole comparison and writes the report.
func (r *Runner) Run(ctx context.Context) (Results, error) {
	r.Opts.Defaults()
	res := Results{Started: time.Now(), Machine: r.HW, Model: r.Cfg.Model.ID, Options: r.Opts}
	variants, err := r.variants()
	if err != nil {
		return res, err
	}
	r.announce(variants)

	// Pause the daemon for the duration; resume whatever happens, on a
	// context of its own because the run's may have been cancelled.
	if r.Daemon != nil {
		st, err := r.Daemon.Status(ctx)
		switch {
		case err != nil:
			// No daemon for this config: nothing to pause.
		case st.Paused:
			r.say("the linfer daemon is already paused; it is left paused")
			res.Notes = append(res.Notes, "the serve daemon was already paused and was left so")
		default:
			r.say("pausing the linfer daemon (%s) for the duration …", st.Backend)
			if _, err := r.Daemon.Pause(ctx); err != nil {
				return res, fmt.Errorf("pause the daemon: %w", err)
			}
			res.Notes = append(res.Notes, "the serve daemon was paused for the run and resumed after")
			defer func() {
				rctx, cancel := context.WithTimeout(context.Background(), time.Minute)
				defer cancel()
				if _, err := r.Daemon.Resume(rctx); err != nil {
					r.say("WARNING: could not resume the daemon: %v (run `linfer resume`)", err)
				} else {
					r.say("linfer daemon resumed")
				}
			}()
		}
	}

	for _, v := range variants {
		if ctx.Err() != nil {
			break
		}
		res.Backends = append(res.Backends, r.runVariant(ctx, v))
	}
	res.Duration = time.Since(res.Started).Seconds()
	if err := r.write(res); err != nil {
		return res, err
	}
	fmt.Fprintln(r.Out)
	WriteTable(r.Out, res)
	fmt.Fprintln(r.Out)
	for _, line := range Verdict(res) {
		r.say("- %s", line)
	}
	r.say("\nreport: %s", filepath.Join(r.Opts.Out, "report.md"))
	if ctx.Err() != nil {
		return res, ctx.Err()
	}
	return res, nil
}

func (r *Runner) variants() ([]Variant, error) {
	all := Variants(r.Cfg, r.HW)
	if len(all) == 0 {
		return nil, errors.New("no backend can run this model here: run `linfer doctor`")
	}
	if len(r.Opts.Backends) == 1 && r.Opts.Backends[0] == "all" {
		return all, nil
	}
	var out []Variant
	for _, name := range r.Opts.Backends {
		found := false
		for _, v := range all {
			if v.Name == name {
				out = append(out, v)
				found = true
			}
		}
		if !found {
			var names []string
			for _, v := range all {
				names = append(names, v.Name)
			}
			return nil, fmt.Errorf("backend %q is not available here; available: %s", name, strings.Join(names, ", "))
		}
	}
	return out, nil
}

// announce says what will run and roughly how long, before anything loads.
func (r *Runner) announce(variants []Variant) {
	o := r.Opts
	requests, seconds := 0, 0.0
	if o.has("speed") {
		for _, c := range o.Contexts {
			for _, conc := range o.Concurrency {
				requests += 2 * o.Runs * conc // a cold and a warm pass
				// A rough cost model: prefill at 500 tok/s, the settle,
				// decode at 30 tok/s per stream, concurrent streams sharing
				// the decode.
				seconds += float64(o.Runs) * (float64(c)/500*float64(conc) + r.settle().Seconds() + float64(o.DecodeTokens)/30*1.5)
			}
		}
	}
	if o.has("tools") {
		requests += len(Cases) * o.ToolRuns
		seconds += float64(len(Cases)*o.ToolRuns) * 8
	}
	if o.has("context") {
		for _, c := range o.Contexts {
			requests += len(o.Depths)
			seconds += float64(len(o.Depths)) * (float64(c)/500 + 5)
		}
	}
	var names []string
	for _, v := range variants {
		names = append(names, v.Name)
	}
	r.say("bench: %s on %s", r.Cfg.Model.ID, strings.Join(names, ", "))
	r.say("  suites %s; contexts %v tokens; concurrency %v; %d run(s) per cell; %d pass(es) of %d tool cases",
		strings.Join(o.Suites, ","), o.Contexts, o.Concurrency, o.Runs, o.ToolRuns, len(Cases))
	est := time.Duration(seconds*float64(len(variants))) * time.Second
	r.say("  about %d requests per backend; rough estimate %s plus loading each model", requests, est.Round(time.Minute))
}

// runVariant loads one backend, runs the suites, stops it and waits for the
// exit, so the next one finds the memory free.
func (r *Runner) runVariant(ctx context.Context, v Variant) BackendResult {
	br := BackendResult{Name: v.Name, Backend: v.Backend}
	r.say("\n== %s", v.Name)
	plan, weights, err := r.Plan(v)
	if err != nil {
		br.Error = err.Error()
		r.say("  %s", br.Error)
		return br
	}
	br.Plan, br.Weights, br.WeightGB = plan, weights, float64(plan.Weights)/linfer.GiB
	if r.FreeMemory != nil {
		free := r.FreeMemory()
		for deadline := time.Now().Add(r.MemoryWait); free > 0 && plan.Weights > free && time.Now().Before(deadline); free = r.FreeMemory() {
			select {
			case <-ctx.Done():
				br.Error = ctx.Err().Error()
				return br
			case <-time.After(2 * time.Second):
			}
		}
		if free > 0 && plan.Weights > free {
			br.Error = fmt.Sprintf("%s of weights but only %s of memory free; stop what holds it first", linfer.HumanBytes(plan.Weights), linfer.HumanBytes(free))
			r.say("  %s", br.Error)
			return br
		}
	}
	if answers(r.Cfg.URL() + "/models") {
		br.Error = "something already answers on " + r.Cfg.Listen + "; stop it first"
		r.say("  %s", br.Error)
		return br
	}
	r.say("  loading %s (%.1f GiB) …", weights, br.WeightGB)
	t0 := time.Now()
	run, err := r.Start(ctx, v, plan)
	if err != nil {
		br.Error = err.Error()
		r.say("  %s", br.Error)
		return br
	}
	br.LoadS = time.Since(t0).Seconds()
	defer func() {
		r.say("  stopping %s …", v.Name)
		run.Stop()
		r.say("  stopped")
	}()
	br.RSS = run.RSS()
	r.say("  up in %.0fs, rss %s", br.LoadS, linfer.HumanBytes(br.RSS))

	cl := newClient(r.Cfg.URL(), r.Cfg.Model.ID)
	// Warm-up: the first request after a load pays for buffers and compiles.
	if st := cl.Complete(ctx, []Message{{Role: "user", Content: "Say hello."}}, 8); st.Err != "" {
		br.Failures = append(br.Failures, "warm-up: "+st.Err)
	}
	cpt, err := calibrate(ctx, cl)
	if err != nil {
		br.Failures = append(br.Failures, err.Error())
		cpt = 3.8
	}
	maxCtx := plan.Context
	if v.Backend == linfer.BackendMLX {
		maxCtx = plan.ContextWindow
	}
	progress := func(s string) { r.say("%s", s) }
	if r.Opts.has("speed") && ctx.Err() == nil {
		r.say("  speed")
		br.Speed = runSpeed(ctx, cl, speedPlan{
			contexts: r.Opts.Contexts, concurrency: r.Opts.Concurrency, runs: r.Opts.Runs,
			decodeTokens: r.Opts.DecodeTokens, charsPerToken: cpt, maxContext: maxCtx, settle: r.settle(),
		}, progress)
	}
	if r.Opts.has("tools") && ctx.Err() == nil {
		r.say("  tools")
		ts := runTools(ctx, cl, r.Opts.ToolRuns, progress)
		br.Tools = &ts
	}
	if r.Opts.has("context") && ctx.Err() == nil {
		r.say("  context")
		br.Context = runContext(ctx, cl, r.Opts.Contexts, r.Opts.Depths, cpt, maxCtx, progress)
	}
	if exited, why := run.Exited(); exited {
		br.Failures = append(br.Failures, "the backend exited during the run: "+why)
	}
	br.RSS = max(br.RSS, run.RSS())
	return br
}

func answers(url string) bool {
	c := &http.Client{Timeout: time.Second}
	resp, err := c.Get(url)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return true
}

// write puts results.json and report.md in Out.
func (r *Runner) write(res Results) error {
	if r.Opts.Out == "" {
		r.Opts.Out = filepath.Join(r.Cfg.Paths().Root, "bench", res.Started.Format("2006-01-02-150405"))
		res.Options.Out = r.Opts.Out
	}
	if err := os.MkdirAll(r.Opts.Out, 0o755); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(r.Opts.Out, "results.json"), raw, 0o644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(r.Opts.Out, "report.md"), []byte(Markdown(res)), 0o644)
}
