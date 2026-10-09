// Package bench is `linfer bench`: it compares the backends this machine can
// run for the configured model, one at a time, each launched with exactly
// the profile serve would use, and hands each one's URL to linfer-bench,
// which does the measuring, the scoring and the reports. This package only
// does what needs linfer: which backends exist, pausing the serve daemon,
// waiting for the memory, starting and stopping the backend, and the
// launch details each row carries.
package bench

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	lb "github.com/sudiptadeb/linfer-bench/bench"
	"github.com/sudiptadeb/linfer/internal/linfer"
)

// Options is what `linfer bench` was asked for: which backends, the tags
// for the run, where the report goes, and linfer-bench's own options.
type Options struct {
	Backends []string // variant names, or "all"
	Tags     []string
	Out      string
	// Store is the linfer-bench results store; NoStore skips it.
	Store   string
	NoStore bool
	Bench   lb.Options
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
	// Color paints the terminal table.
	Color bool
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

// Machine is the bench's view of the hardware linfer detected.
func Machine(hw linfer.Hardware) lb.Machine {
	gpu := hw.GPU
	if hw.GPUName != "" {
		gpu += " " + hw.GPUName
	}
	return lb.Machine{OS: hw.OS, Arch: hw.Arch, Chip: hw.Chip, CPUs: hw.CPUs, RAM: hw.RAM, GPU: gpu, GPUMem: hw.GPUMem, GPUMemSource: hw.GPUMemSource}
}

// Run does the whole comparison and writes the report.
func (r *Runner) Run(ctx context.Context) (lb.Results, error) {
	r.Opts.Bench.Defaults()
	if len(r.Opts.Backends) == 0 {
		r.Opts.Backends = []string{"all"}
	}
	res := lb.Results{Schema: lb.SchemaVersion, Tool: "linfer bench (linfer-bench " + lb.Version + ")", Started: time.Now(),
		Title: r.Cfg.Model.ID, Machine: Machine(r.HW), Options: r.Opts.Bench, Tags: r.Opts.Tags}
	res.ID = lb.NewID(res.Started, res.Title)
	variants, err := r.variants()
	if err != nil {
		return res, err
	}
	suite, err := r.Opts.Bench.Suite()
	if err != nil {
		return res, fmt.Errorf("cases: %w", err)
	}
	r.announce(variants, len(suite.Cases), suite.Name)

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
		res.Rows = append(res.Rows, r.runVariant(ctx, v))
	}
	res.Duration = time.Since(res.Started).Seconds()
	if r.Opts.Out == "" {
		r.Opts.Out = filepath.Join(r.Cfg.Paths().Root, "bench", res.Started.Format("2006-01-02-150405"))
	}
	files, err := lb.WriteFiles(r.Opts.Out, res)
	if err != nil {
		return res, err
	}
	fmt.Fprintln(r.Out)
	lb.Table(r.Out, res, r.Color)
	fmt.Fprintln(r.Out)
	for _, line := range lb.Verdict(res) {
		r.say("- %s", line)
	}
	r.say("\nreport: %s", files[2])
	if !r.Opts.NoStore {
		if _, err := (lb.Store{Dir: r.Opts.Store}).Put(res); err != nil {
			r.say("WARNING: could not store the run: %v", err)
		} else {
			r.say("stored as run %s (linfer-bench runs; linfer-bench compare --tags backend:llama,backend:mlx)", res.ID)
		}
	}
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
func (r *Runner) announce(variants []Variant, cases int, suite string) {
	o := r.Opts.Bench
	var names []string
	for _, v := range variants {
		names = append(names, v.Name)
	}
	requests, est := lb.Estimate(o, cases)
	r.say("bench: %s on %s", r.Cfg.Model.ID, strings.Join(names, ", "))
	r.say("  %s profile; suites %s; contexts %v tokens; concurrency %v; %d run(s) per cell; %d-token replies; %d pass(es) of %d tool cases (%s)",
		o.Profile, strings.Join(o.Suites, ","), []int(o.Contexts), o.Concurrency, o.Runs, o.ReplyTokens, o.ToolPasses, cases, suite)
	total := est * time.Duration(len(variants))
	if total > 2*time.Minute {
		total = total.Round(time.Minute)
	} else {
		total = total.Round(time.Second)
	}
	r.say("  about %d requests per backend; rough estimate %s plus loading each model", requests, total)
	if len(r.Opts.Tags) > 0 {
		r.say("  tags %s", strings.Join(r.Opts.Tags, ", "))
	}
}

// runVariant loads one backend, runs the suites through linfer-bench, stops
// it and waits for the exit, so the next one finds the memory free.
func (r *Runner) runVariant(ctx context.Context, v Variant) lb.Row {
	row := lb.Row{Label: v.Name, Target: v.Name, URL: r.Cfg.URL(), Model: r.Cfg.Model.ID, Tags: []string{"backend:" + v.Name}, Started: time.Now()}
	r.say("\n== %s", v.Name)
	plan, weights, err := r.Plan(v)
	if err != nil {
		row.Error = err.Error()
		r.say("  %s", row.Error)
		return row
	}
	row.Info = []lb.KV{{Key: "weights", Value: fmt.Sprintf("%.1f GiB", float64(plan.Weights)/linfer.GiB)}, {Key: "launch", Value: launch(v, plan)}}
	if r.FreeMemory != nil {
		free := r.FreeMemory()
		for deadline := time.Now().Add(r.MemoryWait); free > 0 && plan.Weights > free && time.Now().Before(deadline); free = r.FreeMemory() {
			select {
			case <-ctx.Done():
				row.Error = ctx.Err().Error()
				return row
			case <-time.After(2 * time.Second):
			}
		}
		if free > 0 && plan.Weights > free {
			row.Error = fmt.Sprintf("%s of weights but only %s of memory free; stop what holds it first", linfer.HumanBytes(plan.Weights), linfer.HumanBytes(free))
			r.say("  %s", row.Error)
			return row
		}
	}
	if answers(r.Cfg.URL() + "/models") {
		row.Error = "something already answers on " + r.Cfg.Listen + "; stop it first"
		r.say("  %s", row.Error)
		return row
	}
	r.say("  loading %s (%s) …", weights, row.Info[0].Value)
	t0 := time.Now()
	run, err := r.Start(ctx, v, plan)
	if err != nil {
		row.Error = err.Error()
		r.say("  %s", row.Error)
		return row
	}
	loadS := time.Since(t0).Seconds()
	defer func() {
		r.say("  stopping %s …", v.Name)
		run.Stop()
		r.say("  stopped")
	}()
	rss := run.RSS()
	r.say("  up in %.0fs, rss %s", loadS, linfer.HumanBytes(rss))

	maxCtx := plan.Context
	if v.Backend == linfer.BackendMLX {
		maxCtx = plan.ContextWindow
	}
	pair := lb.Pair{Target: lb.Target{Name: v.Name, URL: r.Cfg.URL(), MaxContext: maxCtx}, Model: r.Cfg.Model.ID, Label: v.Name}
	measured := lb.RunRow(ctx, pair, r.Opts.Bench, func(e lb.Event) {
		switch {
		case e.Suite == "":
			r.say("  %s", e.Text)
		case e.Text == e.Suite || strings.HasPrefix(e.Text, e.Suite+" "):
			r.say("  %s", e.Text)
		default:
			r.say("    %s", e.Text)
		}
	})
	// Keep what this side knows about the row: the weights and launch, the
	// load time and memory, the backend tag.
	measured.Label, measured.Tags = row.Label, row.Tags
	measured.Info = append(row.Info, lb.KV{Key: "load", Value: fmt.Sprintf("%.0fs, rss %s", loadS, linfer.HumanBytes(max(rss, run.RSS())))})
	if exited, why := run.Exited(); exited {
		measured.Failures = append(measured.Failures, "the backend exited during the run: "+why)
	}
	return measured
}

// launch is the one-line launch profile: slots and context for llama,
// concurrency and window for mlx.
func launch(v Variant, p linfer.Plan) string {
	if v.Backend == linfer.BackendMLX {
		return fmt.Sprintf("%d concurrent, window %d", p.MaxConcurrent, p.ContextWindow)
	}
	return fmt.Sprintf("%d×%d, cache %d MB", p.Slots, p.Context, p.CacheRAMMB)
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
