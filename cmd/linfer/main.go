// linfer sets up and runs the best local inference backend for one model on
// this machine, and tells the client the URL.
//
//	linfer setup              install the backend, fetch the weights, size the launch
//	linfer doctor [--json]    report the machine, the config and the plan; change nothing
//	linfer serve              run the backend under supervision (the service)
//	linfer status [--json]    backend, url, model, pid, healthy, rss
//	linfer pause              stop the backend and free its memory; the daemon stays up
//	linfer resume             start it again
//	linfer switch llama|mlx|auto
//	linfer bench [--backends all|llama,mlx] [--suites speed,tools,context]
//	             [--contexts 2k,32k] [--concurrency 1,2,4,8] [--quick|--full]
//	             [--cases FILE] [-f bench.yaml] [--tag T]... [--out DIR]
//	                          compare every backend this machine can run for the model
//
// -config names the file; the default is ~/.config/linfer/linfer.yaml.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	lb "github.com/sudiptadeb/linfer-bench/bench"
	"github.com/sudiptadeb/linfer/internal/bench"
	"github.com/sudiptadeb/linfer/internal/linfer"
)

func main() {
	fs := flag.NewFlagSet("linfer", flag.ContinueOnError)
	cfgPath := fs.String("config", linfer.DefaultConfigPath(), "config file")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: linfer [-config FILE] setup|doctor|serve|status|pause|resume|switch BACKEND|bench [flags]")
		fs.PrintDefaults()
	}
	if err := fs.Parse(os.Args[1:]); err != nil {
		os.Exit(2)
	}
	args := fs.Args()
	if len(args) == 0 {
		fs.Usage()
		os.Exit(2)
	}
	if err := run(*cfgPath, args[0], args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "linfer:", err)
		os.Exit(1)
	}
}

func run(cfgPath, cmd string, args []string) error {
	cfg, err := linfer.Load(cfgPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("no config at %s; write one (see the README's example) or pass -config", cfgPath)
		}
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	asJSON := len(args) > 0 && args[0] == "--json"

	switch cmd {
	case "setup":
		r, err := linfer.Setup(ctx, cfg, linfer.NewFetcher().WithProgress(os.Stderr), os.Stdout)
		if err != nil {
			return err
		}
		fmt.Println()
		linfer.PrintReport(os.Stdout, r)
		fmt.Printf("\nstart it: linfer -config %s serve\n", cfgPath)
		return nil
	case "doctor":
		r := linfer.Doctor(cfg)
		if asJSON {
			return printJSON(r)
		}
		linfer.PrintHardware(os.Stdout, r.Hardware)
		linfer.PrintReport(os.Stdout, r)
		if len(r.Problems) > 0 {
			return fmt.Errorf("%d problem(s); `linfer setup` fixes what it can", len(r.Problems))
		}
		return nil
	case "serve":
		return serve(ctx, cfg)
	case "bench":
		return runBench(ctx, cfg, args)
	}

	// The rest talk to a running daemon.
	cl := linfer.Client{Socket: cfg.Paths().Socket}
	var st linfer.Status
	switch cmd {
	case "status":
		st, err = cl.Status(ctx)
	case "pause":
		st, err = cl.Pause(ctx)
	case "resume":
		st, err = cl.Resume(ctx)
	case "switch":
		if len(args) != 1 {
			return fmt.Errorf("usage: linfer switch llama|mlx|auto")
		}
		st, err = cl.Switch(ctx, args[0])
	default:
		return fmt.Errorf("unknown command %q", cmd)
	}
	if err != nil {
		return err
	}
	if asJSON {
		return printJSON(st)
	}
	printStatus(st)
	return nil
}

// serve runs the daemon: the supervisor, and the control socket beside it.
func serve(ctx context.Context, cfg linfer.Config) error {
	log := slog.New(slog.NewTextHandler(os.Stderr, nil))
	hw, err := linfer.Detect(linfer.SystemProbe(), cfg.Dir, cfg.GPU, linfer.MLXPython(cfg))
	if err != nil {
		return err
	}
	if err := os.MkdirAll(cfg.Dir, 0o755); err != nil {
		return err
	}
	s := linfer.NewSupervisor(cfg, hw, log)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	errc := make(chan error, 1)
	go func() { errc <- linfer.ServeControl(ctx, cfg.Paths().Socket, s) }()
	log.Info("linfer serving", "url", cfg.URL(), "model", cfg.Model.ID, "socket", cfg.Paths().Socket)
	go func() {
		if err := <-errc; err != nil {
			log.Error("control socket", "err", err)
			cancel()
		}
	}()
	return s.Run(ctx)
}

func printStatus(st linfer.Status) {
	state := "stopped"
	switch {
	case st.Paused:
		state = "paused"
	case st.Healthy:
		state = "healthy"
	case st.PID != 0:
		state = "starting"
	}
	fmt.Printf("backend   %s (%s)\n", st.Backend, state)
	fmt.Printf("url       %s\n", st.URL)
	fmt.Printf("model     %s\n", st.Model)
	if st.PID != 0 {
		fmt.Printf("pid       %d, up %s, rss %s\n", st.PID, time.Since(st.Since).Round(time.Second), linfer.HumanBytes(st.RSSBytes))
	}
	if st.Plan != nil {
		switch st.Plan.Backend {
		case linfer.BackendLlama:
			fmt.Printf("launch    %d slots × %d tokens, cache-ram %d MB\n", st.Plan.Slots, st.Plan.Context, st.Plan.CacheRAMMB)
		case linfer.BackendMLX:
			fmt.Printf("launch    %d concurrent, window %d\n", st.Plan.MaxConcurrent, st.Plan.ContextWindow)
		}
	}
	if st.Restarts > 0 {
		fmt.Printf("restarts  %d\n", st.Restarts)
	}
	if st.LastError != "" {
		fmt.Printf("last err  %s\n", st.LastError)
	}
}

// list is a repeatable string flag.
type list []string

func (l *list) String() string     { return strings.Join(*l, ",") }
func (l *list) Set(s string) error { *l = append(*l, s); return nil }

// runBench compares the backends this machine can run for the model, one at
// a time, each launched as serve would launch it, measured by linfer-bench.
func runBench(ctx context.Context, cfg linfer.Config, args []string) error {
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	backends := fs.String("backends", "all", "all, or a comma list: llama,mlx")
	file := fs.String("f", "", "a linfer-bench bench.yaml for the options (its targets are ignored: linfer starts the backends)")
	suites := fs.String("suites", "", "comma list of speed, tools, context (default all)")
	conc := fs.String("concurrency", "", "comma list of stream counts (profile default)")
	contexts := fs.String("contexts", "", "comma list of prompt sizes, k = 1024 tokens (profile default)")
	quick := fs.Bool("quick", false, "the quick profile: a few minutes")
	full := fs.Bool("full", false, "the full profile: 2k and 32k, up to 8 streams, 3 runs")
	runs := fs.Int("runs", 0, "runs per speed cell, median taken (profile default)")
	reply := fs.Int("reply-tokens", 0, "tokens per reply in the speed suite (profile default)")
	passes := fs.Int("tool-passes", 0, "passes over the tool suite (profile default)")
	cases := fs.String("cases", "", "a tool-calling cases file, YAML or JSON lines (default: the built-in suite)")
	var tags list
	fs.Var(&tags, "tag", "a tag for this run, for `linfer-bench compare --tags` (repeatable; each backend's row is also tagged backend:<name>)")
	out := fs.String("out", "", "directory for the report (default <dir>/bench/<timestamp>)")
	store := fs.String("store", lb.DefaultStoreDir(), "the linfer-bench results store")
	noStore := fs.Bool("no-store", false, "do not keep the run in the store")
	if err := fs.Parse(args); err != nil {
		return err
	}
	opts := bench.Options{Out: *out, Store: *store, NoStore: *noStore, Tags: tags}
	if *backends != "all" {
		opts.Backends = strings.Split(*backends, ",")
	}
	if *file != "" {
		f, err := lb.LoadFile(*file)
		if err != nil {
			return err
		}
		opts.Bench = f.Options
		opts.Tags = append(f.Tags, tags...)
	}
	o := &opts.Bench
	switch {
	case *quick && *full:
		return errors.New("--quick and --full are exclusive")
	case *quick:
		o.Profile = lb.ProfileQuick
	case *full:
		o.Profile = lb.ProfileFull
	}
	if *suites != "" {
		o.Suites = strings.Split(*suites, ",")
	}
	var err error
	if *conc != "" {
		if o.Concurrency, err = lb.ParseList(*conc); err != nil {
			return fmt.Errorf("--concurrency %w", err)
		}
	}
	if *contexts != "" {
		var cs []int
		if cs, err = lb.ParseList(*contexts); err != nil {
			return fmt.Errorf("--contexts %w", err)
		}
		o.Contexts = cs
	}
	for _, p := range []struct {
		dst *int
		v   int
	}{{&o.Runs, *runs}, {&o.ReplyTokens, *reply}, {&o.ToolPasses, *passes}} {
		if p.v > 0 {
			*p.dst = p.v
		}
	}
	if *cases != "" {
		o.Cases = *cases
	}
	if err := o.Validate(); err != nil {
		return err
	}
	hw, err := linfer.Detect(linfer.SystemProbe(), cfg.Dir, cfg.GPU, linfer.MLXPython(cfg))
	if err != nil {
		return err
	}
	linfer.PrintHardware(os.Stdout, hw)
	r := bench.New(cfg, hw, opts, os.Stdout)
	if st, err := os.Stdout.Stat(); err == nil && st.Mode()&os.ModeCharDevice != 0 && os.Getenv("NO_COLOR") == "" {
		r.Color = true
	}
	_, err = r.Run(ctx)
	return err
}

func printJSON(v any) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}
