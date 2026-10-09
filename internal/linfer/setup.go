package linfer

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// Report is what doctor and setup found. Problems are what stands between
// this machine and a running backend; setup fixes the ones it can.
type Report struct {
	Hardware Hardware  `json:"hardware"`
	Backend  string    `json:"backend"`
	Reason   string    `json:"reason"`
	LlamaBin string    `json:"llama_bin"`
	LlamaOK  bool      `json:"llama_ok"`
	OMLXBin  string    `json:"omlx_bin,omitempty"`
	OMLXOK   bool      `json:"omlx_ok"`
	GGUF     string    `json:"gguf,omitempty"`
	GGUFOK   bool      `json:"gguf_ok"`
	MLX      string    `json:"mlx,omitempty"`
	MLXOK    bool      `json:"mlx_ok"`
	Model    ModelInfo `json:"model"`
	Plan     Plan      `json:"plan"`
	URL      string    `json:"url"`
	ModelID  string    `json:"model_id"`
	Problems []string  `json:"problems,omitempty"`
}

// Doctor reports on the machine and the config without changing anything.
func Doctor(cfg Config) Report {
	hw, err := Detect(SystemProbe(), cfg.Dir, cfg.GPU, mlxPython(cfg))
	r := Report{Hardware: hw, URL: cfg.URL(), ModelID: cfg.Model.ID}
	if err != nil {
		r.Problems = append(r.Problems, err.Error())
		return r
	}
	r.inspect(cfg, hw)
	return r
}

// inspect fills the report from what is on disk.
func (r *Report) inspect(cfg Config, hw Hardware) {
	r.LlamaBin = cfg.LlamaBinPath()
	r.LlamaOK = FileExists(r.LlamaBin)
	if !r.LlamaOK {
		r.Problems = append(r.Problems, "no llama-server at "+r.LlamaBin)
	}
	if cfg.Model.GGUF != "" {
		r.GGUF = cfg.GGUFPath()
		r.GGUFOK = FileExists(r.GGUF) && (cfg.Model.MMProj == "" || FileExists(cfg.MMProjPath()))
		if !r.GGUFOK {
			r.Problems = append(r.Problems, "no GGUF weights at "+r.GGUF)
		}
	}
	if WantsMLX(cfg, hw) {
		r.OMLXBin = cfg.OMLXBinPath()
		r.OMLXOK = FileExists(r.OMLXBin)
		if !r.OMLXOK {
			r.Problems = append(r.Problems, "no omlx at "+r.OMLXBin)
		}
		r.MLX = cfg.MLXPath()
		r.MLXOK = FileExists(filepath.Join(r.MLX, "config.json"))
		if !r.MLXOK {
			r.Problems = append(r.Problems, "no MLX weights at "+r.MLX)
		}
	}
	r.Backend, r.Reason = Choose(cfg, hw, FileExists)
	m, err := ReadModel(cfg, r.Backend)
	if err != nil {
		r.Problems = append(r.Problems, err.Error())
		return
	}
	r.Model = m
	plan, err := Size(hw, m, r.Backend, cfg)
	r.Plan = plan
	if err != nil {
		r.Problems = append(r.Problems, err.Error())
	}
}

// Setup gets the machine to a working backend: it installs llama.cpp (and
// oMLX where it applies), fetches hf:// weights after checking they fit in
// disk and memory, sizes the launch, and writes oMLX's files. It prints as
// it goes, and returns the report for the final summary. Every step is a
// no-op when its result is already there, so re-running is cheap.
func Setup(ctx context.Context, cfg Config, f *Fetcher, out io.Writer) (Report, error) {
	say := func(format string, args ...any) { fmt.Fprintf(out, format+"\n", args...) }
	hw, err := Detect(SystemProbe(), cfg.Dir, cfg.GPU, mlxPython(cfg))
	r := Report{Hardware: hw, URL: cfg.URL(), ModelID: cfg.Model.ID}
	if err != nil {
		return r, err
	}
	PrintHardware(out, hw)
	if err := os.MkdirAll(cfg.Dir, 0o755); err != nil {
		return r, err
	}

	// llama.cpp: always, since it runs everywhere and is what auto falls
	// back to.
	if cfg.LlamaBin != "" {
		if err := CheckLlama(cfg.LlamaBin, cfg.LlamaRelease); err != nil {
			// A configured binary of another build is fine; only a missing one is not.
			if !FileExists(cfg.LlamaBin) {
				return r, fmt.Errorf("llama_bin: %w", err)
			}
			say("llama-server: %s (configured; not release %s, kept as is)", cfg.LlamaBin, cfg.LlamaRelease)
		} else {
			say("llama-server: %s (configured)", cfg.LlamaBin)
		}
	} else if FileExists(cfg.LlamaBinPath()) {
		say("llama-server: %s (installed)", cfg.LlamaBinPath())
	} else {
		say("llama-server: installing release %s for %s/%s %s …", cfg.LlamaRelease, hw.OS, hw.Arch, hw.GPU)
		bin, err := f.InstallLlama(ctx, cfg, hw)
		if err != nil {
			return r, err
		}
		say("llama-server: %s", bin)
	}

	// oMLX, on Apple Silicon when the model has MLX weights.
	wantMLX := WantsMLX(cfg, hw)
	if wantMLX {
		switch {
		case cfg.OMLXBin != "":
			if !FileExists(cfg.OMLXBin) {
				return r, fmt.Errorf("omlx_bin %s does not exist", cfg.OMLXBin)
			}
			say("omlx: %s (configured)", cfg.OMLXBin)
		case FileExists(cfg.OMLXBinPath()):
			say("omlx: %s (installed)", cfg.OMLXBinPath())
		default:
			say("omlx: installing %s@%s into %s …", cfg.OMLXRepo, cfg.OMLXRef, cfg.Paths().Venv)
			if err := InstallOMLX(ctx, cfg, out); err != nil {
				return r, err
			}
			say("omlx: %s", cfg.OMLXBinPath())
		}
		// Now that mlx is importable, the Metal working set can be measured
		// instead of estimated.
		if hw2, err := Detect(SystemProbe(), cfg.Dir, cfg.GPU, mlxPython(cfg)); err == nil {
			hw = hw2
			r.Hardware = hw
		}
	}

	// Weights. Local paths must exist; hf:// references are fetched after
	// the fit checks.
	if cfg.Model.GGUF != "" {
		if err := fetchFile(ctx, cfg, hw, f, cfg.Model.GGUF, cfg.GGUFPath(), true, say); err != nil {
			return r, err
		}
		if cfg.Model.MMProj != "" {
			if err := fetchFile(ctx, cfg, hw, f, cfg.Model.MMProj, cfg.MMProjPath(), false, say); err != nil {
				return r, err
			}
		}
	}
	if wantMLX {
		if err := fetchRepo(ctx, cfg, hw, f, say); err != nil {
			return r, err
		}
	}

	r.inspect(cfg, hw)
	if len(r.Problems) > 0 {
		return r, fmt.Errorf("%s", strings.Join(r.Problems, "; "))
	}
	if r.Backend == BackendMLX {
		if err := PrepareOMLX(cfg, r.Plan); err != nil {
			return r, err
		}
	}
	return r, nil
}

// fetchFile makes sure one weights file is present. checkMemory is for the
// main weights; the projector rides on the same check.
func fetchFile(ctx context.Context, cfg Config, hw Hardware, f *Fetcher, ref, dest string, checkMemory bool, say func(string, ...any)) error {
	if !isHF(ref) {
		if !FileExists(dest) {
			return fmt.Errorf("no file at %s", dest)
		}
		say("weights: %s", dest)
		return nil
	}
	if FileExists(dest) {
		say("weights: %s (downloaded)", dest)
		return nil
	}
	org, repo, path, _ := splitHF(ref, true)
	hf, err := f.HFFetchFile(ctx, org+"/"+repo, path, "")
	if err != nil {
		return err
	}
	if err := fits(hw, hf.Size, cfg, checkMemory); err != nil {
		return err
	}
	say("weights: downloading %s (%s) to %s …", ref, HumanBytes(uint64(hf.Size)), dest)
	if _, err := f.HFFetchFile(ctx, org+"/"+repo, path, dest); err != nil {
		return err
	}
	say("weights: %s verified", dest)
	return nil
}

func fetchRepo(ctx context.Context, cfg Config, hw Hardware, f *Fetcher, say func(string, ...any)) error {
	ref, dir := cfg.Model.MLX, cfg.MLXPath()
	if !isHF(ref) {
		if !FileExists(filepath.Join(dir, "config.json")) {
			return fmt.Errorf("no MLX model at %s (no config.json)", dir)
		}
		say("mlx weights: %s", dir)
		return nil
	}
	org, repo, _, _ := splitHF(ref, false)
	files, err := f.HFList(ctx, org+"/"+repo)
	if err != nil {
		return err
	}
	need := HFRepoSize(files)
	// Already complete: every file there at its size.
	complete := true
	for _, hf := range files {
		st, err := os.Stat(filepath.Join(dir, filepath.FromSlash(hf.Path)))
		if err != nil || st.Size() != hf.Size {
			complete = false
			break
		}
	}
	if complete {
		say("mlx weights: %s (downloaded)", dir)
		return nil
	}
	if err := fits(hw, need, cfg, true); err != nil {
		return err
	}
	say("mlx weights: downloading %s (%d files, %s) to %s …", ref, len(files), HumanBytes(uint64(need)), dir)
	if err := f.HFFetchRepo(ctx, org+"/"+repo, files, dir); err != nil {
		return err
	}
	say("mlx weights: %s verified", dir)
	return nil
}

// fits refuses a download that would not fit on the disk, or whose weights
// would not fit in memory: better a sentence now than 190 GB fetched for a
// backend that dies loading them.
func fits(hw Hardware, size int64, cfg Config, memory bool) error {
	if err := CheckSpace(hw, size, cfg.KeepFreeGB); err != nil {
		return err
	}
	if memory && uint64(size)+reserve > hw.Budget() {
		return fmt.Errorf("%s of weights would not fit in memory: %s available (%s) less %s reserved",
			HumanBytes(uint64(size)), HumanBytes(hw.Budget()), hw.GPUMemSource, HumanBytes(reserve))
	}
	return nil
}

// --- oMLX install -------------------------------------------------------------

// MLXPython is a Python that can import mlx, for measuring the Metal
// working set: the venv's, or the one beside a configured omlx. Empty when
// there is none yet.
func MLXPython(cfg Config) string { return mlxPython(cfg) }

func mlxPython(cfg Config) string {
	py := filepath.Join(filepath.Dir(cfg.OMLXBinPath()), "python")
	if FileExists(py) {
		return py
	}
	return ""
}

// InstallOMLX makes the venv and pip-installs oMLX from git at the pinned
// ref. oMLX is not on PyPI, and mlx has no wheels for Python 3.14, so the
// interpreter is the configured one, else 3.13/3.12/3.11 on PATH, else what
// uv can find or install.
func InstallOMLX(ctx context.Context, cfg Config, out io.Writer) error {
	py, err := findPython(ctx, cfg.Python)
	if err != nil {
		return err
	}
	venv := cfg.Paths().Venv
	fmt.Fprintf(out, "omlx: venv with %s\n", py)
	if err := run(ctx, out, nil, py, "-m", "venv", venv); err != nil {
		return err
	}
	pip := filepath.Join(venv, "bin", "pip")
	env := []string{}
	if cfg.OMLXCustomKernels {
		env = append(env, "OMLX_WITH_CUSTOM_KERNEL=1")
	}
	spec := "omlx @ git+" + cfg.OMLXRepo + "@" + cfg.OMLXRef
	if err := run(ctx, out, env, pip, "install", spec); err != nil {
		return err
	}
	return run(ctx, out, nil, cfg.OMLXBinPath(), "--version")
}

func findPython(ctx context.Context, configured string) (string, error) {
	if configured != "" {
		if !FileExists(configured) {
			return "", fmt.Errorf("python %s does not exist", configured)
		}
		return configured, nil
	}
	for _, name := range []string{"python3.13", "python3.12", "python3.11"} {
		if p, err := exec.LookPath(name); err == nil {
			return p, nil
		}
	}
	if uv, err := exec.LookPath("uv"); err == nil {
		for _, v := range []string{"3.13", "3.12", "3.11"} {
			if out, err := exec.CommandContext(ctx, uv, "python", "find", v).Output(); err == nil {
				return strings.TrimSpace(string(out)), nil
			}
		}
		if err := exec.CommandContext(ctx, uv, "python", "install", "3.12").Run(); err == nil {
			if out, err := exec.CommandContext(ctx, uv, "python", "find", "3.12").Output(); err == nil {
				return strings.TrimSpace(string(out)), nil
			}
		}
	}
	return "", fmt.Errorf("no Python 3.11–3.13 found (mlx has no 3.14 wheels): install one, or uv, or set python: in the config")
}

func run(ctx context.Context, out io.Writer, env []string, name string, args ...string) error {
	c := exec.CommandContext(ctx, name, args...)
	c.Stdout, c.Stderr = out, out
	c.Env = append(os.Environ(), env...)
	if err := c.Run(); err != nil {
		return fmt.Errorf("%s %s: %w", filepath.Base(name), strings.Join(args, " "), err)
	}
	return nil
}

// --- printing -------------------------------------------------------------------

// PrintHardware is the machine as doctor and setup show it.
func PrintHardware(w io.Writer, h Hardware) {
	fmt.Fprintf(w, "machine: %s/%s, %s, %d CPUs, %s RAM\n", h.OS, h.Arch, h.Chip, h.CPUs, HumanBytes(h.RAM))
	switch {
	case h.GPU == GPUCPU:
		fmt.Fprintf(w, "gpu:     none usable (CPU build; sizing against 80%% of RAM)\n")
	case h.GPUMem > 0:
		fmt.Fprintf(w, "gpu:     %s %s, %s usable (%s)", h.GPU, h.GPUName, HumanBytes(h.GPUMem), h.GPUMemSource)
		if h.CUDAVersion != "" {
			fmt.Fprintf(w, ", CUDA %s", h.CUDAVersion)
		}
		fmt.Fprintln(w)
	default:
		fmt.Fprintf(w, "gpu:     %s %s, memory unknown (%s); sizing against 80%% of RAM\n", h.GPU, h.GPUName, h.GPUMemSource)
	}
	fmt.Fprintf(w, "disk:    %s free at %s\n", HumanBytes(h.DiskFree), h.DiskPath)
	// macOS caps what one process may wire for the GPU below the RAM; the
	// cap is an operator's sysctl. Worth knowing when a model is close.
	if h.GPU == GPUMetal && h.GPUMemSource != "sysctl iogpu.wired_limit_mb" && h.RAM > h.GPUMem+16*GiB {
		fmt.Fprintf(w, "note:    macOS lets the GPU use %s of %s by default; to raise it: sudo sysctl iogpu.wired_limit_mb=%d (not persistent across reboots)\n",
			HumanBytes(h.GPUMem), HumanBytes(h.RAM), (h.RAM-8*GiB)/MiB)
	}
}

// PrintReport is the rest of doctor's and setup's output.
func PrintReport(w io.Writer, r Report) {
	mark := func(ok bool) string {
		if ok {
			return "ok     "
		}
		return "MISSING"
	}
	fmt.Fprintf(w, "%s\n", r.Reason)
	fmt.Fprintf(w, "  %s llama-server  %s\n", mark(r.LlamaOK), r.LlamaBin)
	if r.OMLXBin != "" {
		fmt.Fprintf(w, "  %s omlx          %s\n", mark(r.OMLXOK), r.OMLXBin)
	}
	if r.GGUF != "" {
		fmt.Fprintf(w, "  %s gguf          %s\n", mark(r.GGUFOK), r.GGUF)
	}
	if r.MLX != "" {
		fmt.Fprintf(w, "  %s mlx           %s\n", mark(r.MLXOK), r.MLX)
	}
	if r.Model.Arch != "" {
		m := r.Model
		fmt.Fprintf(w, "model:   %s (%s): %d layers (%d attention), %d KV heads × %d, context %d, %s on disk, %d KiB KV per token\n",
			m.Name, m.Arch, m.Layers, m.AttnLayers, m.KVHeads, m.HeadK, m.Context, HumanBytes(uint64(m.WeightBytes)), m.KVBytesPerToken()/1024)
	}
	for _, n := range r.Plan.Notes {
		fmt.Fprintf(w, "  %s\n", n)
	}
	if len(r.Problems) > 0 {
		fmt.Fprintln(w, "problems:")
		for _, p := range r.Problems {
			fmt.Fprintf(w, "  - %s\n", p)
		}
		return
	}
	fmt.Fprintln(w)
	switch r.Plan.Backend {
	case BackendLlama:
		fmt.Fprintf(w, "launch:  llama-server, %d slots × %d tokens, cache-ram %d MB\n", r.Plan.Slots, r.Plan.Context, r.Plan.CacheRAMMB)
	case BackendMLX:
		fmt.Fprintf(w, "launch:  omlx, %d concurrent requests, context window %d\n", r.Plan.MaxConcurrent, r.Plan.ContextWindow)
	}
	fmt.Fprintf(w, "url:     %s\n", r.URL)
	fmt.Fprintf(w, "model:   %s\n", r.ModelID)
	fmt.Fprintf(w, "client:  an OpenAI-compatible provider with base_url %s and model %q\n", r.URL, r.ModelID)
}
