// Package linfer sets up and runs the best local inference backend for one
// model on this machine, the way Ollama or oMLX do for theirs: it reads one
// config file, detects the hardware, fetches the backend and the weights when
// they are not there, picks the backend (llama.cpp's llama-server, or oMLX on
// Apple Silicon), sizes it to the memory left after the weights, launches it
// as a child, restarts it when it dies, and reports the URL to use.
//
// It is not a proxy. Inference requests go from the client straight to the
// backend's own OpenAI-compatible /v1 endpoint on `listen`, which is the same
// address whichever backend is running, so the client's configuration never
// changes. linfer only chooses, launches, supervises and reports.
//
// The daemon (`linfer serve`) is driven over a unix socket under its own
// directory (control.go): `status`, `pause` (stop the backend and free its
// memory while the daemon stays up, so a benchmark can have the machine),
// `resume` and `switch`. `setup` gets a fresh box to a working backend and
// prints what the client needs; `doctor` reports without changing anything.
package linfer

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"gopkg.in/yaml.v3"
)

// Backends. "auto" is resolved by Choose.
const (
	BackendAuto  = "auto"
	BackendLlama = "llama"
	BackendMLX   = "mlx"
)

// Defaults for what the config leaves out. The llama.cpp release is the
// pinned prebuilt (release.go); the oMLX ref is a commit known to run the
// reference model with multi-token prediction, since oMLX is not published
// on PyPI and is installed from git.
const (
	DefaultListen       = "127.0.0.1:8090"
	DefaultLlamaRelease = "b11429"
	DefaultOMLXRepo     = "https://github.com/jundot/omlx.git"
	DefaultOMLXRef      = "70791316e7"
	DefaultKeepFreeGB   = 200
	DefaultEffort       = "medium"
)

// Config is linfer.yaml. The file is the configuration: linfer has no
// database, and the operator edits it in place. Every path may start with ~,
// and a model path may be an hf:// reference (Model), which setup downloads
// into linfer's own models directory and serve resolves to the same place.
type Config struct {
	Backend string `yaml:"backend"` // auto | llama | mlx
	// Listen is the host:port the backend binds to. It is the same for every
	// backend, so the client's base URL (http://<listen>/v1) never changes.
	Listen string `yaml:"listen"`
	Model  Model  `yaml:"model"`

	// Backend binaries. Empty means linfer's own install under Dir, which
	// setup creates: <Dir>/bin/llama-<release>/llama-server and
	// <Dir>/venv-omlx/bin/omlx. A configured path that exists is used as is.
	LlamaBin     string `yaml:"llama_bin"`
	LlamaRelease string `yaml:"llama_release"` // ggml-org/llama.cpp release tag, default b11429
	OMLXBin      string `yaml:"omlx_bin"`
	OMLXRepo     string `yaml:"omlx_repo"` // git URL oMLX is pip-installed from
	OMLXRef      string `yaml:"omlx_ref"`  // commit or tag of that repo
	// Python is an interpreter for the oMLX venv, 3.11 to 3.13 (mlx has no
	// 3.14 wheels). Empty: python3.13, python3.12, python3.11 on PATH, then
	// `uv python find`.
	Python string `yaml:"python"`
	// OMLXCustomKernels builds oMLX's native Metal kernels
	// (OMLX_WITH_CUSTOM_KERNEL=1); needs the Metal toolchain. Off by default.
	OMLXCustomKernels bool `yaml:"omlx_custom_kernels"`

	// GPU overrides detection on Linux: cuda | rocm | vulkan | cpu. Empty is auto.
	GPU string `yaml:"gpu"`
	// Dir is linfer's own directory: bin/, models/, venv-omlx/, the oMLX
	// state and the control socket. Default $XDG_DATA_HOME/linfer, which is
	// ~/.local/share/linfer.
	Dir string `yaml:"dir"`
	// KeepFreeGB is how much disk a download must leave free. Default 200.
	KeepFreeGB int `yaml:"keep_free_gb"`
	// CacheRAMMB is llama-server's --cache-ram; 0 is auto (sizing.go).
	CacheRAMMB int `yaml:"cache_ram_mb"`
	// MaxConcurrent is oMLX's --max-concurrent-requests; 0 is auto. 6 is the
	// most auto will pick: 8 concurrent ~65k-token requests ran oMLX out of
	// Metal memory on the 256 GB reference box.
	MaxConcurrent int `yaml:"max_concurrent"`
}

// Model is the one model linfer serves. Its id is what the client names,
// stable across backends: llama-server gets it as --alias, and oMLX reads it
// from the name of the directory the weights are in, so linfer gives oMLX a
// directory holding a symlink of that name.
type Model struct {
	ID string `yaml:"id"`
	// GGUF is the weights for llama.cpp: a path, or hf://<org>/<repo>/<file>.
	GGUF string `yaml:"gguf"`
	// MMProj is the vision projector that goes with GGUF, same forms. Optional.
	MMProj string `yaml:"mmproj"`
	// MLX is the MLX weights directory for oMLX: a path, or hf://<org>/<repo>.
	MLX string `yaml:"mlx"`
	// Slots and Context (tokens per slot) size llama-server: -np Slots and
	// -c Slots*Context. 0 is auto. Context also caps oMLX's context window.
	Slots   int `yaml:"slots"`
	Context int `yaml:"context"`
	// ReasoningEffort goes to the chat template as reasoning_effort. Without
	// it Qwen3.8's template writes "Reasoning effort is set to xhigh" into
	// every prompt. Default medium.
	ReasoningEffort string `yaml:"reasoning_effort"`
}

// Load reads and checks a config file, filling the defaults. Unknown keys
// are an error: a misspelt key silently falling back to a default would be
// found at 3 a.m. by a model running with the wrong context.
func Load(path string) (Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Config{}, err
	}
	c, err := Parse(raw)
	if err != nil {
		return Config{}, fmt.Errorf("%s: %w", path, err)
	}
	return c, nil
}

// Parse is Load on bytes.
func Parse(raw []byte) (Config, error) {
	var c Config
	dec := yaml.NewDecoder(strings.NewReader(string(raw)))
	dec.KnownFields(true)
	if err := dec.Decode(&c); err != nil {
		return Config{}, err
	}
	c.defaults()
	if err := c.check(); err != nil {
		return Config{}, err
	}
	return c, nil
}

func (c *Config) defaults() {
	if c.Backend == "" {
		c.Backend = BackendAuto
	}
	if c.Listen == "" {
		c.Listen = DefaultListen
	}
	if c.LlamaRelease == "" {
		c.LlamaRelease = DefaultLlamaRelease
	}
	if c.OMLXRepo == "" {
		c.OMLXRepo = DefaultOMLXRepo
	}
	if c.OMLXRef == "" {
		c.OMLXRef = DefaultOMLXRef
	}
	if c.KeepFreeGB == 0 {
		c.KeepFreeGB = DefaultKeepFreeGB
	}
	if c.Model.ReasoningEffort == "" {
		c.Model.ReasoningEffort = DefaultEffort
	}
	if c.Dir == "" {
		c.Dir = DefaultDir()
	}
	for _, p := range []*string{&c.LlamaBin, &c.OMLXBin, &c.Python, &c.Dir, &c.Model.GGUF, &c.Model.MMProj, &c.Model.MLX} {
		*p = expandHome(*p)
	}
}

func (c Config) check() error {
	switch c.Backend {
	case BackendAuto, BackendLlama, BackendMLX:
	default:
		return fmt.Errorf("backend %q: want auto, llama or mlx", c.Backend)
	}
	switch c.GPU {
	case "", GPUCUDA, GPUROCm, GPUVulkan, GPUCPU:
	default:
		return fmt.Errorf("gpu %q: want cuda, rocm, vulkan or cpu (or leave it out)", c.GPU)
	}
	if !strings.Contains(c.Listen, ":") {
		return fmt.Errorf("listen %q: want host:port", c.Listen)
	}
	m := c.Model
	if m.ID == "" {
		return fmt.Errorf("model.id is required")
	}
	if strings.ContainsAny(m.ID, "/ \t\n") {
		// It becomes a directory name for oMLX and the tail of the client's
		// `<provider>/<id>`; a slash would break both.
		return fmt.Errorf("model.id %q: no slash or whitespace", m.ID)
	}
	if m.GGUF == "" && m.MLX == "" {
		return fmt.Errorf("model needs gguf (for llama) or mlx (for oMLX), or both")
	}
	if c.Backend == BackendLlama && m.GGUF == "" {
		return fmt.Errorf("backend llama needs model.gguf")
	}
	if c.Backend == BackendMLX && m.MLX == "" {
		return fmt.Errorf("backend mlx needs model.mlx")
	}
	if m.Slots < 0 || m.Context < 0 || c.CacheRAMMB < 0 || c.MaxConcurrent < 0 {
		return fmt.Errorf("slots, context, cache_ram_mb and max_concurrent are 0 (auto) or positive")
	}
	for _, ref := range []string{m.GGUF, m.MMProj} {
		if isHF(ref) {
			if _, _, _, err := splitHF(ref, true); err != nil {
				return err
			}
		}
	}
	if isHF(m.MLX) {
		if _, _, _, err := splitHF(m.MLX, false); err != nil {
			return err
		}
	}
	return nil
}

// --- paths --------------------------------------------------------------------

// Paths are the places under Dir. Everything linfer makes is here, so
// removing the directory removes linfer's footprint.
type Paths struct {
	Root     string
	Bin      string // prebuilt llama.cpp releases, one directory each
	Models   string // downloaded weights: <org>/<repo>/...
	Venv     string // the oMLX virtualenv
	State    string // oMLX --base-path: model_settings.json and its caches
	ModelDir string // oMLX --model-dir: holds the one symlink named Model.ID
	Socket   string // the control socket
	Log      string // the backend's stdout and stderr
}

func (c Config) Paths() Paths {
	r := c.Dir
	return Paths{
		Root:     r,
		Bin:      filepath.Join(r, "bin"),
		Models:   filepath.Join(r, "models"),
		Venv:     filepath.Join(r, "venv-omlx"),
		State:    filepath.Join(r, "omlx-state"),
		ModelDir: filepath.Join(r, "omlx-models"),
		Socket:   filepath.Join(r, "linfer.sock"),
		Log:      filepath.Join(r, "backend.log"),
	}
}

// LlamaBinPath is the llama-server to run: the configured one, else
// linfer's own install of the pinned release.
func (c Config) LlamaBinPath() string {
	if c.LlamaBin != "" {
		return c.LlamaBin
	}
	return filepath.Join(c.Paths().Bin, "llama-"+c.LlamaRelease, "llama-server")
}

// OMLXBinPath is the omlx to run: the configured one, else linfer's venv.
func (c Config) OMLXBinPath() string {
	if c.OMLXBin != "" {
		return c.OMLXBin
	}
	return filepath.Join(c.Paths().Venv, "bin", "omlx")
}

// GGUFPath, MMProjPath and MLXPath are where the weights are, or will be
// after setup: a local path as given, an hf:// reference under Models.
func (c Config) GGUFPath() string   { return c.localPath(c.Model.GGUF, true) }
func (c Config) MMProjPath() string { return c.localPath(c.Model.MMProj, true) }
func (c Config) MLXPath() string    { return c.localPath(c.Model.MLX, false) }

func (c Config) localPath(ref string, file bool) string {
	if ref == "" || !isHF(ref) {
		return ref
	}
	org, repo, rest, _ := splitHF(ref, file)
	return filepath.Join(c.Paths().Models, org, repo, filepath.FromSlash(rest))
}

// URL is the backend's OpenAI-compatible base, what the client points at.
func (c Config) URL() string { return "http://" + c.Listen + "/v1" }

// --- hf:// references -------------------------------------------------------------

func isHF(ref string) bool { return strings.HasPrefix(ref, "hf://") }

// splitHF takes hf://<org>/<repo>[/<path>] apart. A file reference needs a
// path; a repo reference (an MLX directory) must not have one.
func splitHF(ref string, file bool) (org, repo, rest string, err error) {
	parts := strings.SplitN(strings.TrimPrefix(ref, "hf://"), "/", 3)
	if len(parts) < 2 || parts[0] == "" || parts[1] == "" {
		if file {
			return "", "", "", fmt.Errorf("%s: want hf://<org>/<repo>/<file>", ref)
		}
		return "", "", "", fmt.Errorf("%s: want hf://<org>/<repo>", ref)
	}
	if len(parts) == 3 {
		rest = parts[2]
	}
	if file && rest == "" {
		return "", "", "", fmt.Errorf("%s: want hf://<org>/<repo>/<file>", ref)
	}
	if !file && rest != "" {
		return "", "", "", fmt.Errorf("%s: an MLX model is a whole repo, hf://<org>/<repo>", ref)
	}
	return parts[0], parts[1], rest, nil
}

func expandHome(p string) string {
	if p == "~" || strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, p[1:])
		}
	}
	return p
}

// DefaultConfigPath is $XDG_CONFIG_HOME/linfer/linfer.yaml, which is
// ~/.config/linfer/linfer.yaml; the same on macOS, where a dotfile directory
// is what a command-line tool's user expects to edit.
func DefaultConfigPath() string {
	return filepath.Join(xdg("XDG_CONFIG_HOME", ".config"), "linfer", "linfer.yaml")
}

// DefaultDir is $XDG_DATA_HOME/linfer, which is ~/.local/share/linfer. The
// weights live here too, so a big model wants `dir` pointed at a big disk.
func DefaultDir() string {
	return filepath.Join(xdg("XDG_DATA_HOME", filepath.Join(".local", "share")), "linfer")
}

func xdg(env, fallback string) string {
	if runtime.GOOS == "linux" || os.Getenv(env) != "" {
		if v := os.Getenv(env); v != "" {
			return v
		}
	}
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, fallback)
}
