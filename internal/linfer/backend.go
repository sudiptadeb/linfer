package linfer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
)

// Choose resolves Config.Backend. The auto rule: oMLX only on Apple Silicon,
// when the model has MLX weights that are present and an omlx binary exists;
// anything else is llama.cpp, which runs everywhere. The reason is for the
// report.
func Choose(cfg Config, h Hardware, exists func(string) bool) (backend, reason string) {
	switch cfg.Backend {
	case BackendLlama, BackendMLX:
		return cfg.Backend, "backend: " + cfg.Backend + " from config"
	}
	if h.OS != "darwin" || h.Arch != "arm64" {
		return BackendLlama, "backend: llama (MLX needs Apple Silicon)"
	}
	if cfg.Model.MLX == "" {
		return BackendLlama, "backend: llama (the model has no mlx weights)"
	}
	if !exists(cfg.MLXPath()) {
		return BackendLlama, "backend: llama (no MLX weights at " + cfg.MLXPath() + "; run setup with backend: mlx to fetch them)"
	}
	if !exists(cfg.OMLXBinPath()) {
		return BackendLlama, "backend: llama (no omlx at " + cfg.OMLXBinPath() + "; run setup with backend: mlx to install it)"
	}
	return BackendMLX, "backend: mlx (Apple Silicon, MLX weights and omlx present)"
}

// WantsMLX says whether setup should fetch oMLX and the MLX weights: when
// the config asks for mlx outright, or auto could end up there.
func WantsMLX(cfg Config, h Hardware) bool {
	if cfg.Backend == BackendLlama || cfg.Model.MLX == "" {
		return false
	}
	return h.OS == "darwin" && h.Arch == "arm64"
}

// FileExists is Choose's exists in production.
func FileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

// Command is what to run.
type Command struct {
	Path string
	Args []string
	Env  []string // additions to the environment
	// Health is the URL that answers 200 once the backend is ready.
	Health string
}

// LlamaCommand is llama-server with the measured launch profile.
//
// Beyond the model and the plan's slots, context and cache: 4 restore
// points per slot (--ctx-checkpoints) so a slot whose place was taken
// restores from the newest instead of reprocessing; -sps 0.6 is the prefix
// similarity at which a cached prompt is reused; --jinja so tool calls are
// grammar-constrained from the template; flash attention; --load-mode none
// (plain reads, which were as fast as mmap here and keep the page cache
// honest); batch 2048 / micro-batch 512; the model's own sampling (temp 1.0,
// top-k 20, top-p 0.95, min-p 0, no repeat penalty), since llama-server's
// defaults (0.8, 40, 0.05) are not what the model card says; and the
// reasoning effort for the template. No speculative decoding: n-gram
// drafting was tried and judged an over-optimisation for agent traffic.
func LlamaCommand(cfg Config, p Plan) Command {
	host, port := splitListen(cfg.Listen)
	args := []string{"--model", cfg.GGUFPath()}
	if cfg.Model.MMProj != "" {
		args = append(args, "--mmproj", cfg.MMProjPath())
	}
	args = append(args,
		"--host", host, "--port", port,
		"--alias", cfg.Model.ID,
		"-np", strconv.Itoa(p.Slots),
		"-c", strconv.Itoa(p.Slots*p.Context),
		"--cache-ram", strconv.Itoa(p.CacheRAMMB),
		"--ctx-checkpoints", "4",
		"-sps", "0.6",
		"--jinja",
		"--flash-attn", "on",
		"--load-mode", "none",
		"-b", "2048", "-ub", "512",
		"--temp", "1.0", "--top-k", "20", "--top-p", "0.95", "--min-p", "0.0", "--repeat-penalty", "1.0",
		"--chat-template-kwargs", templateKwargs(cfg),
	)
	return Command{Path: cfg.LlamaBinPath(), Args: args, Health: "http://" + cfg.Listen + "/health"}
}

// OMLXCommand is oMLX serving the one directory linfer keeps for it, with
// its state (model_settings.json, caches) under linfer's directory rather
// than ~/.omlx. --no-hf-cache: by default oMLX also lists every model in the
// Hugging Face cache, and this server is to serve exactly one id.
func OMLXCommand(cfg Config, p Plan) Command {
	host, port := splitListen(cfg.Listen)
	paths := cfg.Paths()
	args := []string{"serve",
		"--model-dir", paths.ModelDir,
		"--base-path", paths.State,
		"--host", host, "--port", port,
		"--max-concurrent-requests", strconv.Itoa(p.MaxConcurrent),
		"--no-hf-cache",
	}
	return Command{Path: cfg.OMLXBinPath(), Args: args, Health: "http://" + cfg.Listen + "/v1/models"}
}

// CommandFor is the command for the plan's backend.
func CommandFor(cfg Config, p Plan) Command {
	if p.Backend == BackendMLX {
		return OMLXCommand(cfg, p)
	}
	return LlamaCommand(cfg, p)
}

func templateKwargs(cfg Config) string {
	b, _ := json.Marshal(map[string]string{"reasoning_effort": cfg.Model.ReasoningEffort})
	return string(b)
}

func splitListen(listen string) (host, port string) {
	for i := len(listen) - 1; i >= 0; i-- {
		if listen[i] == ':' {
			return listen[:i], listen[i+1:]
		}
	}
	return listen, ""
}

// --- oMLX's files -------------------------------------------------------------------

// PrepareOMLX writes what oMLX reads at start: the model directory, holding
// one symlink named Model.ID at the weights (oMLX's model id is the
// directory's name, so this is how the id stays the same as llama's), and
// model_settings.json under the state directory with the launch profile:
// multi-token prediction on, the model pinned in memory, the same sampling
// llama gets, the reasoning effort, and the context window.
func PrepareOMLX(cfg Config, p Plan) error {
	paths := cfg.Paths()
	if err := os.MkdirAll(paths.ModelDir, 0o755); err != nil {
		return err
	}
	if err := os.MkdirAll(paths.State, 0o755); err != nil {
		return err
	}
	// One entry only: an old id's symlink would be served too.
	entries, err := os.ReadDir(paths.ModelDir)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if err := os.Remove(filepath.Join(paths.ModelDir, e.Name())); err != nil {
			return err
		}
	}
	target, err := filepath.Abs(cfg.MLXPath())
	if err != nil {
		return err
	}
	if err := os.Symlink(target, filepath.Join(paths.ModelDir, cfg.Model.ID)); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(paths.State, "model_settings.json"), OMLXSettings(cfg, p), 0o644)
}

// omlxModelSettings is one model's entry in model_settings.json. The field
// order is the file's order.
type omlxModelSettings struct {
	MTPEnabled         bool              `json:"mtp_enabled"`
	IsPinned           bool              `json:"is_pinned"`
	MaxContextWindow   int               `json:"max_context_window"`
	Temperature        float64           `json:"temperature"`
	TopP               float64           `json:"top_p"`
	TopK               int               `json:"top_k"`
	MinP               float64           `json:"min_p"`
	RepetitionPenalty  float64           `json:"repetition_penalty"`
	ChatTemplateKwargs map[string]string `json:"chat_template_kwargs"`
}

// OMLXSettings is model_settings.json, version 1: {"version":1,"models":{id:{…}}}.
func OMLXSettings(cfg Config, p Plan) []byte {
	doc := struct {
		Version int                          `json:"version"`
		Models  map[string]omlxModelSettings `json:"models"`
	}{
		Version: 1,
		Models: map[string]omlxModelSettings{cfg.Model.ID: {
			MTPEnabled:         true,
			IsPinned:           true,
			MaxContextWindow:   p.ContextWindow,
			Temperature:        1.0,
			TopP:               0.95,
			TopK:               20,
			MinP:               0.0,
			RepetitionPenalty:  1.0,
			ChatTemplateKwargs: map[string]string{"reasoning_effort": cfg.Model.ReasoningEffort},
		}},
	}
	b, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		panic(fmt.Sprintf("marshal model_settings: %v", err)) // unreachable: fixed types
	}
	return append(b, '\n')
}
