package linfer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func testConfig(t *testing.T, dir string) Config {
	t.Helper()
	c, err := Parse([]byte(`
listen: 127.0.0.1:8090
model:
  id: qwen3.8-flash-next:125b-a6b-q8_0
  gguf: /models/q8/model.gguf
  mmproj: /models/q8/mmproj.gguf
  mlx: /models/oQ8e-mtp
llama_bin: /opt/llama/llama-server
omlx_bin: /opt/venv/bin/omlx
dir: ` + dir + "\n"))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// The llama-server command line is the measured launch profile, exactly.
func TestLlamaCommand(t *testing.T) {
	cfg := testConfig(t, "/var/linfer")
	cmd := LlamaCommand(cfg, Plan{Backend: BackendLlama, Slots: 8, Context: 131072, CacheRAMMB: 16384})
	want := `--model /models/q8/model.gguf --mmproj /models/q8/mmproj.gguf --host 127.0.0.1 --port 8090 ` +
		`--alias qwen3.8-flash-next:125b-a6b-q8_0 -np 8 -c 1048576 --cache-ram 16384 --ctx-checkpoints 4 -sps 0.6 ` +
		`--jinja --flash-attn on --load-mode none -b 2048 -ub 512 ` +
		`--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 --repeat-penalty 1.0 --chat-template-kwargs {"reasoning_effort":"medium"}`
	if got := strings.Join(cmd.Args, " "); got != want {
		t.Errorf("got\n%s\nwant\n%s", got, want)
	}
	if cmd.Path != "/opt/llama/llama-server" || cmd.Health != "http://127.0.0.1:8090/health" {
		t.Errorf("path %s health %s", cmd.Path, cmd.Health)
	}
}

// An MTP head adds the draft flags right after the weights.
func TestLlamaCommandMTP(t *testing.T) {
	cfg := testConfig(t, "/var/linfer")
	cfg.Model.MTP = "/models/q8/mtp.gguf"
	cmd := LlamaCommand(cfg, Plan{Backend: BackendLlama, Slots: 8, Context: 131072, CacheRAMMB: 16384})
	want := `--mmproj /models/q8/mmproj.gguf -md /models/q8/mtp.gguf --spec-type draft-mtp ` +
		`--spec-draft-n-max 3 --spec-draft-sampling probabilistic --host`
	if got := strings.Join(cmd.Args, " "); !strings.Contains(got, want) {
		t.Errorf("got\n%s\nwant it to contain\n%s", got, want)
	}
}

// The oMLX command line: linfer's own model and state directories, the
// concurrency from the plan, and no Hugging Face cache scan.
func TestOMLXCommand(t *testing.T) {
	cfg := testConfig(t, "/var/linfer")
	cmd := OMLXCommand(cfg, Plan{Backend: BackendMLX, MaxConcurrent: 6, ContextWindow: 262144})
	want := `serve --model-dir /var/linfer/omlx-models --base-path /var/linfer/omlx-state --host 127.0.0.1 --port 8090 --max-concurrent-requests 6 --no-hf-cache`
	if got := strings.Join(cmd.Args, " "); got != want {
		t.Errorf("got\n%s\nwant\n%s", got, want)
	}
	if cmd.Path != "/opt/venv/bin/omlx" || cmd.Health != "http://127.0.0.1:8090/v1/models" {
		t.Errorf("path %s health %s", cmd.Path, cmd.Health)
	}
}

// PrepareOMLX leaves one symlink named after the model id (oMLX's id is the
// directory name) and a model_settings.json with the launch profile.
func TestPrepareOMLX(t *testing.T) {
	dir := t.TempDir()
	weights := filepath.Join(dir, "weights")
	os.Mkdir(weights, 0o755)
	cfg := testConfig(t, dir)
	cfg.Model.MLX = weights
	// A stale entry from an earlier id goes.
	os.MkdirAll(cfg.Paths().ModelDir, 0o755)
	os.Symlink(weights, filepath.Join(cfg.Paths().ModelDir, "old-id"))

	if err := PrepareOMLX(cfg, Plan{Backend: BackendMLX, MaxConcurrent: 6, ContextWindow: 262144}); err != nil {
		t.Fatal(err)
	}
	entries, _ := os.ReadDir(cfg.Paths().ModelDir)
	if len(entries) != 1 || entries[0].Name() != cfg.Model.ID {
		t.Fatalf("model dir holds %v", entries)
	}
	if target, _ := os.Readlink(filepath.Join(cfg.Paths().ModelDir, cfg.Model.ID)); target != weights {
		t.Errorf("symlink to %s", target)
	}
	raw, err := os.ReadFile(filepath.Join(cfg.Paths().State, "model_settings.json"))
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Version int                       `json:"version"`
		Models  map[string]map[string]any `json:"models"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	m := doc.Models[cfg.Model.ID]
	if doc.Version != 1 || m == nil || m["mtp_enabled"] != true || m["is_pinned"] != true || m["max_context_window"] != 262144.0 ||
		m["temperature"] != 1.0 || m["top_p"] != 0.95 || m["top_k"] != 20.0 || m["min_p"] != 0.0 || m["repetition_penalty"] != 1.0 {
		t.Errorf("settings %s", raw)
	}
	if kw, _ := m["chat_template_kwargs"].(map[string]any); kw["reasoning_effort"] != "medium" {
		t.Errorf("chat_template_kwargs %v", m["chat_template_kwargs"])
	}
}

// The auto rule: mlx only on Apple Silicon with MLX weights and omlx both
// present; an explicit backend is taken as is.
func TestChoose(t *testing.T) {
	cfg := testConfig(t, "/var/linfer")
	mac := Hardware{OS: "darwin", Arch: "arm64"}
	all := func(string) bool { return true }
	only := func(have string) func(string) bool {
		return func(p string) bool { return p == have }
	}
	if b, _ := Choose(cfg, mac, all); b != BackendMLX {
		t.Errorf("mac with everything: %s", b)
	}
	if b, r := Choose(cfg, Hardware{OS: "linux", Arch: "amd64"}, all); b != BackendLlama || !strings.Contains(r, "Apple Silicon") {
		t.Errorf("linux: %s (%s)", b, r)
	}
	if b, r := Choose(cfg, mac, only(cfg.MLXPath())); b != BackendLlama || !strings.Contains(r, "no omlx") {
		t.Errorf("no omlx: %s (%s)", b, r)
	}
	if b, r := Choose(cfg, mac, only(cfg.OMLXBinPath())); b != BackendLlama || !strings.Contains(r, "no MLX weights") {
		t.Errorf("no weights: %s (%s)", b, r)
	}
	noMLX := cfg
	noMLX.Model.MLX = ""
	if b, _ := Choose(noMLX, mac, all); b != BackendLlama {
		t.Errorf("no mlx in config: %s", b)
	}
	forced := cfg
	forced.Backend = BackendLlama
	if b, _ := Choose(forced, mac, all); b != BackendLlama {
		t.Errorf("forced llama: %s", b)
	}
}
