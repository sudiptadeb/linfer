package linfer

import (
	"path/filepath"
	"strings"
	"testing"
)

const exampleYAML = `
backend: auto
listen: 127.0.0.1:8090
model:
  id: qwen3-test:0.6b-q8_0
  gguf: hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
  mlx: hf://mlx-community/Qwen3-0.6B-4bit
dir: /tmp/linfer-test
`

// The documented example parses, defaults fill what it leaves out, and
// hf:// references resolve to deterministic places under dir, so serve
// finds what setup fetched without the config being rewritten.
func TestParseDefaultsAndPaths(t *testing.T) {
	c, err := Parse([]byte(exampleYAML))
	if err != nil {
		t.Fatal(err)
	}
	if c.LlamaRelease != DefaultLlamaRelease || c.OMLXRef != DefaultOMLXRef || c.KeepFreeGB != 200 || c.Model.ReasoningEffort != "medium" {
		t.Errorf("defaults not filled: %+v", c)
	}
	if got, want := c.GGUFPath(), filepath.Join("/tmp/linfer-test/models/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"); got != want {
		t.Errorf("gguf path %s, want %s", got, want)
	}
	if got, want := c.MLXPath(), "/tmp/linfer-test/models/mlx-community/Qwen3-0.6B-4bit"; got != want {
		t.Errorf("mlx path %s, want %s", got, want)
	}
	if got, want := c.LlamaBinPath(), "/tmp/linfer-test/bin/llama-b11429/llama-server"; got != want {
		t.Errorf("llama bin %s, want %s", got, want)
	}
	if c.URL() != "http://127.0.0.1:8090/v1" {
		t.Errorf("url %s", c.URL())
	}
}

// A local path is used as given, with ~ expanded.
func TestLocalPathsKept(t *testing.T) {
	c, err := Parse([]byte("model:\n  id: m\n  gguf: ~/models/x.gguf\nllama_bin: /opt/llama/llama-server\n"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.HasPrefix(c.GGUFPath(), "~") || !strings.HasSuffix(c.GGUFPath(), "/models/x.gguf") {
		t.Errorf("gguf %s", c.GGUFPath())
	}
	if c.LlamaBinPath() != "/opt/llama/llama-server" {
		t.Errorf("llama bin %s", c.LlamaBinPath())
	}
}

// What the config refuses, each with the reason in the error.
func TestParseRefuses(t *testing.T) {
	cases := map[string]string{
		"model:\n  id: m\n  gguf: a.gguf\nbackned: llama\n":          "field backned not found",
		"backend: fast\nmodel:\n  id: m\n  gguf: a.gguf\n":           "backend \"fast\"",
		"model:\n  gguf: a.gguf\n":                                   "model.id is required",
		"model:\n  id: a/b\n  gguf: a.gguf\n":                        "no slash",
		"model:\n  id: m\n":                                          "needs gguf",
		"backend: mlx\nmodel:\n  id: m\n  gguf: a.gguf\n":            "backend mlx needs model.mlx",
		"model:\n  id: m\n  gguf: hf://Qwen/Qwen3-0.6B-GGUF\n":       "want hf://<org>/<repo>/<file>",
		"model:\n  id: m\n  mlx: hf://mlx-community/x/config.json\n": "whole repo",
		"model:\n  id: m\n  gguf: a.gguf\n  slots: -1\n":             "0 (auto) or positive",
	}
	for in, want := range cases {
		_, err := Parse([]byte(in))
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%q: got %v, want %q", in, err, want)
		}
	}
}
