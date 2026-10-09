package linfer

import (
	"strings"
	"testing"
)

// The reference box: a 256 GB M3 Ultra whose Metal working set measures
// 222.7 GiB, and the model it was measured with (175.3 GiB of q8 GGUF plus
// a 0.85 GiB projector; 48 blocks, 12 of them attention, 2 KV heads of 256).
var (
	refBox  = Hardware{OS: "darwin", Arch: "arm64", GPU: GPUMetal, RAM: 274877906944, GPUMem: 239143780352, GPUMemSource: "measured"}
	refGGUF = ModelInfo{Arch: "qwen4exp", Layers: 48, AttnLayers: 12, KVHeads: 2, HeadK: 256, HeadV: 256, Context: 262144,
		WeightBytes: 188224949312 + 907542784}
	refMLX = ModelInfo{Arch: "qwen4_exp", Layers: 48, AttnLayers: 12, KVHeads: 2, HeadK: 256, HeadV: 256, Context: 262144,
		WeightBytes: 182 * GiB}
)

// Auto sizing reproduces the measured llama settings on the reference box:
// 8 slots of 131,072 tokens and a 16 GiB prompt cache.
func TestSizeReferenceLlama(t *testing.T) {
	p, err := Size(refBox, refGGUF, BackendLlama, Config{})
	if err != nil {
		t.Fatal(err)
	}
	if p.Slots != 8 || p.Context != 131072 || p.CacheRAMMB != 16384 {
		t.Errorf("got %d x %d, cache %d; notes:\n%s", p.Slots, p.Context, p.CacheRAMMB, strings.Join(p.Notes, "\n"))
	}
}

// And the measured oMLX setting: 6 concurrent requests at the full window.
func TestSizeReferenceMLX(t *testing.T) {
	p, err := Size(refBox, refMLX, BackendMLX, Config{})
	if err != nil {
		t.Fatal(err)
	}
	if p.MaxConcurrent != 6 || p.ContextWindow != 262144 {
		t.Errorf("got %d concurrent, window %d; notes:\n%s", p.MaxConcurrent, p.ContextWindow, strings.Join(p.Notes, "\n"))
	}
}

// A small box: one 24 GB card and an 8B model at 128 KiB of KV per token
// gets 2 slots, context halved down to 32k so both fit, and a prompt cache
// out of system RAM rather than VRAM.
func TestSizeSmallGPU(t *testing.T) {
	box := Hardware{OS: "linux", Arch: "amd64", GPU: GPUCUDA, RAM: 64 * GiB, GPUMem: 24564 * MiB, GPUMemSource: "nvidia-smi"}
	m := ModelInfo{Layers: 32, AttnLayers: 32, KVHeads: 8, HeadK: 128, HeadV: 128, Context: 131072, WeightBytes: 5260000000}
	p, err := Size(box, m, BackendLlama, Config{})
	if err != nil {
		t.Fatal(err)
	}
	if p.Slots != 2 || p.Context != 32768 || p.CacheRAMMB != 16384 {
		t.Errorf("got %d x %d, cache %d; notes:\n%s", p.Slots, p.Context, p.CacheRAMMB, strings.Join(p.Notes, "\n"))
	}
}

// Weights that do not fit are refused with the numbers, not launched.
func TestSizeRefusesWhatDoesNotFit(t *testing.T) {
	box := Hardware{GPU: GPUCUDA, RAM: 32 * GiB, GPUMem: 8 * GiB, GPUMemSource: "nvidia-smi"}
	_, err := Size(box, ModelInfo{AttnLayers: 32, KVHeads: 8, HeadK: 128, HeadV: 128, WeightBytes: 30 * GiB}, BackendLlama, Config{})
	if err == nil || !strings.Contains(err.Error(), "do not fit") {
		t.Errorf("got %v", err)
	}
}

// Explicit config wins over auto, even past the budget, with a warning.
func TestSizeExplicitOverrides(t *testing.T) {
	cfg := Config{Model: Model{Slots: 4, Context: 262144}, CacheRAMMB: 2048, MaxConcurrent: 2}
	p, err := Size(refBox, refGGUF, BackendLlama, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if p.Slots != 4 || p.Context != 262144 || p.CacheRAMMB != 2048 {
		t.Errorf("llama: %d x %d cache %d", p.Slots, p.Context, p.CacheRAMMB)
	}
	p, err = Size(refBox, refMLX, BackendMLX, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if p.MaxConcurrent != 2 || p.ContextWindow != 262144 {
		t.Errorf("mlx: %d, window %d", p.MaxConcurrent, p.ContextWindow)
	}
	// Past the budget: run as told, say so.
	cfg = Config{Model: Model{Slots: 8, Context: 262144}}
	p, err = Size(refBox, refGGUF, BackendLlama, cfg)
	if err != nil || !strings.Contains(strings.Join(p.Notes, "\n"), "WARNING") {
		t.Errorf("err %v notes %v", err, p.Notes)
	}
}
