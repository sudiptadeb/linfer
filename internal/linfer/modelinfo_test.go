package linfer

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// writeGGUF writes a GGUF v3 file with no tensors and the given metadata,
// in the order given. Values: string, uint32, or []uint32 (an array).
func writeGGUF(t *testing.T, path string, kv []struct {
	k string
	v any
}) {
	t.Helper()
	var b bytes.Buffer
	le := binary.LittleEndian
	b.WriteString("GGUF")
	binary.Write(&b, le, uint32(3))
	binary.Write(&b, le, uint64(0))
	binary.Write(&b, le, uint64(len(kv)))
	str := func(s string) {
		binary.Write(&b, le, uint64(len(s)))
		b.WriteString(s)
	}
	for _, e := range kv {
		str(e.k)
		switch v := e.v.(type) {
		case string:
			binary.Write(&b, le, uint32(ggufStr))
			str(v)
		case uint32:
			binary.Write(&b, le, uint32(ggufUint32))
			binary.Write(&b, le, v)
		case []uint32:
			binary.Write(&b, le, uint32(ggufArray))
			binary.Write(&b, le, uint32(ggufUint32))
			binary.Write(&b, le, uint64(len(v)))
			for _, x := range v {
				binary.Write(&b, le, x)
			}
		case []string:
			binary.Write(&b, le, uint32(ggufArray))
			binary.Write(&b, le, uint32(ggufStr))
			binary.Write(&b, le, uint64(len(v)))
			for _, x := range v {
				str(x)
			}
		}
	}
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

type kv = struct {
	k string
	v any
}

// A hybrid model's header, as the reference model's reads: 48 blocks of
// which one in four is full attention, 2 KV heads of 256, so 24 KiB of KV
// per token. The tokenizer's arrays are skipped on the way.
func TestReadGGUFHybrid(t *testing.T) {
	p := filepath.Join(t.TempDir(), "m.gguf")
	writeGGUF(t, p, []kv{
		{"general.architecture", "qwen4exp"},
		{"general.name", "Test Next"},
		{"tokenizer.ggml.tokens", []string{"a", "b", "c"}},
		{"qwen4exp.block_count", uint32(48)},
		{"qwen4exp.context_length", uint32(262144)},
		{"qwen4exp.attention.head_count", uint32(24)},
		{"qwen4exp.attention.head_count_kv", uint32(2)},
		{"qwen4exp.attention.key_length", uint32(256)},
		{"qwen4exp.attention.value_length", uint32(256)},
		{"qwen4exp.full_attention_interval", uint32(4)},
	})
	m, err := ReadGGUF(p)
	if err != nil {
		t.Fatal(err)
	}
	if m.Arch != "qwen4exp" || m.Name != "Test Next" || m.Layers != 48 || m.AttnLayers != 12 || m.KVHeads != 2 || m.HeadK != 256 || m.Context != 262144 {
		t.Errorf("got %+v", m)
	}
	if m.KVBytesPerToken() != 24*1024 {
		t.Errorf("kv per token %d, want 24 KiB", m.KVBytesPerToken())
	}
	if st, _ := os.Stat(p); m.WeightBytes != st.Size() {
		t.Errorf("size %d, want the file's %d", m.WeightBytes, st.Size())
	}
}

// A per-layer head_count_kv array marks the layers without a cache with 0;
// a model with no key_length derives the head size from the embedding.
func TestReadGGUFPerLayerKV(t *testing.T) {
	p := filepath.Join(t.TempDir(), "m.gguf")
	writeGGUF(t, p, []kv{
		{"general.architecture", "llama"},
		{"llama.block_count", uint32(4)},
		{"llama.embedding_length", uint32(4096)},
		{"llama.attention.head_count", uint32(32)},
		{"llama.attention.head_count_kv", []uint32{8, 0, 8, 0}},
	})
	m, err := ReadGGUF(p)
	if err != nil {
		t.Fatal(err)
	}
	if m.AttnLayers != 2 || m.KVHeads != 8 || m.HeadK != 128 || m.HeadV != 128 {
		t.Errorf("got %+v", m)
	}
}

// An MLX directory: config.json's text_config carries the shape of a vision
// model, layer_types says which layers are full attention, and the size is
// the safetensors.
func TestReadMLX(t *testing.T) {
	dir := t.TempDir()
	cfg := `{"model_type":"qwen4_exp","text_config":{"model_type":"qwen4_exp_text","num_hidden_layers":4,"num_key_value_heads":2,"head_dim":256,
	  "max_position_embeddings":262144,"layer_types":["linear_attention","linear_attention","linear_attention","full_attention"]}}`
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644)
	os.WriteFile(filepath.Join(dir, "model-00001-of-00002.safetensors"), make([]byte, 1000), 0o644)
	os.WriteFile(filepath.Join(dir, "model-00002-of-00002.safetensors"), make([]byte, 500), 0o644)
	os.WriteFile(filepath.Join(dir, "tokenizer.json"), make([]byte, 100), 0o644)
	m, err := ReadMLX(dir)
	if err != nil {
		t.Fatal(err)
	}
	if m.Arch != "qwen4_exp_text" || m.Layers != 4 || m.AttnLayers != 1 || m.KVHeads != 2 || m.HeadK != 256 || m.Context != 262144 || m.WeightBytes != 1500 {
		t.Errorf("got %+v", m)
	}
}
