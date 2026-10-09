package linfer

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// ModelInfo is what sizing needs to know about a set of weights: how big
// they are on disk, and how much KV cache a token costs. Both backends keep
// the cache in 16-bit K and V per attention layer, so one formula serves.
type ModelInfo struct {
	Arch string // general.architecture / model_type
	Name string
	// Layers is the number of blocks; AttnLayers is how many of them are full
	// attention and hold a KV cache. They differ on hybrid models (Qwen3-Next
	// and its relatives run linear attention on three blocks in four, whose
	// state is a small fixed size per slot and is left out here).
	Layers, AttnLayers int
	KVHeads            int
	HeadK, HeadV       int   // dimensions of one key and one value head
	Context            int   // the trained context length
	WeightBytes        int64 // all the weight files, as they are on disk
}

// KVBytesPerToken is the KV cache one token costs across all attention
// layers at f16: K and V, 2 bytes each element.
func (m ModelInfo) KVBytesPerToken() int64 {
	return int64(m.AttnLayers) * int64(m.KVHeads) * int64(m.HeadK+m.HeadV) * 2
}

// --- GGUF -------------------------------------------------------------------------

// GGUF value types (ggml.h).
const (
	ggufUint8 = iota
	ggufInt8
	ggufUint16
	ggufInt16
	ggufUint32
	ggufInt32
	ggufFloat32
	ggufBool
	ggufStr
	ggufArray
	ggufUint64
	ggufInt64
	ggufFloat64
)

// ReadGGUF reads a GGUF file's metadata: the header and the key-value
// section, a few megabytes at most, never the tensors. A split model
// (…-00001-of-00003.gguf) is read from its first part and its size is the
// sum of the parts.
func ReadGGUF(path string) (ModelInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return ModelInfo{}, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)

	var magic [4]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil || string(magic[:]) != "GGUF" {
		return ModelInfo{}, fmt.Errorf("%s: not a GGUF file", path)
	}
	var version uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return ModelInfo{}, err
	}
	if version < 2 {
		return ModelInfo{}, fmt.Errorf("%s: GGUF v%d is too old", path, version)
	}
	var nTensors, nKV uint64
	if err := binary.Read(r, binary.LittleEndian, &nTensors); err != nil {
		return ModelInfo{}, err
	}
	if err := binary.Read(r, binary.LittleEndian, &nKV); err != nil {
		return ModelInfo{}, err
	}
	kv := map[string]any{}
	for i := uint64(0); i < nKV; i++ {
		key, err := ggufString(r)
		if err != nil {
			return ModelInfo{}, fmt.Errorf("%s: key %d: %w", path, i, err)
		}
		var t uint32
		if err := binary.Read(r, binary.LittleEndian, &t); err != nil {
			return ModelInfo{}, err
		}
		// The tokenizer's arrays are the bulk of the section (one string
		// per token); they are skipped, not kept.
		keep := !strings.HasPrefix(key, "tokenizer.")
		v, err := ggufValue(r, t, keep)
		if err != nil {
			return ModelInfo{}, fmt.Errorf("%s: %s: %w", path, key, err)
		}
		if keep {
			kv[key] = v
		}
	}

	arch, _ := kv["general.architecture"].(string)
	if arch == "" {
		return ModelInfo{}, fmt.Errorf("%s: no general.architecture", path)
	}
	m := ModelInfo{Arch: arch}
	m.Name, _ = kv["general.name"].(string)
	m.Layers = ggufInt(kv[arch+".block_count"])
	m.Context = ggufInt(kv[arch+".context_length"])
	heads := ggufInt(kv[arch+".attention.head_count"])
	// head_count_kv is one number, or one per layer on models whose layers
	// differ: a zero marks a layer with no KV cache.
	switch v := kv[arch+".attention.head_count_kv"].(type) {
	case []any:
		for _, x := range v {
			if n := ggufInt(x); n > 0 {
				m.AttnLayers++
				m.KVHeads = max(m.KVHeads, n)
			}
		}
	default:
		m.KVHeads = ggufInt(v)
		m.AttnLayers = m.Layers
	}
	if m.KVHeads == 0 {
		m.KVHeads = heads
	}
	m.HeadK = ggufInt(kv[arch+".attention.key_length"])
	m.HeadV = ggufInt(kv[arch+".attention.value_length"])
	if m.HeadK == 0 && heads > 0 {
		m.HeadK = ggufInt(kv[arch+".embedding_length"]) / heads
	}
	if m.HeadV == 0 {
		m.HeadV = m.HeadK
	}
	// Hybrid models say which blocks are full attention.
	if iv := ggufInt(kv[arch+".full_attention_interval"]); iv > 0 && m.AttnLayers == m.Layers {
		m.AttnLayers = m.Layers / iv
	}
	m.WeightBytes, err = ggufSize(path)
	if err != nil {
		return ModelInfo{}, err
	}
	return m, nil
}

func ggufString(r *bufio.Reader) (string, error) {
	var n uint64
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return "", err
	}
	if n > 1<<20 {
		return "", fmt.Errorf("string of %d bytes", n)
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(r, b); err != nil {
		return "", err
	}
	return string(b), nil
}

// ggufValue reads one value. With keep false it only consumes the bytes and
// returns nil, so a 150,000-string vocabulary costs no allocation.
func ggufValue(r *bufio.Reader, t uint32, keep bool) (any, error) {
	switch t {
	case ggufUint8, ggufInt8, ggufBool:
		b, err := r.ReadByte()
		return int(b), err
	case ggufUint16, ggufInt16:
		var v uint16
		err := binary.Read(r, binary.LittleEndian, &v)
		return int(v), err
	case ggufUint32:
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return int(v), err
	case ggufInt32:
		var v int32
		err := binary.Read(r, binary.LittleEndian, &v)
		return int(v), err
	case ggufFloat32:
		var v float32
		err := binary.Read(r, binary.LittleEndian, &v)
		return float64(v), err
	case ggufUint64, ggufInt64:
		var v uint64
		err := binary.Read(r, binary.LittleEndian, &v)
		return int(v), err
	case ggufFloat64:
		var v float64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufStr:
		if keep {
			return ggufString(r)
		}
		var n uint64
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return nil, err
		}
		_, err := r.Discard(int(n))
		return nil, err
	case ggufArray:
		var et uint32
		if err := binary.Read(r, binary.LittleEndian, &et); err != nil {
			return nil, err
		}
		var n uint64
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return nil, err
		}
		// Only short arrays are worth keeping: per-layer settings, never a
		// vocabulary.
		keep = keep && n <= 1024
		var out []any
		for i := uint64(0); i < n; i++ {
			v, err := ggufValue(r, et, keep)
			if err != nil {
				return nil, err
			}
			if keep {
				out = append(out, v)
			}
		}
		if !keep {
			return nil, nil
		}
		return out, nil
	}
	return nil, fmt.Errorf("unknown value type %d", t)
}

func ggufInt(v any) int {
	switch x := v.(type) {
	case int:
		return x
	case float64:
		return int(x)
	}
	return 0
}

var splitName = regexp.MustCompile(`^(.*)-\d{5}-of-(\d{5})\.gguf$`)

// ggufSize is the file's size, or the sum of the parts of a split model.
func ggufSize(path string) (int64, error) {
	m := splitName.FindStringSubmatch(filepath.Base(path))
	if m == nil {
		st, err := os.Stat(path)
		if err != nil {
			return 0, err
		}
		return st.Size(), nil
	}
	parts, err := filepath.Glob(filepath.Join(filepath.Dir(path), m[1]+"-*-of-"+m[2]+".gguf"))
	if err != nil {
		return 0, err
	}
	var total int64
	for _, p := range parts {
		st, err := os.Stat(p)
		if err != nil {
			return 0, err
		}
		total += st.Size()
	}
	return total, nil
}

// --- MLX ------------------------------------------------------------------------

// ReadMLX reads an MLX model directory: config.json for the shape (the
// text_config of a vision model), and the safetensors for the size.
func ReadMLX(dir string) (ModelInfo, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return ModelInfo{}, err
	}
	var top map[string]any
	if err := json.Unmarshal(raw, &top); err != nil {
		return ModelInfo{}, fmt.Errorf("%s/config.json: %w", dir, err)
	}
	cfg := top
	if tc, ok := top["text_config"].(map[string]any); ok {
		cfg = tc
	}
	m := ModelInfo{Name: filepath.Base(dir)}
	m.Arch, _ = cfg["model_type"].(string)
	if m.Arch == "" {
		m.Arch, _ = top["model_type"].(string)
	}
	m.Layers = jsonInt(cfg["num_hidden_layers"])
	m.Context = jsonInt(cfg["max_position_embeddings"])
	heads := jsonInt(cfg["num_attention_heads"])
	m.KVHeads = jsonInt(cfg["num_key_value_heads"])
	if m.KVHeads == 0 {
		m.KVHeads = heads
	}
	m.HeadK = jsonInt(cfg["head_dim"])
	if m.HeadK == 0 && heads > 0 {
		m.HeadK = jsonInt(cfg["hidden_size"]) / heads
	}
	m.HeadV = m.HeadK
	m.AttnLayers = m.Layers
	if types, ok := cfg["layer_types"].([]any); ok && len(types) > 0 {
		m.AttnLayers = 0
		for _, t := range types {
			if t == "full_attention" {
				m.AttnLayers++
			}
		}
	} else if iv := jsonInt(cfg["full_attention_interval"]); iv > 0 {
		m.AttnLayers = m.Layers / iv
	}
	if m.Layers == 0 || m.KVHeads == 0 || m.HeadK == 0 {
		return ModelInfo{}, fmt.Errorf("%s/config.json: no layer, head or head_dim counts", dir)
	}
	m.WeightBytes, err = DirSize(dir, ".safetensors")
	if err != nil {
		return ModelInfo{}, err
	}
	if m.WeightBytes == 0 {
		return ModelInfo{}, fmt.Errorf("%s: no .safetensors files", dir)
	}
	return m, nil
}

func jsonInt(v any) int {
	f, ok := v.(float64)
	if !ok {
		return 0
	}
	return int(f)
}

// DirSize sums the files directly in dir with the suffix (all of them when
// suffix is empty), following symlinks.
func DirSize(dir, suffix string) (int64, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0, err
	}
	var total int64
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), suffix) {
			continue
		}
		st, err := os.Stat(filepath.Join(dir, e.Name()))
		if err != nil {
			return 0, err
		}
		if !st.IsDir() {
			total += st.Size()
		}
	}
	return total, nil
}

// ReadModel reads whichever weights the backend uses.
func ReadModel(cfg Config, backend string) (ModelInfo, error) {
	switch backend {
	case BackendLlama:
		m, err := ReadGGUF(cfg.GGUFPath())
		if err != nil {
			return m, err
		}
		if p := cfg.MMProjPath(); p != "" {
			st, err := os.Stat(p)
			if err != nil {
				return m, err
			}
			m.WeightBytes += st.Size()
		}
		return m, nil
	case BackendMLX:
		return ReadMLX(cfg.MLXPath())
	}
	return ModelInfo{}, errors.New("no backend chosen")
}
