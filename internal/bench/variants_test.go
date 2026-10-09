package bench

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/sudiptadeb/linfer/internal/linfer"
)

func TestVariantsComparesMLXEvenWhenPinnedToLlama(t *testing.T) {
	dir := t.TempDir()
	for _, f := range []string{"model.gguf", "llama-server", "omlx", "mlx/config.json"} {
		os.MkdirAll(filepath.Dir(filepath.Join(dir, f)), 0o755)
		os.WriteFile(filepath.Join(dir, f), []byte("x"), 0o755)
	}
	cfg := linfer.Config{Backend: linfer.BackendLlama, LlamaBin: filepath.Join(dir, "llama-server"), OMLXBin: filepath.Join(dir, "omlx"),
		Model: linfer.Model{ID: "m", GGUF: filepath.Join(dir, "model.gguf"), MLX: filepath.Join(dir, "mlx")}}
	got := Variants(cfg, linfer.Hardware{OS: "darwin", Arch: "arm64"})
	if len(got) != 2 || got[0].Name != "llama" || got[1].Name != "mlx" {
		t.Fatalf("variants %+v", got)
	}
}

func TestRunWaitsForMemoryToComeFree(t *testing.T) {
	_, _, r, _ := testSetup(t)
	calls := 0
	r.FreeMemory = func() uint64 {
		calls++
		if calls < 3 {
			return linfer.MiB // the previous model is still letting go
		}
		return 1 << 40
	}
	r.MemoryWait = 30 * time.Second
	res, err := r.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Backends) != 1 || res.Backends[0].Error != "" {
		t.Fatalf("backends %+v", res.Backends)
	}
}
