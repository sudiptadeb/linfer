package linfer

import (
	"reflect"
	"strings"
	"testing"
)

// One asset name per platform, two for CUDA (the build and its runtime).
func TestLlamaAssets(t *testing.T) {
	cases := []struct {
		goos, goarch, gpu, cuda string
		want                    []string
	}{
		{"darwin", "arm64", GPUMetal, "", []string{"llama-b11429-bin-macos-arm64.tar.gz"}},
		{"darwin", "amd64", GPUCPU, "", []string{"llama-b11429-bin-macos-x64.tar.gz"}},
		{"linux", "amd64", GPUCPU, "", []string{"llama-b11429-bin-ubuntu-x64.tar.gz"}},
		{"linux", "arm64", GPUCPU, "", []string{"llama-b11429-bin-ubuntu-arm64.tar.gz"}},
		{"linux", "amd64", GPUVulkan, "", []string{"llama-b11429-bin-ubuntu-vulkan-x64.tar.gz"}},
		{"linux", "amd64", GPUROCm, "", []string{"llama-b11429-bin-ubuntu-rocm-10.0-x64.tar.gz"}},
		{"linux", "amd64", GPUCUDA, "12.8", []string{"llama-b11429-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-b11429-bin-ubuntu-cuda-12.8-x64.tar.gz"}},
		{"linux", "amd64", GPUCUDA, "13.0", []string{"llama-b11429-bin-ubuntu-cuda-13.4-x64.tar.gz", "cudart-llama-b11429-bin-ubuntu-cuda-13.4-x64.tar.gz"}},
		{"linux", "arm64", GPUCUDA, "13.1", []string{"llama-b11429-bin-ubuntu-cuda-13.4-arm64.tar.gz", "cudart-llama-b11429-bin-ubuntu-cuda-13.4-arm64.tar.gz"}},
	}
	for _, c := range cases {
		got, err := LlamaAssets("b11429", c.goos, c.goarch, c.gpu, c.cuda)
		if err != nil {
			t.Errorf("%s/%s %s %s: %v", c.goos, c.goarch, c.gpu, c.cuda, err)
			continue
		}
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s/%s %s %s: got %v, want %v", c.goos, c.goarch, c.gpu, c.cuda, got, c.want)
		}
	}
}

// A driver older than the prebuilt CUDA line is told what to do, not handed
// a build that will not load.
func TestLlamaAssetsOldDriver(t *testing.T) {
	_, err := LlamaAssets("b11429", "linux", "amd64", GPUCUDA, "12.4")
	if err == nil || !strings.Contains(err.Error(), "gpu: vulkan") {
		t.Errorf("got %v", err)
	}
	if _, err := LlamaAssets("b11429", "freebsd", "amd64", GPUCPU, ""); err == nil {
		t.Error("freebsd accepted")
	}
}
