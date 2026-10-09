package linfer

import (
	"errors"
	"strings"
	"testing"
)

// fakeProbe answers commands from a table keyed by the command line; a
// command not in the table is missing.
func fakeProbe(goos, goarch string, cmds map[string]string, files map[string]string) Probe {
	return Probe{
		GOOS: goos, GOARCH: goarch,
		Run: func(name string, args ...string) (string, error) {
			out, ok := cmds[strings.Join(append([]string{name}, args...), " ")]
			if !ok {
				return "", errors.New("not found")
			}
			return out, nil
		},
		ReadFile: func(p string) ([]byte, error) {
			s, ok := files[p]
			if !ok {
				return nil, errors.New("not found")
			}
			return []byte(s), nil
		},
		DiskFree: func(string) (uint64, error) { return 600 * GiB, nil },
	}
}

// Apple Silicon: the chip and RAM from sysctl, and the Metal working set
// measured through MLX when a Python with it is at hand.
func TestDetectAppleSiliconMeasured(t *testing.T) {
	p := fakeProbe("darwin", "arm64", map[string]string{
		"sysctl -n machdep.cpu.brand_string":     "Apple M3 Ultra\n",
		"sysctl -n hw.ncpu":                      "32\n",
		"sysctl -n hw.memsize":                   "274877906944\n",
		"sysctl -n iogpu.wired_limit_mb":         "0\n",
		"/venv/bin/python -c " + metalWorkingSet: "239143780352\n",
	}, nil)
	h, err := Detect(p, "/data", "", "/venv/bin/python")
	if err != nil {
		t.Fatal(err)
	}
	if h.GPU != GPUMetal || h.Chip != "Apple M3 Ultra" || h.CPUs != 32 || h.RAM != 256*GiB {
		t.Errorf("got %+v", h)
	}
	if h.GPUMem != 239143780352 || !strings.HasPrefix(h.GPUMemSource, "measured") {
		t.Errorf("gpu mem %d (%s)", h.GPUMem, h.GPUMemSource)
	}
	if h.DiskFree != 600*GiB || h.DiskPath != "/data" {
		t.Errorf("disk %d at %s", h.DiskFree, h.DiskPath)
	}
}

// Without MLX the working set is the sysctl an operator set, else an
// estimate, and the source says which.
func TestDetectAppleSiliconEstimated(t *testing.T) {
	cmds := map[string]string{
		"sysctl -n machdep.cpu.brand_string": "Apple M2\n",
		"sysctl -n hw.ncpu":                  "8\n",
		"sysctl -n hw.memsize":               "68719476736\n",
		"sysctl -n iogpu.wired_limit_mb":     "0\n",
	}
	h, _ := Detect(fakeProbe("darwin", "arm64", cmds, nil), "/d", "", "")
	if h.GPUMem != 64*GiB/100*87 || !strings.HasPrefix(h.GPUMemSource, "estimate") {
		t.Errorf("estimate: %d (%s)", h.GPUMem, h.GPUMemSource)
	}
	cmds["sysctl -n iogpu.wired_limit_mb"] = "60000\n"
	h, _ = Detect(fakeProbe("darwin", "arm64", cmds, nil), "/d", "", "")
	if h.GPUMem != 60000*MiB || h.GPUMemSource != "sysctl iogpu.wired_limit_mb" {
		t.Errorf("sysctl: %d (%s)", h.GPUMem, h.GPUMemSource)
	}
}

// Linux with an NVIDIA card: VRAM and the driver's CUDA version from nvidia-smi.
func TestDetectLinuxCUDA(t *testing.T) {
	p := fakeProbe("linux", "amd64", map[string]string{
		"nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits": "NVIDIA GeForce RTX 4090, 24564\n",
		"nvidia-smi": "| NVIDIA-SMI 570.86.15   Driver Version: 570.86.15   CUDA Version: 12.8 |\n",
	}, map[string]string{
		"/proc/meminfo": "MemTotal:       65536000 kB\nMemFree:        1000 kB\n",
		"/proc/cpuinfo": "processor\t: 0\nmodel name\t: AMD Ryzen 9 7950X\nprocessor\t: 1\nmodel name\t: AMD Ryzen 9 7950X\n",
	})
	h, err := Detect(p, "/d", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if h.GPU != GPUCUDA || h.GPUName != "NVIDIA GeForce RTX 4090" || h.GPUMem != 24564*MiB || h.CUDAVersion != "12.8" {
		t.Errorf("got %+v", h)
	}
	if h.RAM != 65536000*1024 || h.Chip != "AMD Ryzen 9 7950X" || h.CPUs != 2 {
		t.Errorf("got %+v", h)
	}
}

// Linux with no GPU tool answering is a CPU build sized against RAM; an
// override skips the detection of the other kinds.
func TestDetectLinuxCPUAndOverride(t *testing.T) {
	files := map[string]string{"/proc/meminfo": "MemTotal:       32000000 kB\n", "/proc/cpuinfo": "processor\t: 0\n"}
	h, err := Detect(fakeProbe("linux", "amd64", nil, files), "/d", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if h.GPU != GPUCPU || h.Budget() != h.RAM/5*4 {
		t.Errorf("got %+v budget %d", h, h.Budget())
	}
	cmds := map[string]string{
		"nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits": "NVIDIA X, 8192\n",
		"vulkaninfo": "GPU0:\n\tdeviceName = Intel Arc A770\n\tmemoryHeaps[0]:\n\t\tsize   = 17179869184\n",
	}
	h, _ = Detect(fakeProbe("linux", "amd64", cmds, files), "/d", GPUVulkan, "")
	if h.GPU != GPUVulkan || h.GPUName != "Intel Arc A770" || h.GPUMem != 16*GiB {
		t.Errorf("override: %+v", h)
	}
}
