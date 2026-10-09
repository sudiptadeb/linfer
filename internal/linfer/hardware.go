package linfer

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
)

// GPU kinds, as the llama.cpp builds name them. Metal is implied by
// darwin/arm64; the Linux kinds are detected or set by Config.GPU.
const (
	GPUMetal  = "metal"
	GPUCUDA   = "cuda"
	GPUROCm   = "rocm"
	GPUVulkan = "vulkan"
	GPUCPU    = "cpu"
)

// Hardware is what Detect found. GPUMem is the memory the backend can put
// weights and cache in: Metal's recommended working set on Apple Silicon
// (unified memory, so it is most of the RAM), the card's VRAM on CUDA and
// ROCm, zero when unknown. Budget turns it into a number to size against.
type Hardware struct {
	OS, Arch string
	Chip     string
	CPUs     int
	RAM      uint64
	GPU      string // one of the GPU kinds
	GPUName  string
	GPUMem   uint64
	// GPUMemSource says how GPUMem was found, so a report can show whether it
	// was measured or estimated.
	GPUMemSource string
	CUDAVersion  string // the driver's, e.g. "12.8"; picks the CUDA build
	DiskFree     uint64 // at DiskPath
	DiskPath     string
}

// Budget is the memory sizing works within: the GPU's, or 80% of RAM when
// nothing better is known (CPU and Vulkan builds).
func (h Hardware) Budget() uint64 {
	if h.GPUMem > 0 {
		return h.GPUMem
	}
	return h.RAM / 5 * 4
}

// Probe is how Detect reaches the machine, so a test can hand it another
// machine's answers.
type Probe struct {
	GOOS, GOARCH string
	// Run runs a command and returns its combined output; an error when it
	// is missing or fails.
	Run      func(name string, args ...string) (string, error)
	ReadFile func(path string) ([]byte, error)
	// DiskFree is the free bytes on the filesystem holding path.
	DiskFree func(path string) (uint64, error)
}

// SystemProbe reaches the real machine.
func SystemProbe() Probe {
	return Probe{
		GOOS:   runtime.GOOS,
		GOARCH: runtime.GOARCH,
		Run: func(name string, args ...string) (string, error) {
			out, err := exec.Command(name, args...).CombinedOutput()
			return string(out), err
		},
		ReadFile: os.ReadFile,
		DiskFree: diskFree,
	}
}

// metalWorkingSet is what MLX reports as the memory Metal will let one
// process wire: the figure that bounds how big a model this machine serves.
// It needs an MLX install to ask; mlxPython is one, empty when there is none.
const metalWorkingSet = "import mlx.core as mx; print(mx.device_info()['max_recommended_working_set_size'])"

// Detect describes the machine. dir is where the weights go (for free disk);
// gpuOverride is Config.GPU; mlxPython is a Python with mlx importable, or "".
func Detect(p Probe, dir, gpuOverride, mlxPython string) (Hardware, error) {
	h := Hardware{OS: p.GOOS, Arch: p.GOARCH, DiskPath: dir}
	switch p.GOOS {
	case "darwin":
		h.Chip = strings.TrimSpace(firstOut(p, "sysctl", "-n", "machdep.cpu.brand_string"))
		h.CPUs = atoi(firstOut(p, "sysctl", "-n", "hw.ncpu"))
		h.RAM = uint64(atoi64(firstOut(p, "sysctl", "-n", "hw.memsize")))
		if p.GOARCH == "arm64" {
			h.GPU, h.GPUName = GPUMetal, h.Chip
			h.GPUMem, h.GPUMemSource = metalMemory(p, h.RAM, mlxPython)
		} else {
			h.GPU = GPUCPU
		}
	case "linux":
		mem, _ := p.ReadFile("/proc/meminfo")
		h.RAM = uint64(meminfoKB(string(mem), "MemTotal")) * 1024
		cpu, _ := p.ReadFile("/proc/cpuinfo")
		h.Chip, h.CPUs = cpuinfo(string(cpu))
		detectLinuxGPU(p, &h, gpuOverride)
	default:
		return h, fmt.Errorf("%s is not supported: linfer runs on macOS and Linux", p.GOOS)
	}
	if h.RAM == 0 {
		return h, fmt.Errorf("could not read the machine's memory size")
	}
	if p.DiskFree != nil {
		if free, err := p.DiskFree(dir); err == nil {
			h.DiskFree = free
		}
	}
	return h, nil
}

// metalMemory is the Metal working set: measured through MLX when it can be,
// else the iogpu.wired_limit_mb sysctl when an operator has set one, else an
// estimate. 87% is what macOS gave a 256 GB M3 Ultra by default (222.7 GiB),
// and is close to the 75% documented for smaller machines once the kernel's
// own share is taken; the report marks it as an estimate.
func metalMemory(p Probe, ram uint64, mlxPython string) (uint64, string) {
	if mlxPython != "" {
		if out, err := p.Run(mlxPython, "-c", metalWorkingSet); err == nil {
			if n := atoi64(out); n > 0 {
				return uint64(n), "measured (mlx device_info)"
			}
		}
	}
	if mb := atoi64(firstOut(p, "sysctl", "-n", "iogpu.wired_limit_mb")); mb > 0 {
		return uint64(mb) << 20, "sysctl iogpu.wired_limit_mb"
	}
	return ram / 100 * 87, "estimate (87% of RAM)"
}

var (
	cudaVersionRE = regexp.MustCompile(`CUDA Version:\s*([0-9]+\.[0-9]+)`)
	vkDeviceRE    = regexp.MustCompile(`deviceName\s*=\s*(.+)`)
	vkHeapRE      = regexp.MustCompile(`memoryHeaps\[0\]:\s*\n\s*size\s*=\s*([0-9]+)`)
)

// detectLinuxGPU fills GPU, GPUName, GPUMem and CUDAVersion. The override
// names the kind and skips the others' detection, but still asks the chosen
// tool for the memory. Several cards: the first one, because one model is
// served on one device.
func detectLinuxGPU(p Probe, h *Hardware, override string) {
	try := func(kind string) bool {
		return override == "" || override == kind
	}
	if try(GPUCUDA) {
		if out, err := p.Run("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"); err == nil {
			if line := firstLine(out); line != "" {
				name, mib, _ := strings.Cut(line, ",")
				h.GPU, h.GPUName = GPUCUDA, strings.TrimSpace(name)
				h.GPUMem, h.GPUMemSource = uint64(atoi64(mib))<<20, "nvidia-smi memory.total"
				if m := cudaVersionRE.FindStringSubmatch(firstOut(p, "nvidia-smi")); m != nil {
					h.CUDAVersion = m[1]
				}
				return
			}
		}
	}
	if try(GPUROCm) {
		if out, err := p.Run("rocm-smi", "--showmeminfo", "vram", "--csv"); err == nil {
			// device,VRAM Total Memory (B),VRAM Total Used Memory (B)
			for _, line := range strings.Split(out, "\n") {
				f := strings.Split(line, ",")
				if len(f) >= 2 && strings.HasPrefix(f[0], "card") {
					h.GPU, h.GPUName = GPUROCm, f[0]
					h.GPUMem, h.GPUMemSource = uint64(atoi64(f[1])), "rocm-smi VRAM Total"
					return
				}
			}
		}
	}
	if try(GPUVulkan) {
		if out, err := p.Run("vulkaninfo"); err == nil {
			if m := vkDeviceRE.FindStringSubmatch(out); m != nil {
				h.GPU, h.GPUName = GPUVulkan, strings.TrimSpace(m[1])
				if hm := vkHeapRE.FindStringSubmatch(out); hm != nil {
					h.GPUMem, h.GPUMemSource = uint64(atoi64(hm[1])), "vulkaninfo memoryHeaps[0]"
				} else {
					h.GPUMemSource = "unknown (vulkaninfo gave no heap size)"
				}
				return
			}
		}
	}
	h.GPU = GPUCPU
	if override != "" && override != GPUCPU {
		h.GPUMemSource = override + " requested but its tool answered nothing"
	}
}

func firstOut(p Probe, name string, args ...string) string {
	out, err := p.Run(name, args...)
	if err != nil {
		return ""
	}
	return out
}

func firstLine(s string) string {
	line, _, _ := strings.Cut(strings.TrimSpace(s), "\n")
	return strings.TrimSpace(line)
}

func atoi(s string) int { return int(atoi64(s)) }

func atoi64(s string) int64 {
	n, _ := strconv.ParseInt(strings.TrimSpace(firstLine(s)), 10, 64)
	return n
}

// meminfoKB reads one /proc/meminfo line: "MemTotal:       65536000 kB".
func meminfoKB(meminfo, key string) int64 {
	for _, line := range strings.Split(meminfo, "\n") {
		if k, v, ok := strings.Cut(line, ":"); ok && k == key {
			return atoi64(strings.TrimSuffix(strings.TrimSpace(v), " kB"))
		}
	}
	return 0
}

// cpuinfo reads the model name and the processor count from /proc/cpuinfo.
func cpuinfo(info string) (string, int) {
	var name string
	var n int
	for _, line := range strings.Split(info, "\n") {
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		switch strings.TrimSpace(k) {
		case "model name":
			if name == "" {
				name = strings.TrimSpace(v)
			}
		case "processor":
			n++
		}
	}
	return name, n
}

// nearestExisting walks up from path to a directory that exists, so free
// disk can be read for a directory setup has not created yet.
func nearestExisting(path string) string {
	for p := path; ; p = filepath.Dir(p) {
		if _, err := os.Stat(p); err == nil {
			return p
		}
		if filepath.Dir(p) == p {
			return p
		}
	}
}

// --- units ------------------------------------------------------------------------

const (
	GiB = 1 << 30
	MiB = 1 << 20
)

// HumanBytes prints a size the way an operator reads one: "222.7 GiB".
func HumanBytes(n uint64) string {
	switch {
	case n >= GiB:
		return fmt.Sprintf("%.1f GiB", float64(n)/GiB)
	case n >= MiB:
		return fmt.Sprintf("%.0f MiB", float64(n)/MiB)
	}
	return fmt.Sprintf("%d B", n)
}
