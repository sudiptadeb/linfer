package linfer

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// LlamaAssets names the prebuilt release assets for a machine: the build
// itself and, for CUDA, the runtime libraries that must sit beside it. The
// names follow ggml-org/llama.cpp's release layout
// (llama-<release>-bin-<os>[-<backend>]-<arch>.tar.gz); the macOS build is
// Metal, the Linux ones are built on Ubuntu and run on any glibc distro.
//
// CUDA builds come in two toolkit lines; the driver's CUDA version (from
// nvidia-smi) says which one it can run: 13.x drivers take the 13.4 build,
// 12.8 and later the 12.8 build, and an older driver cannot run either, so
// the operator is told to update it or set gpu: vulkan.
func LlamaAssets(release, goos, goarch, gpu, cudaVersion string) ([]string, error) {
	arch := map[string]string{"amd64": "x64", "arm64": "arm64"}[goarch]
	if arch == "" {
		return nil, fmt.Errorf("no llama.cpp build for %s/%s", goos, goarch)
	}
	bin := func(variant string) string {
		return "llama-" + release + "-bin-" + variant + ".tar.gz"
	}
	switch goos {
	case "darwin":
		return []string{bin("macos-" + arch)}, nil
	case "linux":
		switch gpu {
		case GPUCUDA:
			cuda, err := cudaLine(cudaVersion, goarch)
			if err != nil {
				return nil, err
			}
			v := "ubuntu-cuda-" + cuda + "-" + arch
			return []string{bin(v), "cudart-llama-" + release + "-bin-" + v + ".tar.gz"}, nil
		case GPUROCm:
			if arch != "x64" {
				return nil, fmt.Errorf("no ROCm build for linux/%s", goarch)
			}
			return []string{bin("ubuntu-rocm-10.0-x64")}, nil
		case GPUVulkan:
			return []string{bin("ubuntu-vulkan-" + arch)}, nil
		default:
			return []string{bin("ubuntu-" + arch)}, nil
		}
	}
	return nil, fmt.Errorf("no llama.cpp build for %s/%s", goos, goarch)
}

func cudaLine(version, goarch string) (string, error) {
	major, minor, _ := strings.Cut(version, ".")
	maj, _ := strconv.Atoi(major)
	min, _ := strconv.Atoi(minor)
	switch {
	case version == "":
		// nvidia-smi answered but gave no version line; 12.8 runs on the
		// widest range of drivers.
		if goarch == "arm64" {
			return "13.4", nil
		}
		return "12.8", nil
	case maj >= 13:
		return "13.4", nil
	case maj == 12 && min >= 8 && goarch == "amd64":
		return "12.8", nil
	}
	return "", fmt.Errorf("the NVIDIA driver supports CUDA %s; the prebuilt llama.cpp needs 12.8 or 13.x (update the driver, or set gpu: vulkan)", version)
}

// AssetURL is where a release asset is downloaded from.
func (f *Fetcher) AssetURL(release, name string) string {
	return f.ReleaseBase + "/" + release + "/" + name
}

// InstallLlama downloads the release for this machine into <Bin>, unpacks
// it, and checks the binary runs and is the build asked for. It returns the
// llama-server path. The archives unpack to a flat llama-<release>/
// directory with the libraries beside the binary (rpath @executable_path
// on macOS, $ORIGIN on Linux), so nothing needs fixing after extraction.
func (f *Fetcher) InstallLlama(ctx context.Context, cfg Config, h Hardware) (string, error) {
	assets, err := LlamaAssets(cfg.LlamaRelease, h.OS, h.Arch, h.GPU, h.CUDAVersion)
	if err != nil {
		return "", err
	}
	bin := cfg.Paths().Bin
	dir := filepath.Join(bin, "llama-"+cfg.LlamaRelease)
	server := filepath.Join(dir, "llama-server")
	if err := os.MkdirAll(bin, 0o755); err != nil {
		return "", err
	}
	for _, name := range assets {
		archive := filepath.Join(bin, name)
		if err := f.Download(ctx, f.AssetURL(cfg.LlamaRelease, name), archive, Verify{}); err != nil {
			return "", fmt.Errorf("llama.cpp %s: %w", name, err)
		}
		if err := untarGz(archive, bin); err != nil {
			return "", fmt.Errorf("unpack %s: %w", name, err)
		}
		os.Remove(archive)
		// The cudart archive unpacks into its own directory
		// (cudart-llama-<release>-bin-…/libcublas.so.12 …); the loader looks
		// beside llama-server, so its libraries move in there.
		if own := filepath.Join(bin, strings.TrimSuffix(name, ".tar.gz")); own != dir {
			if err := mergeDir(own, dir); err != nil {
				return "", fmt.Errorf("place %s: %w", name, err)
			}
		}
	}
	if _, err := os.Stat(server); err != nil {
		return "", fmt.Errorf("%s has no llama-server after unpacking %v", dir, assets)
	}
	if err := CheckLlama(server, cfg.LlamaRelease); err != nil {
		return "", err
	}
	return server, nil
}

// CheckLlama runs `llama-server --version` and checks the build number: the
// output reads "version: 0.6.0-dev (build 11429, commit d81235049)".
func CheckLlama(server, release string) error {
	out, err := exec.Command(server, "--version").CombinedOutput()
	if err != nil {
		return fmt.Errorf("%s --version: %w\n%s", server, err, strings.TrimSpace(string(out)))
	}
	want := "build " + strings.TrimPrefix(release, "b")
	if !strings.Contains(string(out), want) {
		return fmt.Errorf("%s is not release %s: %s", server, release, strings.TrimSpace(string(out)))
	}
	return nil
}

// mergeDir moves the entries of src into dst and removes src. Nothing to do
// when src is not there (the archive unpacked flat already).
func mergeDir(src, dst string) error {
	entries, err := os.ReadDir(src)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	for _, e := range entries {
		to := filepath.Join(dst, e.Name())
		os.Remove(to)
		if err := os.Rename(filepath.Join(src, e.Name()), to); err != nil {
			return err
		}
	}
	return os.Remove(src)
}

// untarGz unpacks a .tar.gz under dir, refusing paths that escape it.
func untarGz(archive, dir string) error {
	f, err := os.Open(archive)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	tr := tar.NewReader(gz)
	root, _ := filepath.Abs(dir)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		target := filepath.Join(root, filepath.FromSlash(hdr.Name))
		if !strings.HasPrefix(target, root+string(filepath.Separator)) {
			return fmt.Errorf("entry %q escapes the archive", hdr.Name)
		}
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			out, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, os.FileMode(hdr.Mode)&0o777|0o600)
			if err != nil {
				return err
			}
			if _, err := io.Copy(out, tr); err != nil {
				out.Close()
				return err
			}
			if err := out.Close(); err != nil {
				return err
			}
		case tar.TypeSymlink:
			os.Remove(target)
			if err := os.Symlink(hdr.Linkname, target); err != nil {
				return err
			}
		}
	}
}
