package linfer

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Fetcher downloads release assets and weights. The bases are fields so a
// test can point them at a local server.
type Fetcher struct {
	Client      *http.Client
	ReleaseBase string // ggml-org/llama.cpp release downloads
	HFBase      string // Hugging Face
	HFToken     string // for gated repos; HF_TOKEN
	// Progress is called as bytes arrive, with the file's name, bytes done
	// and the total (0 when unknown). Nil is fine.
	Progress func(name string, done, total int64)
}

// NewFetcher is the production Fetcher.
func NewFetcher() *Fetcher {
	return &Fetcher{
		Client:      &http.Client{Timeout: 0}, // a 190 GB file has no sensible deadline; the context cancels
		ReleaseBase: "https://github.com/ggml-org/llama.cpp/releases/download",
		HFBase:      "https://huggingface.co",
		HFToken:     os.Getenv("HF_TOKEN"),
	}
}

// WithProgress prints progress lines to w as a download goes.
func (f *Fetcher) WithProgress(w io.Writer) *Fetcher {
	f.Progress = func(name string, done, total int64) {
		if total > 0 {
			fmt.Fprintf(w, "\r  %s: %s of %s (%d%%)", name, HumanBytes(uint64(done)), HumanBytes(uint64(total)), done*100/total)
		} else {
			fmt.Fprintf(w, "\r  %s: %s", name, HumanBytes(uint64(done)))
		}
		if done == total {
			fmt.Fprintln(w)
		}
	}
	return f
}

// Verify is what a downloaded file must match. Size 0 or an empty SHA256
// skips that check.
type Verify struct {
	Size   int64
	SHA256 string
}

// Download fetches url to dest, resuming a partial download left in
// dest.part, and verifies size and sha256 before the file takes its name. A
// dest already there with the right size is left alone: setup re-runs are
// cheap and a 190 GB file is not re-hashed every time.
func (f *Fetcher) Download(ctx context.Context, u, dest string, want Verify) error {
	if st, err := os.Stat(dest); err == nil && (want.Size == 0 || st.Size() == want.Size) {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}
	part := dest + ".part"
	var offset int64
	if st, err := os.Stat(part); err == nil {
		offset = st.Size()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return err
	}
	f.auth(req)
	if offset > 0 {
		req.Header.Set("Range", "bytes="+strconv.FormatInt(offset, 10)+"-")
	}
	resp, err := f.Client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	flags := os.O_CREATE | os.O_WRONLY
	switch resp.StatusCode {
	case http.StatusPartialContent:
		flags |= os.O_APPEND
	case http.StatusOK:
		// The server ignored the range: start over.
		flags |= os.O_TRUNC
		offset = 0
	case http.StatusRequestedRangeNotSatisfiable:
		// Nothing past what is there: the part is complete, if it checks.
		if want.Size == 0 || offset != want.Size {
			return fmt.Errorf("%s: the partial file (%d bytes) does not match the server's size", dest, offset)
		}
		return f.finish(part, dest, want)
	default:
		return fmt.Errorf("GET %s: %s", u, resp.Status)
	}
	total := offset + resp.ContentLength
	if resp.ContentLength < 0 {
		total = want.Size
	}
	out, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return err
	}
	done := offset
	buf := make([]byte, 1<<20)
	last := time.Now()
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := out.Write(buf[:n]); werr != nil {
				out.Close()
				return werr
			}
			done += int64(n)
			if f.Progress != nil && (time.Since(last) > time.Second || done == total) {
				f.Progress(filepath.Base(dest), done, total)
				last = time.Now()
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			out.Close()
			return fmt.Errorf("%s: %w (re-run to resume)", dest, rerr)
		}
	}
	if err := out.Close(); err != nil {
		return err
	}
	return f.finish(part, dest, want)
}

// finish checks the part and gives it its name. A mismatch removes the part:
// resuming a corrupt file cannot mend it.
func (f *Fetcher) finish(part, dest string, want Verify) error {
	st, err := os.Stat(part)
	if err != nil {
		return err
	}
	if want.Size > 0 && st.Size() != want.Size {
		os.Remove(part)
		return fmt.Errorf("%s: got %d bytes, expected %d", dest, st.Size(), want.Size)
	}
	if want.SHA256 != "" {
		sum, err := fileSHA256(part)
		if err != nil {
			return err
		}
		if !strings.EqualFold(sum, want.SHA256) {
			os.Remove(part)
			return fmt.Errorf("%s: sha256 %s, expected %s", dest, sum, want.SHA256)
		}
	}
	return os.Rename(part, dest)
}

func fileSHA256(path string) (string, error) {
	r, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer r.Close()
	h := sha256.New()
	if _, err := io.Copy(h, r); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func (f *Fetcher) auth(req *http.Request) {
	if f.HFToken != "" && strings.HasPrefix(req.URL.String(), f.HFBase) {
		req.Header.Set("Authorization", "Bearer "+f.HFToken)
	}
}

// --- Hugging Face -------------------------------------------------------------------

// HFFile is one file of a repo as the Hub lists it. LFS files carry their
// sha256; small files do not, and are checked by size alone.
type HFFile struct {
	Path   string
	Size   int64
	SHA256 string
}

// HFList lists a repo's files with sizes and hashes
// (GET /api/models/<repo>?blobs=true).
func (f *Fetcher) HFList(ctx context.Context, repo string) ([]HFFile, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, f.HFBase+"/api/models/"+repo+"?blobs=true", nil)
	if err != nil {
		return nil, err
	}
	f.auth(req)
	resp, err := f.Client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
			return nil, fmt.Errorf("hf.co/%s: %s (a gated repo needs HF_TOKEN)", repo, resp.Status)
		}
		return nil, fmt.Errorf("hf.co/%s: %s", repo, resp.Status)
	}
	var body struct {
		Siblings []struct {
			Rfilename string `json:"rfilename"`
			Size      int64  `json:"size"`
			LFS       *struct {
				SHA256 string `json:"sha256"`
			} `json:"lfs"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("hf.co/%s: %w", repo, err)
	}
	var out []HFFile
	for _, s := range body.Siblings {
		hf := HFFile{Path: s.Rfilename, Size: s.Size}
		if s.LFS != nil {
			hf.SHA256 = s.LFS.SHA256
		}
		out = append(out, hf)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("hf.co/%s: no files listed", repo)
	}
	return out, nil
}

// HFURL is a file's download URL.
func (f *Fetcher) HFURL(repo, path string) string {
	return f.HFBase + "/" + repo + "/resolve/main/" + (&url.URL{Path: path}).EscapedPath()
}

// HFFetchFile downloads one file of a repo to dest, verified against the
// Hub's listing. It returns the file's size for the fit check that precedes
// the download when size is all the caller wants (dest "").
func (f *Fetcher) HFFetchFile(ctx context.Context, repo, path, dest string) (HFFile, error) {
	files, err := f.HFList(ctx, repo)
	if err != nil {
		return HFFile{}, err
	}
	for _, hf := range files {
		if hf.Path == path {
			if dest == "" {
				return hf, nil
			}
			return hf, f.Download(ctx, f.HFURL(repo, path), dest, Verify{Size: hf.Size, SHA256: hf.SHA256})
		}
	}
	return HFFile{}, fmt.Errorf("hf.co/%s has no file %s", repo, path)
}

// HFRepoSize is the size of every file in a repo.
func HFRepoSize(files []HFFile) int64 {
	var n int64
	for _, hf := range files {
		n += hf.Size
	}
	return n
}

// HFFetchRepo downloads a whole repo (an MLX model) into dir, each file
// verified. Files already complete are skipped, so a run that stopped
// resumes where it was.
func (f *Fetcher) HFFetchRepo(ctx context.Context, repo string, files []HFFile, dir string) error {
	for _, hf := range files {
		if strings.Contains(hf.Path, "..") {
			return fmt.Errorf("hf.co/%s: refusing path %q", repo, hf.Path)
		}
		dest := filepath.Join(dir, filepath.FromSlash(hf.Path))
		if err := f.Download(ctx, f.HFURL(repo, hf.Path), dest, Verify{Size: hf.Size, SHA256: hf.SHA256}); err != nil {
			return err
		}
	}
	return nil
}

// ErrNoSpace is returned by CheckSpace.
var ErrNoSpace = errors.New("not enough disk")

// CheckSpace refuses a download that would leave less than keepFree on the
// disk. Weights are big and a full disk takes the database down with it.
func CheckSpace(h Hardware, need int64, keepFreeGB int) error {
	keep := uint64(keepFreeGB) * GiB
	if uint64(need)+keep > h.DiskFree {
		return fmt.Errorf("%w: %s to download, %s free at %s, and %d GiB must stay free (keep_free_gb)",
			ErrNoSpace, HumanBytes(uint64(need)), HumanBytes(h.DiskFree), h.DiskPath, keepFreeGB)
	}
	return nil
}
