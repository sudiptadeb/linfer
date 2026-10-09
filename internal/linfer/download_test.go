package linfer

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// blobServer serves one file with Range support, and cuts the first full
// download short after `cut` bytes so the client has to resume.
func blobServer(t *testing.T, blob []byte, cut int) (*httptest.Server, *int32) {
	t.Helper()
	var requests int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&requests, 1)
		if n == 1 && cut > 0 {
			// Promise the whole file, deliver part of it, and drop the connection.
			w.Header().Set("Content-Length", fmt.Sprint(len(blob)))
			w.WriteHeader(http.StatusOK)
			w.Write(blob[:cut])
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			panic(http.ErrAbortHandler)
		}
		http.ServeContent(w, r, "blob", time.Time{}, bytes.NewReader(blob))
	}))
	t.Cleanup(srv.Close)
	return srv, &requests
}

func sha(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

// A download that breaks off resumes from the partial file with a Range
// request, and the result is verified by size and sha256.
func TestDownloadResumes(t *testing.T) {
	blob := bytes.Repeat([]byte("0123456789abcdef"), 4096) // 64 KiB
	srv, requests := blobServer(t, blob, 10000)
	f := &Fetcher{Client: srv.Client()}
	dest := filepath.Join(t.TempDir(), "blob.bin")

	err := f.Download(context.Background(), srv.URL+"/blob", dest, Verify{Size: int64(len(blob)), SHA256: sha(blob)})
	if err == nil {
		t.Fatal("the cut download passed")
	}
	if st, err := os.Stat(dest + ".part"); err != nil || st.Size() != 10000 {
		t.Fatalf("partial file: %v %v", st, err)
	}
	if err := f.Download(context.Background(), srv.URL+"/blob", dest, Verify{Size: int64(len(blob)), SHA256: sha(blob)}); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(dest)
	if !bytes.Equal(got, blob) {
		t.Error("content differs after resume")
	}
	if _, err := os.Stat(dest + ".part"); !os.IsNotExist(err) {
		t.Error("part file left behind")
	}
	if *requests != 2 {
		t.Errorf("%d requests, want 2 (the cut one and the resume)", *requests)
	}
	// Already there at the right size: no request at all.
	if err := f.Download(context.Background(), srv.URL+"/blob", dest, Verify{Size: int64(len(blob))}); err != nil || *requests != 2 {
		t.Errorf("re-download: err %v, requests %d", err, *requests)
	}
}

// A hash mismatch is an error and removes the part, so a resume cannot
// mend a corrupt file.
func TestDownloadVerifies(t *testing.T) {
	blob := []byte("hello")
	srv, _ := blobServer(t, blob, 0)
	f := &Fetcher{Client: srv.Client()}
	dest := filepath.Join(t.TempDir(), "x")
	err := f.Download(context.Background(), srv.URL+"/x", dest, Verify{Size: 5, SHA256: strings.Repeat("0", 64)})
	if err == nil || !strings.Contains(err.Error(), "sha256") {
		t.Errorf("got %v", err)
	}
	for _, p := range []string{dest, dest + ".part"} {
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Errorf("%s left behind", p)
		}
	}
	if err := f.Download(context.Background(), srv.URL+"/x", dest, Verify{Size: 99}); err == nil || !strings.Contains(err.Error(), "expected 99") {
		t.Errorf("size check: %v", err)
	}
}

// Hugging Face: the Hub's listing gives sizes and LFS hashes, and a file is
// fetched from its resolve URL and verified against them.
func TestHFFetch(t *testing.T) {
	weights := bytes.Repeat([]byte("w"), 5000)
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/org/repo", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("blobs") != "true" {
			http.Error(w, "want blobs=true", 400)
			return
		}
		fmt.Fprintf(w, `{"siblings":[{"rfilename":"config.json","size":2},{"rfilename":"model.gguf","size":%d,"lfs":{"sha256":%q}}]}`, len(weights), sha(weights))
	})
	mux.HandleFunc("/org/repo/resolve/main/model.gguf", func(w http.ResponseWriter, r *http.Request) {
		http.ServeContent(w, r, "model.gguf", time.Time{}, bytes.NewReader(weights))
	})
	mux.HandleFunc("/org/repo/resolve/main/config.json", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("{}"))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	f := &Fetcher{Client: srv.Client(), HFBase: srv.URL}

	files, err := f.HFList(context.Background(), "org/repo")
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 2 || files[1].SHA256 != sha(weights) || HFRepoSize(files) != int64(len(weights))+2 {
		t.Errorf("listing %+v", files)
	}
	dest := filepath.Join(t.TempDir(), "model.gguf")
	if _, err := f.HFFetchFile(context.Background(), "org/repo", "model.gguf", dest); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(dest); !bytes.Equal(got, weights) {
		t.Error("content differs")
	}
	if _, err := f.HFFetchFile(context.Background(), "org/repo", "missing.gguf", ""); err == nil || !strings.Contains(err.Error(), "no file") {
		t.Errorf("missing file: %v", err)
	}
	dir := t.TempDir()
	if err := f.HFFetchRepo(context.Background(), "org/repo", files, dir); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		t.Error(err)
	}
}

// The disk check keeps the configured margin free.
func TestCheckSpace(t *testing.T) {
	h := Hardware{DiskFree: 300 * GiB, DiskPath: "/d"}
	if err := CheckSpace(h, 50*GiB, 200); err != nil {
		t.Error(err)
	}
	if err := CheckSpace(h, 150*GiB, 200); err == nil || !strings.Contains(err.Error(), "keep_free_gb") {
		t.Errorf("got %v", err)
	}
}
