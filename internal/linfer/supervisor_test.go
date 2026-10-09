package linfer

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// The test binary doubles as the fake backend: run with LINFER_FAKE_BACKEND
// set it serves /health on --port until SIGTERM, which is what the
// supervisor sends. LINFER_FAKE_CRASH names a file whose presence makes it
// exit 3 at once (consuming the file), to stage one crash.
func TestMain(m *testing.M) {
	if os.Getenv("LINFER_FAKE_BACKEND") == "1" {
		fakeBackend()
		return
	}
	os.Exit(m.Run())
}

func fakeBackend() {
	port := ""
	for i, a := range os.Args {
		if a == "--port" && i+1 < len(os.Args) {
			port = os.Args[i+1]
		}
	}
	if crash := os.Getenv("LINFER_FAKE_CRASH"); crash != "" {
		if _, err := os.Stat(crash); err == nil {
			os.Remove(crash)
			os.Exit(3)
		}
	}
	ln, err := net.Listen("tcp", "127.0.0.1:"+port)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM)
	defer stop()
	srv := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) })}
	go srv.Serve(ln)
	<-ctx.Done()
	srv.Close()
}

func freePort(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	_, port, _ := net.SplitHostPort(ln.Addr().String())
	return port
}

// fakeSupervisor runs the fake backend under a Supervisor with a short
// backoff. crash is the staging file for fakeBackend.
func fakeSupervisor(t *testing.T) (*Supervisor, string) {
	t.Helper()
	dir := t.TempDir()
	port := freePort(t)
	cfg, err := Parse([]byte("listen: 127.0.0.1:" + port + "\nmodel:\n  id: fake\n  gguf: /nonexistent.gguf\n  mlx: /nonexistent\ndir: " + dir + "\n"))
	if err != nil {
		t.Fatal(err)
	}
	crash := filepath.Join(dir, "crash")
	s := NewSupervisor(cfg, Hardware{OS: "test"}, slog.New(slog.NewTextHandler(os.Stderr, nil)))
	s.Backoff = 50 * time.Millisecond
	s.Prepare = func() (Plan, Command, error) {
		return Plan{Backend: BackendLlama, Slots: 1, Context: 1024}, Command{
			Path:   os.Args[0],
			Args:   []string{"--port", port},
			Env:    []string{"LINFER_FAKE_BACKEND=1", "LINFER_FAKE_CRASH=" + crash},
			Health: "http://127.0.0.1:" + port + "/health",
		}, nil
	}
	return s, crash
}

// waitFor polls until cond holds, or fails the test after a while.
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

// A backend that dies is restarted: the status shows a new pid, the
// restart count and the exit reason, and health comes back.
func TestSupervisorRestartsAfterCrash(t *testing.T) {
	s, _ := fakeSupervisor(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go s.Run(ctx)

	waitFor(t, "first start healthy", func() bool { return s.Status().Healthy })
	pid := s.Status().PID
	// Kill it from outside, as a crash would.
	if err := syscall.Kill(pid, syscall.SIGKILL); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "restart", func() bool {
		st := s.Status()
		return st.Healthy && st.PID != pid
	})
	st := s.Status()
	if st.Restarts != 1 || st.LastError == "" || st.Backend != BackendLlama || st.Plan == nil {
		t.Errorf("status %+v", st)
	}
}

// Pause stops the child and holds it stopped; resume brings it back. Neither
// counts as a crash.
func TestSupervisorPauseResume(t *testing.T) {
	s, _ := fakeSupervisor(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go s.Run(ctx)
	waitFor(t, "healthy", func() bool { return s.Status().Healthy })
	pid := s.Status().PID

	if err := s.Pause(); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "paused with no child", func() bool {
		st := s.Status()
		return st.Paused && st.PID == 0
	})
	if err := syscall.Kill(pid, 0); err == nil {
		t.Error("the old child is still alive")
	}
	time.Sleep(200 * time.Millisecond) // long past the backoff: it must stay down
	if st := s.Status(); st.PID != 0 || st.Restarts != 0 {
		t.Errorf("while paused: %+v", st)
	}
	if err := s.Pause(); err == nil {
		t.Error("second pause accepted")
	}
	if err := s.Resume(); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "resumed", func() bool { return s.Status().Healthy })
	if st := s.Status(); st.Restarts != 0 || st.Paused {
		t.Errorf("after resume: %+v", st)
	}
}

// A backend whose start fails keeps being retried (with the backoff) and
// the reason is in the status, so a service started before setup waits.
func TestSupervisorRetriesFailedPrepare(t *testing.T) {
	s, _ := fakeSupervisor(t)
	calls := 0
	s.Prepare = func() (Plan, Command, error) {
		calls++
		return Plan{}, Command{}, fmt.Errorf("no weights yet")
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go s.Run(ctx)
	waitFor(t, "retries", func() bool { return calls >= 3 })
	if st := s.Status(); st.LastError != "no weights yet" || st.PID != 0 {
		t.Errorf("status %+v", st)
	}
}

// The control socket: the client's status, pause, resume and switch reach
// the supervisor; a bad switch is refused with its reason.
func TestControlSocket(t *testing.T) {
	s, _ := fakeSupervisor(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go s.Run(ctx)
	socket := s.cfg.Paths().Socket
	go ServeControl(ctx, socket, s)
	cl := Client{Socket: socket}
	waitFor(t, "socket answers healthy", func() bool {
		st, err := cl.Status(ctx)
		return err == nil && st.Healthy
	})
	st, err := cl.Pause(ctx)
	if err != nil || !st.Paused {
		t.Fatalf("pause: %v %+v", err, st)
	}
	if _, err := cl.Pause(ctx); err == nil || err.Error() != "already paused" {
		t.Errorf("second pause: %v", err)
	}
	if st, err := cl.Resume(ctx); err != nil || st.Paused {
		t.Fatalf("resume: %v %+v", err, st)
	}
	if _, err := cl.Switch(ctx, "cuda"); err == nil {
		t.Error("switch cuda accepted")
	}
	if _, err := cl.Switch(ctx, BackendMLX); err != nil {
		t.Errorf("switch mlx: %v", err)
	}
	waitFor(t, "healthy after switch", func() bool {
		st, err := cl.Status(ctx)
		return err == nil && st.Healthy
	})
	// A second daemon on the same socket is refused, not a silent takeover.
	if err := ServeControl(ctx, socket, s); err == nil {
		t.Error("second daemon accepted")
	}
	// Nothing listening: the client says so.
	if _, err := (Client{Socket: filepath.Join(t.TempDir(), "none.sock")}).Status(ctx); err == nil {
		t.Error("no daemon, no error")
	}
}
