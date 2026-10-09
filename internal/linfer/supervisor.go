package linfer

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
	"syscall"
	"time"
)

// Supervisor runs the backend as a child: it prepares the plan, launches,
// health-checks, logs memory, restarts after a crash with a growing delay,
// and stops the child on pause, switch or shutdown. One backend at a time.
type Supervisor struct {
	cfg Config
	hw  Hardware
	log *slog.Logger
	// Prepare chooses the backend and builds the command. It is a field so a
	// test can run a fake backend without weights on disk.
	Prepare func() (Plan, Command, error)
	// Backoff is the first restart delay; it doubles to thirty times itself.
	Backoff time.Duration

	mu       sync.Mutex
	backend  string // what Choose picked, for Status
	plan     Plan
	cmd      *exec.Cmd
	started  time.Time
	healthy  bool
	rss      uint64
	restarts int
	lastErr  string
	paused   bool
	stopping bool          // the child is being killed on purpose; its exit is not a crash
	wake     chan struct{} // nudges Run after pause, resume or switch
}

// NewSupervisor makes one for cfg on this machine.
func NewSupervisor(cfg Config, hw Hardware, log *slog.Logger) *Supervisor {
	s := &Supervisor{cfg: cfg, hw: hw, log: log, Backoff: time.Second, wake: make(chan struct{}, 1)}
	s.Prepare = s.prepare
	return s
}

// prepare is the production Prepare: choose, read the weights, size, and
// write oMLX's files when that is the choice.
func (s *Supervisor) prepare() (Plan, Command, error) {
	s.mu.Lock()
	cfg := s.cfg
	s.mu.Unlock()
	backend, reason := Choose(cfg, s.hw, FileExists)
	s.log.Info(reason)
	return PlanFor(cfg, s.hw, backend)
}

// Run supervises until ctx is done. It never returns while the context
// lives: a backend that cannot start is retried with the backoff, so a
// service started before `setup` has run simply waits for it.
func (s *Supervisor) Run(ctx context.Context) error {
	delay := s.Backoff
	for {
		s.mu.Lock()
		paused := s.paused
		s.mu.Unlock()
		if paused {
			select {
			case <-ctx.Done():
				return nil
			case <-s.wake:
			}
			continue
		}
		plan, cmd, err := s.Prepare()
		if err != nil {
			s.fail(err)
			if !s.sleep(ctx, delay) {
				return nil
			}
			delay = min(delay*2, 30*s.Backoff)
			continue
		}
		exited, err := s.start(plan, cmd)
		if err != nil {
			s.fail(err)
			if !s.sleep(ctx, delay) {
				return nil
			}
			delay = min(delay*2, 30*s.Backoff)
			continue
		}
		stopped := s.watch(ctx, cmd, exited)
		switch {
		case ctx.Err() != nil:
			return nil
		case stopped:
			// Pause or switch asked for it: go round at once.
			delay = s.Backoff
		default:
			// A crash. A child that stayed up a while earns a fresh backoff.
			if time.Since(s.started) > 60*s.Backoff {
				delay = s.Backoff
			}
			if !s.sleep(ctx, delay) {
				return nil
			}
			delay = min(delay*2, 30*s.Backoff)
		}
	}
}

// start launches the child with its output appended to the backend log.
func (s *Supervisor) start(plan Plan, cmd Command) (<-chan error, error) {
	logf, err := os.OpenFile(s.cfg.Paths().Log, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	c := exec.Command(cmd.Path, cmd.Args...)
	c.Stdout, c.Stderr = logf, logf
	c.Env = append(os.Environ(), cmd.Env...)
	if err := c.Start(); err != nil {
		logf.Close()
		return nil, fmt.Errorf("start %s: %w", cmd.Path, err)
	}
	s.mu.Lock()
	s.cmd, s.plan, s.backend = c, plan, plan.Backend
	s.started, s.healthy, s.rss, s.stopping = time.Now(), false, 0, false
	s.mu.Unlock()
	s.log.Info("backend started", "backend", plan.Backend, "pid", c.Process.Pid, "cmd", cmd.Path, "args", strings.Join(cmd.Args, " "))
	exited := make(chan error, 1)
	go func() {
		exited <- c.Wait()
		logf.Close()
	}()
	return exited, nil
}

// watch health-checks and logs memory until the child exits, or until a
// pause, switch or shutdown asks for it to go. It returns true when the exit
// was asked for.
func (s *Supervisor) watch(ctx context.Context, cmd Command, exited <-chan error) (stopped bool) {
	tick := time.NewTicker(2 * time.Second)
	defer tick.Stop()
	lastLog := time.Time{}
	client := &http.Client{Timeout: 2 * time.Second}
	for {
		select {
		case err := <-exited:
			s.mu.Lock()
			stopping := s.stopping
			s.cmd, s.healthy, s.rss = nil, false, 0
			if !stopping {
				s.restarts++
				s.lastErr = exitReason(err)
			}
			s.mu.Unlock()
			if !stopping {
				s.log.Error("backend exited", "reason", exitReason(err), "restarts", s.restarts)
			}
			return stopping
		case <-ctx.Done():
			s.kill()
			<-exited
			return true
		case <-s.wake:
			s.kill()
			<-exited
			s.mu.Lock()
			s.cmd, s.healthy, s.rss = nil, false, 0
			s.mu.Unlock()
			return true
		case <-tick.C:
			healthy := probe(client, cmd.Health)
			s.mu.Lock()
			pid := 0
			if s.cmd != nil {
				pid = s.cmd.Process.Pid
			}
			was := s.healthy
			s.healthy = healthy
			s.rss = rssOf(pid)
			rss := s.rss
			s.mu.Unlock()
			if healthy && !was {
				s.log.Info("backend healthy", "url", cmd.Health, "rss", HumanBytes(rss))
			}
			if time.Since(lastLog) > time.Minute {
				s.log.Info("backend", "healthy", healthy, "rss", HumanBytes(rss))
				lastLog = time.Now()
			}
		}
	}
}

// kill stops the child: SIGTERM, then SIGKILL after fifteen seconds. Both
// backends release their memory on SIGTERM; the kill is for one that hangs.
func (s *Supervisor) kill() {
	s.mu.Lock()
	c := s.cmd
	s.stopping = true
	s.mu.Unlock()
	if c == nil || c.Process == nil {
		return
	}
	c.Process.Signal(syscall.SIGTERM)
	done := make(chan struct{})
	go func() {
		for {
			if c.ProcessState != nil {
				close(done)
				return
			}
			if err := c.Process.Signal(syscall.Signal(0)); err != nil {
				close(done)
				return
			}
			time.Sleep(100 * time.Millisecond)
		}
	}()
	select {
	case <-done:
	case <-time.After(15 * time.Second):
		c.Process.Kill()
	}
}

func (s *Supervisor) fail(err error) {
	s.mu.Lock()
	s.lastErr = err.Error()
	s.mu.Unlock()
	s.log.Error("backend not started", "err", err)
}

func (s *Supervisor) sleep(ctx context.Context, d time.Duration) bool {
	select {
	case <-ctx.Done():
		return false
	case <-time.After(d):
		return true
	case <-s.wake:
		return true
	}
}

func (s *Supervisor) nudge() {
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

// --- Controller ---------------------------------------------------------------------

func (s *Supervisor) Status() Status {
	s.mu.Lock()
	defer s.mu.Unlock()
	st := Status{
		Backend:   s.backend,
		URL:       s.cfg.URL(),
		Model:     s.cfg.Model.ID,
		Healthy:   s.healthy,
		Paused:    s.paused,
		RSSBytes:  s.rss,
		Restarts:  s.restarts,
		Since:     s.started,
		LastError: s.lastErr,
	}
	if s.cmd != nil && s.cmd.Process != nil {
		st.PID = s.cmd.Process.Pid
		p := s.plan
		st.Plan = &p
	}
	if st.Backend == "" {
		st.Backend = s.cfg.Backend
	}
	return st
}

// Pause stops the backend and keeps it stopped, freeing its memory, until
// Resume. The daemon stays up and keeps answering status. It returns once
// the child is gone, so a script that pauses to run a benchmark can trust
// that the memory is free when the command comes back.
func (s *Supervisor) Pause() error {
	s.mu.Lock()
	if s.paused {
		s.mu.Unlock()
		return fmt.Errorf("already paused")
	}
	s.paused = true
	s.mu.Unlock()
	s.nudge()
	return s.await(30*time.Second, func() bool { return s.cmd == nil })
}

// Resume starts the backend again and returns once it has been launched
// (not once it is healthy: loading the weights takes as long as it takes,
// and status says when it is ready).
func (s *Supervisor) Resume() error {
	s.mu.Lock()
	if !s.paused {
		s.mu.Unlock()
		return fmt.Errorf("not paused")
	}
	s.paused = false
	s.mu.Unlock()
	s.nudge()
	return s.await(30*time.Second, func() bool { return s.cmd != nil || s.lastErr != "" })
}

// await polls cond under the lock until it holds or the time is up.
func (s *Supervisor) await(d time.Duration, cond func() bool) error {
	deadline := time.Now().Add(d)
	for {
		s.mu.Lock()
		ok := cond()
		s.mu.Unlock()
		if ok {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("the backend did not change state within %s; see status", d)
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// Switch changes the backend for this daemon's lifetime and restarts the
// child. The config file is not rewritten: a switch is an operator's
// experiment, and the file says what a restart goes back to.
func (s *Supervisor) Switch(backend string) error {
	switch backend {
	case BackendAuto, BackendLlama, BackendMLX:
	default:
		return fmt.Errorf("backend %q: want llama, mlx or auto", backend)
	}
	s.mu.Lock()
	if backend == BackendLlama && s.cfg.Model.GGUF == "" {
		s.mu.Unlock()
		return fmt.Errorf("the model has no gguf weights")
	}
	if backend == BackendMLX && s.cfg.Model.MLX == "" {
		s.mu.Unlock()
		return fmt.Errorf("the model has no mlx weights")
	}
	s.cfg.Backend = backend
	s.mu.Unlock()
	s.nudge()
	return nil
}

// --- helpers --------------------------------------------------------------------

func probe(c *http.Client, url string) bool {
	resp, err := c.Get(url)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// rssOf is the process's resident set, from ps (which both platforms have),
// in bytes. Zero when it cannot be read.
func rssOf(pid int) uint64 {
	if pid == 0 {
		return 0
	}
	out, err := exec.Command("ps", "-o", "rss=", "-p", fmt.Sprint(pid)).Output()
	if err != nil {
		return 0
	}
	return uint64(atoi64(string(out))) * 1024
}

func exitReason(err error) string {
	if err == nil {
		return "exit 0"
	}
	return err.Error()
}
