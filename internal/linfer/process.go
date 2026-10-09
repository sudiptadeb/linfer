package linfer

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"time"
)

// PlanFor is the launch for one backend: read the weights, size, write
// oMLX's files when that is the backend, and build the command. The
// supervisor uses it after Choose; bench uses it with each backend in turn,
// so a benchmarked backend runs exactly what serve would run.
func PlanFor(cfg Config, hw Hardware, backend string) (Plan, Command, error) {
	m, err := ReadModel(cfg, backend)
	if err != nil {
		return Plan{}, Command{}, fmt.Errorf("%w (run `linfer setup` to fetch what is missing)", err)
	}
	plan, err := Size(hw, m, backend, cfg)
	if err != nil {
		return plan, Command{}, err
	}
	if backend == BackendMLX {
		if err := PrepareOMLX(cfg, plan); err != nil {
			return plan, Command{}, err
		}
	}
	return plan, CommandFor(cfg, plan), nil
}

// Process is one launched backend, outside the supervisor: bench starts
// each backend, waits for it, runs its suites and stops it, and wants the
// exit to be final before the next one loads.
type Process struct {
	Command Command
	cmd     *exec.Cmd
	done    chan error
	logf    *os.File
}

// Start launches cmd with its output appended to logPath.
func Start(cmd Command, logPath string) (*Process, error) {
	logf, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
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
	p := &Process{Command: cmd, cmd: c, done: make(chan error, 1), logf: logf}
	go func() {
		p.done <- c.Wait()
		logf.Close()
	}()
	return p, nil
}

func (p *Process) PID() int { return p.cmd.Process.Pid }

// RSS is the child's resident set now.
func (p *Process) RSS() uint64 { return rssOf(p.PID()) }

// WaitHealthy polls the health URL until it answers 200, the child exits,
// or the time is up.
func (p *Process) WaitHealthy(ctx context.Context, timeout time.Duration) error {
	client := &http.Client{Timeout: 2 * time.Second}
	deadline := time.Now().Add(timeout)
	for {
		if probe(client, p.Command.Health) {
			return nil
		}
		select {
		case err := <-p.done:
			p.done <- err
			return fmt.Errorf("backend exited while loading: %s (see its log)", exitReason(err))
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Second):
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("backend not healthy after %s", timeout)
		}
	}
}

// Exited reports whether the child has gone, with its reason.
func (p *Process) Exited() (bool, string) {
	select {
	case err := <-p.done:
		p.done <- err
		return true, exitReason(err)
	default:
		return false, ""
	}
}

// Stop ends the child: SIGTERM, SIGKILL after the grace period, and then
// waits for the exit, which is when its memory is free again.
func (p *Process) Stop(grace time.Duration) {
	p.cmd.Process.Signal(syscall.SIGTERM)
	select {
	case err := <-p.done:
		p.done <- err
	case <-time.After(grace):
		p.cmd.Process.Kill()
		err := <-p.done
		p.done <- err
	}
}

// FreeMemory is the memory a new backend could take now: on macOS the free,
// inactive, speculative and purgeable pages (file cache in active pages is
// not counted, so this is conservative); on Linux MemAvailable. Zero when
// it cannot be read.
func FreeMemory(p Probe) uint64 {
	switch p.GOOS {
	case "darwin":
		out, err := p.Run("vm_stat")
		if err != nil {
			return 0
		}
		return vmStatFree(out)
	case "linux":
		mem, err := p.ReadFile("/proc/meminfo")
		if err != nil {
			return 0
		}
		return uint64(meminfoKB(string(mem), "MemAvailable")) * 1024
	}
	return 0
}

// vmStatFree reads vm_stat's output: the page size from its header, then
// the page counts.
func vmStatFree(out string) uint64 {
	var pageSize uint64 = 4096
	var pages uint64
	for _, line := range strings.Split(out, "\n") {
		if strings.Contains(line, "page size of") {
			f := strings.Fields(line)
			for i, w := range f {
				if w == "of" && i+1 < len(f) {
					pageSize = uint64(atoi64(f[i+1]))
				}
			}
			continue
		}
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		switch strings.TrimSpace(k) {
		case "Pages free", "Pages inactive", "Pages speculative", "Pages purgeable":
			pages += uint64(atoi64(strings.TrimSuffix(strings.TrimSpace(v), ".")))
		}
	}
	return pages * pageSize
}
