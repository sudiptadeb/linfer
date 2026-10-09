package linfer

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// The control channel is HTTP over a unix socket at Paths.Socket. A socket
// rather than a loopback port: it is not on the inference path, so nothing
// needs to reach it over TCP; the commands find it from the config with no
// port to pick or collide with; and the filesystem decides who may use it
// (the directory is the user's), where a port would let any local process
// pause the production model. HTTP rather than a bespoke framing so
// `curl --unix-socket` works when the binary does not.
//
// Routes: GET /status, POST /pause, POST /resume, POST /switch {"backend":…}.

// Status is what `linfer status` prints.
type Status struct {
	Backend  string    `json:"backend"`
	URL      string    `json:"url"`
	Model    string    `json:"model"`
	PID      int       `json:"pid"`
	Healthy  bool      `json:"healthy"`
	Paused   bool      `json:"paused"`
	RSSBytes uint64    `json:"rss_bytes"`
	Restarts int       `json:"restarts"`
	Since    time.Time `json:"since"`
	// LastError is why the backend last exited, when it did not exit clean.
	LastError string `json:"last_error,omitempty"`
	Plan      *Plan  `json:"plan,omitempty"`
}

// Controller is what the daemon exposes; the Supervisor implements it.
type Controller interface {
	Status() Status
	Pause() error
	Resume() error
	Switch(backend string) error
}

// ServeControl answers on the socket until ctx is done. A socket file left by
// a daemon that died is removed; one that answers means another daemon is
// running, which is an error rather than a silent takeover.
func ServeControl(ctx context.Context, socket string, c Controller) error {
	if _, err := os.Stat(socket); err == nil {
		if _, err := (Client{Socket: socket}).Status(ctx); err == nil {
			return fmt.Errorf("another linfer answers on %s", socket)
		}
		os.Remove(socket)
	}
	if err := os.MkdirAll(filepath.Dir(socket), 0o700); err != nil {
		return err
	}
	ln, err := net.Listen("unix", socket)
	if err != nil {
		return err
	}
	os.Chmod(socket, 0o600)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /status", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, c.Status())
	})
	mux.HandleFunc("POST /pause", func(w http.ResponseWriter, r *http.Request) {
		reply(w, c.Pause(), c)
	})
	mux.HandleFunc("POST /resume", func(w http.ResponseWriter, r *http.Request) {
		reply(w, c.Resume(), c)
	})
	mux.HandleFunc("POST /switch", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Backend string `json:"backend"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		reply(w, c.Switch(body.Backend), c)
	})
	srv := &http.Server{Handler: mux}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		srv.Shutdown(shutdown)
		os.Remove(socket)
	}()
	if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

func reply(w http.ResponseWriter, err error, c Controller) {
	if err != nil {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}
	writeJSON(w, c.Status())
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

// Client talks to a daemon over its socket.
type Client struct {
	Socket string
}

func (c Client) http() *http.Client {
	return &http.Client{
		Timeout: 30 * time.Second,
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
				return (&net.Dialer{}).DialContext(ctx, "unix", c.Socket)
			},
		},
	}
}

func (c Client) call(ctx context.Context, method, path string, body any) (Status, error) {
	var buf bytes.Buffer
	if body != nil {
		json.NewEncoder(&buf).Encode(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, "http://linfer"+path, &buf)
	if err != nil {
		return Status{}, err
	}
	resp, err := c.http().Do(req)
	if err != nil {
		return Status{}, fmt.Errorf("no linfer daemon at %s (%v); start one with `linfer serve`", c.Socket, errors.Unwrap(err))
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(resp.Body)
		return Status{}, fmt.Errorf("%s", bytes.TrimSpace(msg))
	}
	var st Status
	if err := json.NewDecoder(resp.Body).Decode(&st); err != nil {
		return Status{}, err
	}
	return st, nil
}

func (c Client) Status(ctx context.Context) (Status, error) {
	return c.call(ctx, "GET", "/status", nil)
}
func (c Client) Pause(ctx context.Context) (Status, error) { return c.call(ctx, "POST", "/pause", nil) }
func (c Client) Resume(ctx context.Context) (Status, error) {
	return c.call(ctx, "POST", "/resume", nil)
}
func (c Client) Switch(ctx context.Context, backend string) (Status, error) {
	return c.call(ctx, "POST", "/switch", map[string]string{"backend": backend})
}
