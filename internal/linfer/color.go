package linfer

import (
	"io"
	"os"
)

// painter colours doctor's and setup's report: only when the writer is a
// terminal and NO_COLOR is unset, so pipes and logs stay plain text.
type painter bool

func painterFor(w io.Writer) painter {
	if os.Getenv("NO_COLOR") != "" {
		return false
	}
	f, ok := w.(*os.File)
	if !ok {
		return false
	}
	st, err := f.Stat()
	return painter(err == nil && st.Mode()&os.ModeCharDevice != 0)
}

func (p painter) wrap(code, s string) string {
	if !p {
		return s
	}
	return "\x1b[" + code + "m" + s + "\x1b[0m"
}

// label is a line's key, padded to the report's column.
func (p painter) label(s string) string { return p.wrap("36", padRight(s, 8)) }
func (p painter) bold(s string) string  { return p.wrap("1", s) }
func (p painter) dim(s string) string   { return p.wrap("2", s) }
func (p painter) good(s string) string  { return p.wrap("32", s) }
func (p painter) bad(s string) string   { return p.wrap("1;31", s) }

func padRight(s string, n int) string {
	for len(s) < n {
		s += " "
	}
	return s
}
