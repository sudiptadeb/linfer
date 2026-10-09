package bench

import (
	"context"
	"fmt"
	"strings"
)

// The needle: a fact with no pattern a model could guess.
const needle = "The vault passphrase is amber-falcon-7291."
const needleAnswer = "amber-falcon-7291"

// ContextRow is one retrieval: a haystack of about Tokens tokens with the
// needle at Depth percent of the way in.
type ContextRow struct {
	Context string  `json:"context"`
	Tokens  int     `json:"tokens"` // as the backend counted the prompt
	Depth   int     `json:"depth_pct"`
	Found   bool    `json:"found"`
	Answer  string  `json:"answer"`
	Err     string  `json:"err,omitempty"`
	Seconds float64 `json:"seconds"`
}

// runContext runs needle-in-haystack at each context size and depth. The
// check is exact: the passphrase must appear in the answer.
func runContext(ctx context.Context, cl *Client, contexts []int, depths []int, charsPerToken float64, maxContext int, progress func(string)) []ContextRow {
	var rows []ContextRow
	for _, tokens := range contexts {
		if tokens+600 > maxContext {
			rows = append(rows, ContextRow{Context: tokensLabel(tokens), Err: fmt.Sprintf("skipped: %d tokens do not fit the %d-token slot", tokens, maxContext)})
			continue
		}
		for _, depth := range depths {
			if ctx.Err() != nil {
				break
			}
			hay := prose(int(float64(tokens)*charsPerToken), fmt.Sprintf("needle-%d-%d", tokens, depth))
			at := len(hay) * depth / 100
			// Place it at a sentence boundary.
			if i := strings.Index(hay[at:], ". "); i >= 0 {
				at += i + 2
			}
			text := hay[:at] + needle + " " + hay[at:]
			prompt := "Here is a log.\n\n" + text + "\n\nWhat is the vault passphrase? Reply with the passphrase only."
			st := cl.Complete(ctx, []Message{{Role: "user", Content: prompt}}, 256)
			row := ContextRow{Context: tokensLabel(tokens), Tokens: st.PromptTokens, Depth: depth, Seconds: st.Total.Seconds()}
			if st.Err != "" {
				row.Err = st.Err
			} else {
				row.Answer = firstLine(st.Content)
				row.Found = strings.Contains(st.Content, needleAnswer)
			}
			mark := "FAIL"
			if row.Found {
				mark = "ok  "
			}
			progress(fmt.Sprintf("    %s %s at %d%%: %s", mark, row.Context, depth, ifs(row.Err != "", row.Err, row.Answer)))
			rows = append(rows, row)
		}
	}
	return rows
}
