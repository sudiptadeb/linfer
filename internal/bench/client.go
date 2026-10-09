// Package bench measures the backends linfer can run for the configured
// model, one at a time, each launched with the profile serve would use, and
// compares them: decode and prefill speed from the client's side of a
// stream, a tool-calling suite, and needle-in-haystack retrieval.
package bench

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Client talks OpenAI chat completions to one backend.
type Client struct {
	Base  string // http://host:port/v1
	Model string
	HTTP  *http.Client
}

func newClient(base, model string) *Client {
	return &Client{Base: base, Model: model, HTTP: &http.Client{Timeout: 30 * time.Minute}}
}

// Message is one chat message; ToolCalls and ToolCallID carry the tool
// protocol.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ToolCall is a call the model made.
type ToolCall struct {
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// Tool is a tool definition as the API takes it.
type Tool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string         `json:"name"`
		Description string         `json:"description"`
		Parameters  map[string]any `json:"parameters"`
	} `json:"function"`
}

// Stream is what one streamed completion measured. Times are from the
// moment the request was written, so they are comparable across backends;
// Server holds whatever timing the backend itself reported (llama-server's
// `timings`, oMLX's extra `usage` fields), keyed as the backend named them.
type Stream struct {
	TTFT             time.Duration // first content or reasoning token
	Total            time.Duration
	PromptTokens     int
	CompletionTokens int
	// Chunks is how many deltas carried text, and FirstChunkChars how long
	// the first was. llama-server sends one token per delta; oMLX sends
	// reasoning in bursts of many tokens, so its first chunk is already a
	// good part of the output, and a decode rate that assumed one token
	// there would come out several times too high.
	Chunks          int
	FirstChunkChars int
	Content         string
	Server          map[string]float64
	Status          int
	Err             string
}

// DecodeTPS is the stream's decode rate after the first chunk: the tokens
// that came after it, over the time after it. The first chunk's tokens are
// estimated from its length at charsPerToken.
func (s Stream) DecodeTPS(charsPerToken float64) float64 {
	if s.CompletionTokens < 2 || s.Total <= s.TTFT {
		return 0
	}
	return float64(s.CompletionTokens-s.FirstChunkTokens(charsPerToken)) / (s.Total - s.TTFT).Seconds()
}

// FirstChunkTokens estimates how many tokens the first chunk carried: at
// least one, at most all but one.
func (s Stream) FirstChunkTokens(charsPerToken float64) int {
	first := 1
	if charsPerToken > 0 {
		first = max(1, int(float64(s.FirstChunkChars)/charsPerToken+0.5))
	}
	return max(1, min(first, s.CompletionTokens-1))
}

// TokensPerChunk is the stream's granularity: 1 is token-by-token.
func (s Stream) TokensPerChunk() float64 {
	if s.Chunks == 0 {
		return 0
	}
	return float64(s.CompletionTokens) / float64(s.Chunks)
}

// Server-reported rates, where the backend names them: llama-server's
// timings, oMLX's extra usage fields. Zero when the backend gave none.
func (s Stream) ServerDecodeTPS() float64 {
	return firstOf(s.Server, "timings.predicted_per_second", "usage.generation_tokens_per_second")
}

func (s Stream) ServerPrefillTPS() float64 {
	return firstOf(s.Server, "timings.prompt_per_second", "usage.prompt_tokens_per_second")
}

func firstOf(m map[string]float64, keys ...string) float64 {
	for _, k := range keys {
		if v, ok := m[k]; ok && v > 0 {
			return v
		}
	}
	return 0
}

// PrefillTPS is the prompt rate implied by the time to first token: the
// first decode step is in there too, which costs the prefill a few percent
// on a long prompt and nothing on a short one.
func (s Stream) PrefillTPS() float64 {
	if s.PromptTokens == 0 || s.TTFT <= 0 {
		return 0
	}
	return float64(s.PromptTokens) / s.TTFT.Seconds()
}

// Complete runs one streamed completion and measures it. A non-2xx is
// returned in the Stream, not as an error: a 400 is a result to score.
func (c *Client) Complete(ctx context.Context, msgs []Message, maxTokens int) Stream {
	body := map[string]any{
		"model": c.Model, "messages": msgs, "max_tokens": maxTokens, "stream": true,
		"stream_options": map[string]any{"include_usage": true},
	}
	raw, _ := json.Marshal(body)
	start := time.Now()
	req, _ := http.NewRequestWithContext(ctx, "POST", c.Base+"/chat/completions", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return Stream{Err: err.Error(), Total: time.Since(start)}
	}
	defer resp.Body.Close()
	st := Stream{Status: resp.StatusCode, Server: map[string]float64{}}
	if resp.StatusCode/100 != 2 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		st.Err = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, firstLine(string(msg)))
		st.Total = time.Since(start)
		return st
	}
	var content strings.Builder
	deltas := 0
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content          *string `json:"content"`
					ReasoningContent *string `json:"reasoning_content"`
					Reasoning        *string `json:"reasoning"`
				} `json:"delta"`
			} `json:"choices"`
			Usage   map[string]any `json:"usage"`
			Timings map[string]any `json:"timings"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		for _, ch := range chunk.Choices {
			d := ch.Delta
			text := ""
			if d.Content != nil {
				text = *d.Content
				content.WriteString(text)
			} else if d.ReasoningContent != nil {
				text = *d.ReasoningContent
			} else if d.Reasoning != nil {
				text = *d.Reasoning
			}
			if text != "" {
				if st.TTFT == 0 {
					st.TTFT = time.Since(start)
					st.FirstChunkChars = len(text)
				}
				deltas++
			}
		}
		if chunk.Usage != nil {
			st.PromptTokens = intOf(chunk.Usage["prompt_tokens"])
			st.CompletionTokens = intOf(chunk.Usage["completion_tokens"])
			for k, v := range chunk.Usage {
				if f, ok := v.(float64); ok && !strings.HasSuffix(k, "_tokens") {
					st.Server["usage."+k] = f
				}
			}
		}
		for k, v := range chunk.Timings {
			if f, ok := v.(float64); ok {
				st.Server["timings."+k] = f
			}
		}
	}
	st.Total = time.Since(start)
	if err := sc.Err(); err != nil {
		st.Err = err.Error()
	}
	st.Chunks = deltas
	if st.CompletionTokens == 0 {
		// No usage from this server: the deltas are the best count there is.
		st.CompletionTokens = deltas
	}
	st.Content = content.String()
	return st
}

// Reply is a non-streamed completion, for the tool suite, where the content
// and the calls matter and the time does not.
type Reply struct {
	Content   string
	Reasoning string
	ToolCalls []ToolCall
	Status    int
	Err       string
	Raw       string
}

// Chat runs one completion with tools and returns what came back.
func (c *Client) Chat(ctx context.Context, msgs []Message, tools []Tool, maxTokens int) Reply {
	body := map[string]any{"model": c.Model, "messages": msgs, "max_tokens": maxTokens}
	if len(tools) > 0 {
		body["tools"] = tools
	}
	raw, _ := json.Marshal(body)
	req, _ := http.NewRequestWithContext(ctx, "POST", c.Base+"/chat/completions", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return Reply{Err: err.Error()}
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<22))
	r := Reply{Status: resp.StatusCode, Raw: string(out)}
	if resp.StatusCode/100 != 2 {
		r.Err = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, firstLine(string(out)))
		return r
	}
	return parseReply(out, r)
}

// parseReply reads a chat completion body into a Reply.
func parseReply(out []byte, r Reply) Reply {
	var body struct {
		Choices []struct {
			Message struct {
				Content          *string    `json:"content"`
				ReasoningContent string     `json:"reasoning_content"`
				ToolCalls        []ToolCall `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(out, &body); err != nil {
		r.Err = "unparseable response: " + err.Error()
		return r
	}
	if len(body.Choices) == 0 {
		r.Err = "no choices in response"
		return r
	}
	m := body.Choices[0].Message
	if m.Content != nil {
		r.Content = *m.Content
	}
	r.Reasoning = m.ReasoningContent
	r.ToolCalls = m.ToolCalls
	return r
}

func intOf(v any) int {
	f, _ := v.(float64)
	return int(f)
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	if len(s) > 200 {
		s = s[:200] + "…"
	}
	return s
}
