package bench

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strings"
)

// The tool suite: generic, public schemas and the cases an agent harness
// meets every day. Nothing here is specific to any model; the sampling is
// whatever the backend was launched with.

func tool(name, desc string, params map[string]any) Tool {
	var t Tool
	t.Type = "function"
	t.Function.Name = name
	t.Function.Description = desc
	t.Function.Parameters = params
	return t
}

func obj(required []string, props map[string]any) map[string]any {
	m := map[string]any{"type": "object", "properties": props}
	if len(required) > 0 {
		m["required"] = required
	}
	return m
}

func str(desc string) map[string]any { return map[string]any{"type": "string", "description": desc} }

// Tools is the suite's tool set, offered on every case so the model has to
// pick.
var Tools = []Tool{
	tool("get_weather", "Get the current weather for a location.", obj([]string{"location"}, map[string]any{
		"location": str("City name, e.g. Paris"),
		"unit":     map[string]any{"type": "string", "enum": []any{"celsius", "fahrenheit"}, "description": "Temperature unit"},
	})),
	tool("calculate", "Evaluate an arithmetic expression and return the result.", obj([]string{"expression"}, map[string]any{
		"expression": str("The expression, e.g. 2 * (3 + 4)"),
	})),
	tool("read_file", "Read a text file and return its contents.", obj([]string{"path"}, map[string]any{
		"path": str("Path of the file"),
	})),
	tool("write_file", "Write text to a file, replacing it.", obj([]string{"path", "content"}, map[string]any{
		"path":    str("Path of the file"),
		"content": str("The text to write"),
	})),
	tool("web_search", "Search the web.", obj([]string{"query"}, map[string]any{
		"query":       str("The search query"),
		"max_results": map[string]any{"type": "integer", "description": "How many results, 1 to 10"},
	})),
	tool("create_calendar_event", "Create a calendar event.", obj([]string{"title", "start", "end"}, map[string]any{
		"title":     str("Event title"),
		"start":     str("Start, ISO 8601, e.g. 2026-10-12T10:00"),
		"end":       str("End, ISO 8601"),
		"attendees": map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Email addresses"},
		"location": obj([]string{"name"}, map[string]any{
			"name":    str("Place name"),
			"address": str("Street address"),
		}),
	})),
	tool("lookup_account", "Look up a customer account by its id.", obj([]string{"account_id"}, map[string]any{
		"account_id": str("The 10-digit account id, as a string"),
	})),
}

// Case is one tool-calling case. Want is the tool names expected, in any
// order (empty: no call). Check inspects the parsed arguments of the
// expected calls for the case's own condition.
type Case struct {
	Name     string
	Messages []Message
	Want     []string
	Check    func(calls []ParsedCall) string // "" when fine, else why not
}

// ParsedCall is a tool call with its arguments decoded.
type ParsedCall struct {
	Name string
	Args map[string]any
}

var tenDigits = regexp.MustCompile(`^\d{10}$`)

// Cases is the suite.
var Cases = []Case{
	{
		Name:     "pick the right tool, enum arg",
		Messages: []Message{{Role: "user", Content: "What's the weather in Paris right now, in celsius?"}},
		Want:     []string{"get_weather"},
		Check: func(c []ParsedCall) string {
			loc, _ := c[0].Args["location"].(string)
			if !strings.Contains(strings.ToLower(loc), "paris") {
				return fmt.Sprintf("location %q is not Paris", loc)
			}
			if u, _ := c[0].Args["unit"].(string); u != "celsius" {
				return fmt.Sprintf("unit %q, want celsius", u)
			}
			return ""
		},
	},
	{
		Name:     "calculator",
		Messages: []Message{{Role: "user", Content: "Use the calculator to work out 1234 * 5678."}},
		Want:     []string{"calculate"},
		Check: func(c []ParsedCall) string {
			e, _ := c[0].Args["expression"].(string)
			if !strings.Contains(e, "1234") || !strings.Contains(e, "5678") {
				return fmt.Sprintf("expression %q lost an operand", e)
			}
			return ""
		},
	},
	{
		Name:     "no call when none is needed",
		Messages: []Message{{Role: "user", Content: "Write a two-line poem about the sea. Do not use any tools."}},
		Want:     nil,
	},
	{
		Name:     "two calls in one turn",
		Messages: []Message{{Role: "user", Content: "Check the current weather in Tokyo and in Berlin, both at once."}},
		Want:     []string{"get_weather", "get_weather"},
		Check: func(c []ParsedCall) string {
			seen := map[string]bool{}
			for _, call := range c {
				loc, _ := call.Args["location"].(string)
				seen[strings.ToLower(loc)] = true
			}
			if !has(seen, "tokyo") || !has(seen, "berlin") {
				return fmt.Sprintf("locations %v, want Tokyo and Berlin", keys(seen))
			}
			return ""
		},
	},
	{
		Name: "follow-up after a tool result",
		Messages: []Message{
			{Role: "user", Content: "What's the weather in Oslo?"},
			{Role: "assistant", ToolCalls: []ToolCall{call("call_1", "get_weather", `{"location":"Oslo","unit":"celsius"}`)}},
			{Role: "tool", ToolCallID: "call_1", Content: `{"location":"Oslo","temperature":7,"unit":"celsius","condition":"light rain"}`},
		},
		Want:  nil,
		Check: func(c []ParsedCall) string { return "" },
	},
	{
		Name: "argument copied exactly from context",
		Messages: []Message{
			{Role: "user", Content: "Our customer's account id is 4471029385. Keep it handy."},
			{Role: "assistant", Content: "Noted: account id 4471029385."},
			{Role: "user", Content: "Now look up that account."},
		},
		Want: []string{"lookup_account"},
		Check: func(c []ParsedCall) string {
			id, ok := c[0].Args["account_id"].(string)
			if !ok {
				return fmt.Sprintf("account_id is %T, want a string", c[0].Args["account_id"])
			}
			if id != "4471029385" || !tenDigits.MatchString(id) {
				return fmt.Sprintf("account_id %q, want 4471029385", id)
			}
			return ""
		},
	},
	{
		Name:     "nested object and array args",
		Messages: []Message{{Role: "user", Content: "Create a calendar event titled 'Design review' from 2026-10-12T10:00 to 2026-10-12T11:00 at the Hub, 12 Main Street, with alice@example.com and bob@example.com."}},
		Want:     []string{"create_calendar_event"},
		Check: func(c []ParsedCall) string {
			loc, ok := c[0].Args["location"].(map[string]any)
			if !ok {
				return "location is not an object"
			}
			if n, _ := loc["name"].(string); !strings.Contains(strings.ToLower(n), "hub") {
				return fmt.Sprintf("location.name %q, want the Hub", n)
			}
			att, _ := c[0].Args["attendees"].([]any)
			if len(att) != 2 {
				return fmt.Sprintf("%d attendees, want 2", len(att))
			}
			return ""
		},
	},
	{
		Name:     "integer arg",
		Messages: []Message{{Role: "user", Content: "Search the web for the latest Go release notes; I want 3 results."}},
		Want:     []string{"web_search"},
		Check: func(c []ParsedCall) string {
			n, ok := c[0].Args["max_results"].(float64)
			if !ok || n != 3 {
				return fmt.Sprintf("max_results %v, want 3", c[0].Args["max_results"])
			}
			return ""
		},
	},
	{
		Name:     "two required args",
		Messages: []Message{{Role: "user", Content: "Save the text 'hello world' to the file notes.txt."}},
		Want:     []string{"write_file"},
		Check: func(c []ParsedCall) string {
			if p, _ := c[0].Args["path"].(string); !strings.Contains(p, "notes.txt") {
				return fmt.Sprintf("path %q, want notes.txt", p)
			}
			if s, _ := c[0].Args["content"].(string); !strings.Contains(s, "hello world") {
				return fmt.Sprintf("content %q, want hello world", s)
			}
			return ""
		},
	},
}

func call(id, name, args string) ToolCall {
	var c ToolCall
	c.ID, c.Type = id, "function"
	c.Function.Name, c.Function.Arguments = name, args
	return c
}

func has(m map[string]bool, want string) bool {
	for k := range m {
		if strings.Contains(k, want) {
			return true
		}
	}
	return false
}

func keys(m map[string]bool) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	return out
}

// --- scoring ----------------------------------------------------------------------

// Outcome is one case run, scored.
type Outcome struct {
	Case      string `json:"case"`
	Run       int    `json:"run"`
	Pass      bool   `json:"pass"`
	HTTPError bool   `json:"http_error,omitempty"`
	// What went wrong, one line; empty on a pass.
	Why     string `json:"why,omitempty"`
	Content string `json:"content,omitempty"`
}

// ToolsScore is the suite's tally for one backend.
type ToolsScore struct {
	Cases         int       `json:"cases"`
	Passed        int       `json:"passed"`
	ValidCalls    int       `json:"valid_calls"`  // responses whose calls all parsed (of those expecting a call)
	RightTool     int       `json:"right_tool"`   // and named the expected tools
	SchemaValid   int       `json:"schema_valid"` // and whose arguments fit the schema
	NoCallCorrect int       `json:"no_call_correct"`
	NoCallCases   int       `json:"no_call_cases"`
	CallCases     int       `json:"call_cases"`
	Leaks         int       `json:"leaks"` // markup or reasoning in content
	HTTPErrors    int       `json:"http_errors"`
	Outcomes      []Outcome `json:"outcomes"`
	Failures      []string  `json:"failures,omitempty"`
}

// Accuracy is passed over cases, as a percentage.
func (s ToolsScore) Accuracy() float64 {
	if s.Cases == 0 {
		return 0
	}
	return 100 * float64(s.Passed) / float64(s.Cases)
}

var leakRE = regexp.MustCompile(`<think>|</think>|<tool_call>|<\|im_|<\|channel\|>|\{"name"\s*:\s*"[a-z_]+"\s*,\s*"arguments"`)

// Score judges one reply against its case. Every failure has a sentence.
func Score(c Case, r Reply, run int) Outcome {
	o := Outcome{Case: c.Name, Run: run, Content: firstLine(r.Content)}
	if r.Err != "" {
		o.HTTPError = r.Status != 0 && r.Status/100 != 2 || r.Status == 0
		o.Why = r.Err
		return o
	}
	if leakRE.MatchString(r.Content) {
		o.Why = "reasoning or tool markup leaked into content: " + firstLine(r.Content)
		return o
	}
	if len(c.Want) == 0 {
		if len(r.ToolCalls) > 0 {
			o.Why = fmt.Sprintf("called %s when no tool was needed", r.ToolCalls[0].Function.Name)
			return o
		}
		if strings.TrimSpace(r.Content) == "" {
			o.Why = "empty content with no tool call"
			return o
		}
		o.Pass = true
		return o
	}
	if len(r.ToolCalls) == 0 {
		o.Why = "no tool call; content: " + firstLine(r.Content)
		return o
	}
	var parsed []ParsedCall
	for _, tc := range r.ToolCalls {
		var args map[string]any
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
			o.Why = fmt.Sprintf("%s: arguments are not valid JSON: %s", tc.Function.Name, firstLine(tc.Function.Arguments))
			return o
		}
		parsed = append(parsed, ParsedCall{Name: tc.Function.Name, Args: args})
	}
	got := map[string]int{}
	for _, p := range parsed {
		got[p.Name]++
	}
	want := map[string]int{}
	for _, w := range c.Want {
		want[w]++
	}
	for name, n := range want {
		if got[name] != n {
			o.Why = fmt.Sprintf("called %v, want %v", names(parsed), c.Want)
			return o
		}
	}
	if len(parsed) != len(c.Want) {
		o.Why = fmt.Sprintf("called %v, want %v", names(parsed), c.Want)
		return o
	}
	for _, p := range parsed {
		if why := checkSchema(schemaOf(p.Name), p.Args, ""); why != "" {
			o.Why = p.Name + ": " + why
			return o
		}
	}
	if c.Check != nil {
		if why := c.Check(parsed); why != "" {
			o.Why = why
			return o
		}
	}
	o.Pass = true
	return o
}

func names(ps []ParsedCall) []string {
	var out []string
	for _, p := range ps {
		out = append(out, p.Name)
	}
	return out
}

func schemaOf(name string) map[string]any {
	for _, t := range Tools {
		if t.Function.Name == name {
			return t.Function.Parameters
		}
	}
	return nil
}

// checkSchema is the slice of JSON Schema the suite's tools use: type,
// required, properties, enum, items. It returns why a value does not fit.
func checkSchema(schema map[string]any, v any, path string) string {
	if schema == nil {
		return "unknown tool"
	}
	at := func(s string) string {
		if path == "" {
			return s
		}
		return path + ": " + s
	}
	typ, _ := schema["type"].(string)
	switch typ {
	case "object":
		m, ok := v.(map[string]any)
		if !ok {
			return at(fmt.Sprintf("is %s, want an object", jsonType(v)))
		}
		if req, ok := schema["required"].([]string); ok {
			for _, r := range req {
				if _, ok := m[r]; !ok {
					return at("missing required argument " + r)
				}
			}
		} else if req, ok := schema["required"].([]any); ok {
			for _, r := range req {
				if _, ok := m[r.(string)]; !ok {
					return at("missing required argument " + r.(string))
				}
			}
		}
		props, _ := schema["properties"].(map[string]any)
		for k, val := range m {
			ps, ok := props[k].(map[string]any)
			if !ok {
				return at("unknown argument " + k)
			}
			if why := checkSchema(ps, val, joinPath(path, k)); why != "" {
				return why
			}
		}
	case "array":
		arr, ok := v.([]any)
		if !ok {
			return at(fmt.Sprintf("is %s, want an array", jsonType(v)))
		}
		if items, ok := schema["items"].(map[string]any); ok {
			for i, x := range arr {
				if why := checkSchema(items, x, fmt.Sprintf("%s[%d]", path, i)); why != "" {
					return why
				}
			}
		}
	case "string":
		s, ok := v.(string)
		if !ok {
			return at(fmt.Sprintf("is %s, want a string", jsonType(v)))
		}
		if enum, ok := schema["enum"].([]any); ok {
			for _, e := range enum {
				if e == s {
					return ""
				}
			}
			return at(fmt.Sprintf("%q is not one of %v", s, enum))
		}
	case "integer":
		f, ok := v.(float64)
		if !ok || f != math.Trunc(f) {
			return at(fmt.Sprintf("is %s, want an integer", jsonType(v)))
		}
	case "number":
		if _, ok := v.(float64); !ok {
			return at(fmt.Sprintf("is %s, want a number", jsonType(v)))
		}
	case "boolean":
		if _, ok := v.(bool); !ok {
			return at(fmt.Sprintf("is %s, want a boolean", jsonType(v)))
		}
	}
	return ""
}

func joinPath(p, k string) string {
	if p == "" {
		return k
	}
	return p + "." + k
}

func jsonType(v any) string {
	switch v.(type) {
	case nil:
		return "null"
	case string:
		return "a string"
	case float64:
		return "a number"
	case bool:
		return "a boolean"
	case []any:
		return "an array"
	case map[string]any:
		return "an object"
	}
	return fmt.Sprintf("%T", v)
}

// Tally folds outcomes into a score. Each outcome counts once; the
// component counters say where the failures were.
func Tally(cases []Case, outcomes []Outcome, replies []Reply) ToolsScore {
	s := ToolsScore{Outcomes: outcomes}
	byName := map[string]Case{}
	for _, c := range cases {
		byName[c.Name] = c
	}
	for i, o := range outcomes {
		c := byName[o.Case]
		s.Cases++
		if o.Pass {
			s.Passed++
		} else {
			s.Failures = append(s.Failures, fmt.Sprintf("%s (run %d): %s", o.Case, o.Run, o.Why))
		}
		if o.HTTPError {
			s.HTTPErrors++
		}
		if strings.HasPrefix(o.Why, "reasoning or tool markup leaked") {
			s.Leaks++
		}
		if len(c.Want) == 0 {
			s.NoCallCases++
			if o.Pass {
				s.NoCallCorrect++
			}
			continue
		}
		s.CallCases++
		if i >= len(replies) {
			continue
		}
		r := replies[i]
		if r.Err != "" || len(r.ToolCalls) == 0 {
			continue
		}
		valid, right, schema := true, true, true
		var parsed []ParsedCall
		for _, tc := range r.ToolCalls {
			var args map[string]any
			if json.Unmarshal([]byte(tc.Function.Arguments), &args) != nil {
				valid = false
				break
			}
			parsed = append(parsed, ParsedCall{tc.Function.Name, args})
		}
		if !valid {
			continue
		}
		s.ValidCalls++
		want := map[string]int{}
		for _, w := range c.Want {
			want[w]++
		}
		got := map[string]int{}
		for _, p := range parsed {
			got[p.Name]++
		}
		for k, n := range want {
			if got[k] != n {
				right = false
			}
		}
		if !right || len(parsed) != len(c.Want) {
			continue
		}
		s.RightTool++
		for _, p := range parsed {
			if checkSchema(schemaOf(p.Name), p.Args, "") != "" {
				schema = false
			}
		}
		if schema {
			s.SchemaValid++
		}
	}
	return s
}

// runTools runs the suite `runs` times against the backend.
func runTools(ctx context.Context, cl *Client, runs int, progress func(string)) ToolsScore {
	var outcomes []Outcome
	var replies []Reply
	for run := 1; run <= runs; run++ {
		for _, c := range Cases {
			if ctx.Err() != nil {
				break
			}
			r := cl.Chat(ctx, c.Messages, Tools, 1024)
			o := Score(c, r, run)
			outcomes = append(outcomes, o)
			replies = append(replies, r)
			mark := "ok  "
			if !o.Pass {
				mark = "FAIL"
			}
			progress(fmt.Sprintf("    %s %s%s", mark, c.Name, ifs(!o.Pass, ": "+o.Why, "")))
		}
	}
	return Tally(Cases, outcomes, replies)
}

func ifs(cond bool, a, b string) string {
	if cond {
		return a
	}
	return b
}
