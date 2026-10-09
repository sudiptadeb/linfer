package bench

import (
	"strings"
	"testing"
)

func caseNamed(t *testing.T, name string) Case {
	t.Helper()
	for _, c := range Cases {
		if c.Name == name {
			return c
		}
	}
	t.Fatalf("no case %q", name)
	return Case{}
}

func reply(content string, calls ...ToolCall) Reply {
	return Reply{Status: 200, Content: content, ToolCalls: calls}
}

// A clean call with the right tool and arguments passes; each kind of
// malformed reply fails with a sentence that says what was wrong.
func TestScoreToolCases(t *testing.T) {
	weather := caseNamed(t, "pick the right tool, enum arg")
	cases := []struct {
		name string
		c    Case
		r    Reply
		pass bool
		why  string
	}{
		{"good call", weather, reply("", call("1", "get_weather", `{"location":"Paris","unit":"celsius"}`)), true, ""},
		{"wrong tool", weather, reply("", call("1", "web_search", `{"query":"Paris weather"}`)), false, "called [web_search], want [get_weather]"},
		{"broken json", weather, reply("", call("1", "get_weather", `{"location":"Paris",`)), false, "not valid JSON"},
		{"bad enum", weather, reply("", call("1", "get_weather", `{"location":"Paris","unit":"kelvin"}`)), false, `"kelvin" is not one of`},
		{"missing required", weather, reply("", call("1", "get_weather", `{"unit":"celsius"}`)), false, "missing required argument location"},
		{"unknown argument", weather, reply("", call("1", "get_weather", `{"location":"Paris","city":"Paris"}`)), false, "unknown argument city"},
		{"no call", weather, reply("It is sunny in Paris."), false, "no tool call"},
		{"leaked markup", weather, reply(`<tool_call>{"name":"get_weather","arguments":{"location":"Paris"}}</tool_call>`), false, "leaked into content"},
		{"leaked reasoning", weather, reply("<think>I should call get_weather</think>", call("1", "get_weather", `{"location":"Paris","unit":"celsius"}`)), false, "leaked into content"},
		{"http error", weather, Reply{Status: 400, Err: "HTTP 400: prompt exceeds memory"}, false, "HTTP 400"},
		{"no-call case, answered", caseNamed(t, "no call when none is needed"), reply("Waves fold the light\nand carry it home."), true, ""},
		{"no-call case, called anyway", caseNamed(t, "no call when none is needed"), reply("", call("1", "web_search", `{"query":"sea poem"}`)), false, "called web_search when no tool was needed"},
		{"no-call case, empty", caseNamed(t, "no call when none is needed"), reply(""), false, "empty content"},
		{"two calls", caseNamed(t, "two calls in one turn"), reply("", call("1", "get_weather", `{"location":"Tokyo"}`), call("2", "get_weather", `{"location":"Berlin"}`)), true, ""},
		{"one of two", caseNamed(t, "two calls in one turn"), reply("", call("1", "get_weather", `{"location":"Tokyo"}`)), false, "want [get_weather get_weather]"},
		{"follow-up answered", caseNamed(t, "follow-up after a tool result"), reply("It is 7°C in Oslo with light rain."), true, ""},
		{"follow-up re-called", caseNamed(t, "follow-up after a tool result"), reply("", call("2", "get_weather", `{"location":"Oslo"}`)), false, "when no tool was needed"},
		{"exact id", caseNamed(t, "argument copied exactly from context"), reply("", call("1", "lookup_account", `{"account_id":"4471029385"}`)), true, ""},
		{"id as number", caseNamed(t, "argument copied exactly from context"), reply("", call("1", "lookup_account", `{"account_id":4471029385}`)), false, "want a string"},
		{"id mangled", caseNamed(t, "argument copied exactly from context"), reply("", call("1", "lookup_account", `{"account_id":"447102938"}`)), false, "want 4471029385"},
		{"nested ok", caseNamed(t, "nested object and array args"), reply("", call("1", "create_calendar_event",
			`{"title":"Design review","start":"2026-10-12T10:00","end":"2026-10-12T11:00","attendees":["alice@example.com","bob@example.com"],"location":{"name":"the Hub","address":"12 Main Street"}}`)), true, ""},
		{"nested as string", caseNamed(t, "nested object and array args"), reply("", call("1", "create_calendar_event",
			`{"title":"Design review","start":"2026-10-12T10:00","end":"2026-10-12T11:00","attendees":["alice@example.com","bob@example.com"],"location":"the Hub"}`)), false, "want an object"},
		{"integer as string", caseNamed(t, "integer arg"), reply("", call("1", "web_search", `{"query":"Go release notes","max_results":"3"}`)), false, "want an integer"},
		{"integer ok", caseNamed(t, "integer arg"), reply("", call("1", "web_search", `{"query":"Go release notes","max_results":3}`)), true, ""},
	}
	for _, c := range cases {
		o := Score(c.c, c.r, 1)
		if o.Pass != c.pass {
			t.Errorf("%s: pass %v, want %v (%s)", c.name, o.Pass, c.pass, o.Why)
			continue
		}
		if !c.pass && !strings.Contains(o.Why, c.why) {
			t.Errorf("%s: why %q, want it to mention %q", c.name, o.Why, c.why)
		}
	}
}

// The tally counts each component where it fails: valid JSON, right tool,
// schema, no-call, leaks, HTTP errors.
func TestTally(t *testing.T) {
	weather := caseNamed(t, "pick the right tool, enum arg")
	noCall := caseNamed(t, "no call when none is needed")
	replies := []Reply{
		reply("", call("1", "get_weather", `{"location":"Paris","unit":"celsius"}`)), // pass
		reply("", call("1", "get_weather", `{"location":"Paris","unit":"kelvin"}`)),  // right tool, bad schema
		reply("", call("1", "calculate", `{"expression":"1"}`)),                      // valid, wrong tool
		reply("", call("1", "get_weather", `{bad`)),                                  // invalid json
		Reply{Status: 400, Err: "HTTP 400: too long"},                                // http error
		reply("<think>hm</think>ok"),                                                 // leak, on the no-call case
		reply("Here is a poem."),                                                     // no-call pass
	}
	cases := []Case{weather, weather, weather, weather, weather, noCall, noCall}
	var outcomes []Outcome
	for i, r := range replies {
		outcomes = append(outcomes, Score(cases[i], r, 1))
	}
	s := Tally(Cases, outcomes, replies)
	if s.Cases != 7 || s.Passed != 2 || s.CallCases != 5 || s.NoCallCases != 2 || s.NoCallCorrect != 1 {
		t.Errorf("counts %+v", s)
	}
	if s.ValidCalls != 3 || s.RightTool != 2 || s.SchemaValid != 1 || s.HTTPErrors != 1 || s.Leaks != 1 {
		t.Errorf("components valid %d right %d schema %d http %d leaks %d", s.ValidCalls, s.RightTool, s.SchemaValid, s.HTTPErrors, s.Leaks)
	}
	if len(s.Failures) != 5 || s.Accuracy() < 28 || s.Accuracy() > 29 {
		t.Errorf("failures %d, accuracy %.1f", len(s.Failures), s.Accuracy())
	}
}

// Every case's expected tools exist in the suite's tool set, so a right
// answer can always be schema-checked.
func TestCasesNameRealTools(t *testing.T) {
	for _, c := range Cases {
		for _, w := range c.Want {
			if schemaOf(w) == nil {
				t.Errorf("%s wants unknown tool %s", c.Name, w)
			}
		}
	}
}

// parseReply reads what a server sends, including a null content with
// calls, and reports an unparseable body as an error rather than a pass.
func TestParseReply(t *testing.T) {
	r := parseReply([]byte(`{"choices":[{"message":{"role":"assistant","content":null,"reasoning_content":"thinking","tool_calls":[{"id":"c1","type":"function","function":{"name":"calculate","arguments":"{\"expression\":\"1+1\"}"}}]}}]}`), Reply{Status: 200})
	if r.Err != "" || r.Content != "" || r.Reasoning != "thinking" || len(r.ToolCalls) != 1 || r.ToolCalls[0].Function.Name != "calculate" {
		t.Errorf("got %+v", r)
	}
	if r := parseReply([]byte(`not json`), Reply{Status: 200}); r.Err == "" {
		t.Error("garbage parsed")
	}
	if r := parseReply([]byte(`{"choices":[]}`), Reply{Status: 200}); r.Err == "" {
		t.Error("no choices parsed")
	}
}
