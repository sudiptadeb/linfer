package bench

import "testing"

// Median of odd and even counts; Summarise drops zeros (a stream that did
// not measure) and keeps the range.
func TestMedianAndSummarise(t *testing.T) {
	if m := Median([]float64{3, 1, 2}); m != 2 {
		t.Errorf("odd median %v", m)
	}
	if m := Median([]float64{4, 1, 3, 2}); m != 2.5 {
		t.Errorf("even median %v", m)
	}
	if m := Median(nil); m != 0 {
		t.Errorf("empty median %v", m)
	}
	s := Summarise([]float64{0, 30, 0, 34, 32})
	if s.N != 3 || s.Median != 32 || s.Min != 30 || s.Max != 34 {
		t.Errorf("got %+v", s)
	}
	if s := Summarise([]float64{0, 0}); s.N != 0 {
		t.Errorf("all zero: %+v", s)
	}
}

// Two medians are within noise when the difference sits inside either
// range or under 5%; a clear gap with tight ranges is not.
func TestWithinNoise(t *testing.T) {
	a := Stat{Median: 100, Min: 95, Max: 105, N: 3}
	if !WithinNoise(a, Stat{Median: 104, Min: 103, Max: 105, N: 3}) {
		t.Error("4% apart called a difference")
	}
	if !WithinNoise(a, Stat{Median: 92, Min: 90, Max: 94, N: 3}) {
		t.Error("inside a's 10-wide range called a difference")
	}
	if WithinNoise(a, Stat{Median: 50, Min: 49, Max: 51, N: 3}) {
		t.Error("2x called noise")
	}
	if !WithinNoise(a, Stat{}) {
		t.Error("a missing measurement must not be called a difference")
	}
	// Single runs: no range, so only the 5% rule applies.
	if WithinNoise(Stat{Median: 100, Min: 100, Max: 100, N: 1}, Stat{Median: 90, Min: 90, Max: 90, N: 1}) {
		t.Error("10% apart on single runs called noise")
	}
}

// ParseList reads counts and k-suffixed token sizes, and refuses junk.
func TestParseList(t *testing.T) {
	got, err := ParseList("1, 2,4,8")
	if err != nil || len(got) != 4 || got[3] != 8 {
		t.Errorf("%v %v", got, err)
	}
	got, err = ParseList("2k,32K,512")
	if err != nil || got[0] != 2048 || got[1] != 32768 || got[2] != 512 {
		t.Errorf("%v %v", got, err)
	}
	if _, err := ParseList("2k,abc"); err == nil {
		t.Error("abc accepted")
	}
	if _, err := ParseList("0"); err == nil {
		t.Error("0 accepted")
	}
	if got, err := ParseList(""); err != nil || got != nil {
		t.Errorf("empty: %v %v", got, err)
	}
}
