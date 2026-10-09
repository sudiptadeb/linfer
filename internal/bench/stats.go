package bench

import (
	"math"
	"sort"
)

// Stat is a measurement repeated N times: its median and its range. The
// range is what "within noise" is judged by.
type Stat struct {
	Median float64 `json:"median"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	N      int     `json:"n"`
}

// Summarise is the Stat of the samples, ignoring NaN and zero (a zero is a
// measurement that did not happen, not a speed).
func Summarise(samples []float64) Stat {
	var xs []float64
	for _, x := range samples {
		if x > 0 && !math.IsNaN(x) && !math.IsInf(x, 0) {
			xs = append(xs, x)
		}
	}
	if len(xs) == 0 {
		return Stat{}
	}
	sort.Float64s(xs)
	return Stat{Median: Median(xs), Min: xs[0], Max: xs[len(xs)-1], N: len(xs)}
}

// Median of sorted or unsorted samples; the mean of the middle two when
// even.
func Median(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

// WithinNoise says whether two medians are too close to call: their
// difference is inside either measurement's own range, or under 5%. With a
// single run there is no range, so only the 5% applies, and the report says
// so.
func WithinNoise(a, b Stat) bool {
	if a.N == 0 || b.N == 0 {
		return true
	}
	diff := math.Abs(a.Median - b.Median)
	if diff <= 0.05*math.Max(a.Median, b.Median) {
		return true
	}
	return diff <= (a.Max-a.Min) || diff <= (b.Max-b.Min)
}
