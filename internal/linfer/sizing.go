package linfer

import (
	"fmt"
)

// Plan is what the backend is launched with. Size builds it from the
// machine and the model; explicit config always overrides what it computes.
type Plan struct {
	Backend string
	// llama-server: -np Slots, -c Slots*Context, --cache-ram CacheRAMMB.
	Slots      int
	Context    int
	CacheRAMMB int
	// oMLX: --max-concurrent-requests MaxConcurrent; max_context_window
	// ContextWindow is the longest prompt it accepts.
	MaxConcurrent int
	ContextWindow int
	// For the report.
	Weights, Budget uint64
	KVPerToken      int64
	Notes           []string
}

// The constants behind the auto sizing. They are set so that the reference
// box (256 GB M3 Ultra, Metal working set 222.7 GiB) gets the settings that
// were measured there: a 176 GiB q8 GGUF runs 8 slots of 131,072 tokens
// with a 16 GiB prompt cache, and a 182 GiB MLX model runs 6 concurrent
// requests.
const (
	// reserve is left over after weights and cache for the backend's own
	// buffers (compute graphs, the vision projector's activations) and for
	// the machine itself.
	reserve = 4 * GiB
	// contextCap is the most context a slot gets by default. An agent client
	// compacts its transcript long before this, and the KV past it is memory
	// taken from other slots.
	contextCap = 131072
	// contextKnee and contextFloor: auto trades context for slots down to
	// the knee, then gives up slots, and only then goes below the knee.
	contextKnee  = 32768
	contextFloor = 8192
	maxSlots     = 8
	// cacheRAMCapMB is llama-server's RAM prompt cache at most: a slot
	// whose place was taken comes back from here instead of reprocessing its
	// prompt. 16 GiB held every live session on the reference box.
	cacheRAMCapMB = 16384
	// maxConcurrent caps oMLX: 8 concurrent ~65k-token requests ran it out
	// of Metal memory on the reference box; 6 did not.
	maxConcurrent = 6
	// fallbackKV is used when a model's shape cannot be read.
	fallbackKV = 64 * 1024
)

// Size computes the launch plan for backend from the machine and the model.
// It fails when the weights do not fit: the client should hear that from
// setup, not from a backend dying in a restart loop.
func Size(h Hardware, m ModelInfo, backend string, cfg Config) (Plan, error) {
	p := Plan{Backend: backend, Weights: uint64(m.WeightBytes), Budget: h.Budget(), KVPerToken: m.KVBytesPerToken()}
	if p.KVPerToken == 0 {
		p.KVPerToken = fallbackKV
		p.note("model shape unknown; assuming %d KiB of KV cache per token", fallbackKV/1024)
	}
	if p.Weights+reserve > p.Budget {
		return p, fmt.Errorf("%s of weights do not fit: %s available (%s) less %s reserved",
			HumanBytes(p.Weights), HumanBytes(p.Budget), h.GPUMemSource, HumanBytes(reserve))
	}
	headroom := p.Budget - p.Weights - reserve
	p.note("weights %s of %s (%s); %s left after a %s reserve",
		HumanBytes(p.Weights), HumanBytes(p.Budget), h.GPUMemSource, HumanBytes(headroom), HumanBytes(reserve))

	switch backend {
	case BackendLlama:
		return p.sizeLlama(h, m, cfg, headroom)
	case BackendMLX:
		return p.sizeMLX(m, cfg, headroom)
	}
	return p, fmt.Errorf("unknown backend %q", backend)
}

func (p *Plan) sizeLlama(h Hardware, m ModelInfo, cfg Config, headroom uint64) (Plan, error) {
	// The prompt cache is RAM. On unified memory it comes out of the same
	// headroom as the KV cache; on a discrete GPU it is system RAM, separate
	// from the VRAM the KV cache lives in.
	kvBudget := headroom
	switch {
	case cfg.CacheRAMMB > 0:
		p.CacheRAMMB = cfg.CacheRAMMB
		p.note("cache-ram %d MB from config", p.CacheRAMMB)
	case h.GPU == GPUCUDA || h.GPU == GPUROCm:
		p.CacheRAMMB = min(cacheRAMCapMB, int(h.RAM/4/MiB))
		p.note("cache-ram %d MB: a quarter of %s system RAM, at most %d", p.CacheRAMMB, HumanBytes(h.RAM), cacheRAMCapMB)
	default:
		p.CacheRAMMB = min(cacheRAMCapMB, int(headroom/2/MiB))
		p.note("cache-ram %d MB: half the headroom, at most %d", p.CacheRAMMB, cacheRAMCapMB)
	}
	if h.GPU != GPUCUDA && h.GPU != GPUROCm {
		cache := uint64(p.CacheRAMMB) * MiB
		if cache >= kvBudget {
			return *p, fmt.Errorf("cache-ram %d MB leaves nothing for the KV cache (%s after weights)", p.CacheRAMMB, HumanBytes(kvBudget))
		}
		kvBudget -= cache
	}

	p.Slots, p.Context = cfg.Model.Slots, cfg.Model.Context
	autoSlots, autoCtx := p.Slots == 0, p.Context == 0
	if autoSlots {
		switch {
		case p.Budget >= 64*GiB:
			p.Slots = maxSlots
		case p.Budget >= 24*GiB:
			p.Slots = 4
		default:
			p.Slots = 2
		}
	}
	if autoCtx {
		p.Context = contextCap
		if m.Context > 0 && m.Context < p.Context {
			p.Context = m.Context
		}
	}
	fits := func() bool { return uint64(p.Slots)*uint64(p.Context)*uint64(p.KVPerToken) <= kvBudget }
	if !autoSlots && !autoCtx {
		if !fits() {
			p.note("WARNING: %d slots of %d tokens need %s of KV cache, more than the %s available; running as configured",
				p.Slots, p.Context, HumanBytes(uint64(p.Slots)*uint64(p.Context)*uint64(p.KVPerToken)), HumanBytes(kvBudget))
		} else {
			p.note("%d slots of %d tokens from config", p.Slots, p.Context)
		}
		return *p, nil
	}
	// Trade context for slots down to the knee, then slots, then the rest.
	for !fits() {
		switch {
		case autoCtx && p.Context/2 >= contextKnee:
			p.Context /= 2
		case autoSlots && p.Slots > 1:
			p.Slots /= 2
		case autoCtx && p.Context/2 >= contextFloor:
			p.Context /= 2
		default:
			return *p, fmt.Errorf("even %d slot(s) of %d tokens need %s of KV cache; %s is available after weights and cache",
				p.Slots, p.Context, HumanBytes(uint64(p.Slots)*uint64(p.Context)*uint64(p.KVPerToken)), HumanBytes(kvBudget))
		}
	}
	p.note("%d slots of %d tokens: %s of KV cache at %d KiB per token, within %s",
		p.Slots, p.Context, HumanBytes(uint64(p.Slots)*uint64(p.Context)*uint64(p.KVPerToken)), p.KVPerToken/1024, HumanBytes(kvBudget))
	return *p, nil
}

func (p *Plan) sizeMLX(m ModelInfo, cfg Config, headroom uint64) (Plan, error) {
	// oMLX allocates per request as it goes; the window is a limit on a
	// prompt, not a reservation. Concurrency is what costs memory: each
	// request may hold a full window of KV.
	p.ContextWindow = cfg.Model.Context
	if p.ContextWindow == 0 {
		p.ContextWindow = m.Context
		if p.ContextWindow == 0 {
			p.ContextWindow = contextCap
		}
	}
	perRequest := func() uint64 { return uint64(p.ContextWindow) * uint64(p.KVPerToken) }
	if cfg.MaxConcurrent > 0 {
		p.MaxConcurrent = cfg.MaxConcurrent
		p.note("%d concurrent requests from config", p.MaxConcurrent)
		return *p, nil
	}
	for cfg.Model.Context == 0 && perRequest() > headroom && p.ContextWindow/2 >= contextKnee {
		p.ContextWindow /= 2
	}
	p.MaxConcurrent = int(headroom / perRequest())
	if p.MaxConcurrent < 1 {
		return *p, fmt.Errorf("one request of %d tokens needs %s of KV cache; %s is available after weights",
			p.ContextWindow, HumanBytes(perRequest()), HumanBytes(headroom))
	}
	p.MaxConcurrent = min(p.MaxConcurrent, maxConcurrent)
	p.note("%d concurrent requests: each may hold %s of KV cache for a %d-token window, within %s, at most %d",
		p.MaxConcurrent, HumanBytes(perRequest()), p.ContextWindow, HumanBytes(headroom), maxConcurrent)
	return *p, nil
}

func (p *Plan) note(format string, args ...any) {
	p.Notes = append(p.Notes, fmt.Sprintf(format, args...))
}
