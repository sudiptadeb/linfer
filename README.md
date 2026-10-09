# linfer

linfer sets up the best local inference backend for one model on the machine
it runs on, without the hassle. It looks at the hardware, installs the right
backend (llama.cpp's `llama-server`, or oMLX on Apple Silicon), fetches the
weights, sizes the launch to the memory left after them, runs the backend
under supervision, and tells your client one URL and one model id that never
change. It is not a proxy: requests go from your client straight to the
backend's own OpenAI-compatible `/v1` endpoint.

## What it does

- **Detects** the machine: OS and architecture; on Apple Silicon the chip,
  unified memory and the Metal working set; on Linux an NVIDIA (VRAM, CUDA
  version), AMD (ROCm) or Vulkan GPU, else CPU; RAM and free disk.
- **Installs the backend**: the official prebuilt llama.cpp release for the
  platform (macOS Metal; Linux CUDA, ROCm, Vulkan or CPU), pinned to a
  known-good build and verified with `--version`; on Apple Silicon also oMLX
  into its own virtualenv, pinned to a commit. A binary you already have is
  used as is.
- **Fetches the model** from a Hugging Face repo (a GGUF file, or a whole MLX
  repo) into its own directory, resumable, verified by size and sha256,
  after checking it fits on the disk and in memory. Local paths work too.
- **Picks and tunes**: `auto` is oMLX on Apple Silicon when MLX weights are
  available, else llama.cpp with the right GPU build. Slots, context per
  slot, prompt cache and concurrency are computed from the memory headroom;
  anything you set explicitly wins.
- **Serves**: launches the backend as a child, health-checks it, logs its
  memory, restarts it when it dies, and lets you pause it (freeing its
  memory for something else) and resume it without touching the service.

## Quick start

Build (Go 1.22 or later):

```sh
go build -o linfer ./cmd/linfer
```

Write `~/.config/linfer/linfer.yaml` (see the example below), then:

```sh
linfer setup     # install the backend, fetch the weights, print the URL and model id
linfer serve     # run it; put this under launchd or systemd for real use
linfer status    # in another shell
```

### macOS (Apple Silicon)

A config with both `gguf` and `mlx` weights lets `auto` choose oMLX; one with
only `gguf` runs llama.cpp with Metal. oMLX is installed from git into
`~/.local/share/linfer/venv-omlx` with a Python 3.11–3.13 (found on PATH, or
through `uv`). The Metal working set is measured through MLX once that venv
exists, and estimated before.

### Linux (NVIDIA, or CPU)

With `nvidia-smi` present, `setup` downloads the CUDA build matching the
driver's CUDA line (12.8 or 13.x) plus its runtime libraries; with no GPU
tool answering it takes the CPU build (set `gpu: vulkan` or `gpu: rocm` to
force those). The weights are sized against VRAM; the prompt cache against
system RAM.

## Example config

```yaml
# ~/.config/linfer/linfer.yaml
backend: auto                 # auto | llama | mlx
listen: 127.0.0.1:8090        # the backend's address; the client's base_url is http://127.0.0.1:8090/v1

model:
  id: qwen3-coder:30b-q8_0    # what the client names; the same on every backend
  gguf: hf://Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf
  # mmproj: hf://…            # a vision projector, if the model has one
  # mlx: hf://mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit   # lets auto pick oMLX on Apple Silicon
  # slots: 8                  # llama-server parallel slots; 0 = auto
  # context: 131072           # tokens per slot; 0 = auto
  reasoning_effort: medium    # passed to the chat template

# Binaries: empty means linfer's own install under dir.
# llama_bin: /usr/local/bin/llama-server
# omlx_bin: ~/venv-omlx/bin/omlx
# llama_release: b11429
# gpu: cuda                   # Linux override: cuda | rocm | vulkan | cpu
# dir: /big-disk/linfer       # weights, binaries, state and the control socket; default ~/.local/share/linfer
# keep_free_gb: 200           # a download must leave this much disk free
# cache_ram_mb: 0             # llama-server --cache-ram; 0 = auto
# max_concurrent: 0           # oMLX --max-concurrent-requests; 0 = auto (at most 6)
```

Paths may start with `~`. An `hf://org/repo/file` reference is a GGUF file;
`hf://org/repo` is a whole MLX repo. Set `HF_TOKEN` for gated repos.

## Commands

| command | what it does |
|---|---|
| `linfer setup` | detect, install what is missing, fetch the weights, size the launch, print the URL and model id |
| `linfer doctor [--json]` | the same report without changing anything |
| `linfer serve` | run the backend under supervision; this is the service |
| `linfer status [--json]` | backend, URL, model id, pid, healthy, RSS, restarts |
| `linfer pause` | stop the backend and free its memory; the daemon stays up |
| `linfer resume` | start it again |
| `linfer switch llama\|mlx\|auto` | change backend for this daemon's lifetime and restart the child |

`-config FILE` before the command names another config. `serve` talks to
`status`, `pause`, `resume` and `switch` over a unix socket under `dir`, so
only the user who owns that directory can pause the model.

## How selection and sizing work

**Backend.** `auto` is oMLX only on darwin/arm64 when the model has `mlx`
weights that are present and an `omlx` binary exists; everything else is
llama.cpp. `doctor` prints the reason.

**Budget.** On Apple Silicon the budget is the Metal working set (measured
through MLX, else the `iogpu.wired_limit_mb` sysctl, else an estimate of 87%
of RAM, which is what macOS gave a 256 GB machine); on CUDA and ROCm it is
the card's VRAM; otherwise 80% of RAM. Weights that do not fit in the budget
less a 4 GiB reserve are refused before anything is downloaded.

**llama.cpp.** From the headroom (budget − weights − reserve): the prompt
cache (`--cache-ram`) is half the headroom, at most 16 GiB, or a quarter of
system RAM on a discrete GPU; slots start at 8 (budget ≥ 64 GiB), 4 or 2;
context per slot starts at the smaller of the model's context and 131,072.
The KV cache a token costs is read from the GGUF header (attention layers ×
KV heads × head size × 2 bytes × K and V; hybrid models count only their
full-attention layers). While slots × context × KV exceeds what is left,
context halves down to 32k, then slots halve, then context halves down to
8k. The rest of the launch profile is fixed and measured: 4 restore points
per slot, flash attention, `--jinja`, batch 2048 / micro-batch 512, the
model's own sampling, no speculative decoding.

**oMLX.** The context window is the model's; concurrency is the headroom
divided by one full window of KV, at most 6. `model_settings.json` gets
multi-token prediction, the model pinned, the same sampling, and the
reasoning effort.

Every explicit value in the config overrides its computed one, and a
configuration past the budget runs as told with a warning in the report.

## Benchmark your machine

```sh
linfer bench --quick                       # a few minutes: both backends, one run per cell
linfer bench                               # the full run: 3 runs per cell, 2k and 32k contexts, 1/2/4/8 streams
linfer bench --backends llama --suites speed --contexts 2k,32k --concurrency 1,8 --out ./bench
```

`bench` runs every backend this machine can run for the configured model
(`--backends all`, or a list), **one at a time**, each launched with exactly
the profile `serve` would use: load, warm up, measure, stop, wait for the
exit so the memory is free, then the next. If a `linfer serve` daemon for
this config is running it is paused over the control socket first and
resumed at the end, also on Ctrl-C or an error. Three suites:

- **speed**: for each context size and concurrency level, two passes over
  new prompts sent all at once. The cold pass (one token each) gives time to
  first token and prefill tok/s with every prompt arriving together. After a
  short settle, the warm pass on the now-cached prompts gives decode tok/s
  per stream, combined tok/s (all streams' tokens over the pass's wall time)
  and the warm time to first token: the shape of an agent's turn, a cached
  history and a long reply. Combined output on cold prompts with short
  replies would instead measure how a backend queues prefill (oMLX takes new
  prompts one at a time), which can read a third of the decode throughput
  the same backend gives agents. Measured on the
  client from the stream, identically for every backend; the backend's own
  figures (llama-server's `timings`, oMLX's `usage`) are shown beside them,
  with the stream's granularity (tokens per chunk), since a backend that
  streams in bursts looks different to a client than one that streams a
  token at a time. Medians of N runs, with the range.
- **tools**: nine tool-calling cases over generic schemas (weather with an
  enum, calculator, file read/write, web search with an integer, a calendar
  event with a nested object and an array, an account lookup): the right
  tool among several, schema-valid arguments, valid JSON, no call when none
  is needed, two calls in one turn, a follow-up after a tool result, and a
  10-digit id copied exactly from earlier context as a string. Scored for
  valid calls, right tool, schema, no-call correctness, reasoning or markup
  leaking into content, and HTTP errors (an oMLX prefill-memory 400 is a
  failure, listed).
- **context**: needle-in-haystack at several depths, at each context size
  up to the slot's. Exact retrieval or not.

It prints a comparison table and a short verdict, and writes `report.md` and
`results.json` to `--out` (default `<dir>/bench/<timestamp>`). The verdict
names the fastest single stream, the most combined output at the top
concurrency, tool accuracy per backend and any failures, and says when a
difference is within the runs' own spread. A sample, from a quick run of a
0.6B model on both backends (small on purpose, and from before the cold/warm
split; the numbers are the format, not a recommendation):

```
metric                         llama                 mlx
weights                        0.5 GiB               0.3 GiB
load                           1s, rss 2.2 GiB       2s, rss 797 MiB
launch                         2×4096, cache 256 MB  2 concurrent, window 4096
decode tok/s/stream 512×1      312.2                 430.1
  server-reported              300.9                 381.6
  tokens per stream chunk      1.0                   32.0
combined tok/s 512×1           312.2                 430.1
cold ttft s 512×1              0.06                  0.15
prefill tok/s 512×1            8749                  3615
  server-reported              11658                 12553
warm ttft s 512                0.01                  0.11
decode tok/s/stream 512×2      278.0 (278.0–278.1)   318.9 (312.8–325.0)
combined tok/s 512×2           555.8                 539.5
…
tools accuracy                 100% (9/9)            100% (9/9)
  valid / right tool / schema  7 / 7 / 7 of 7        7 / 7 / 7 of 7
  no-call correct              2 of 2                2 of 2
  leaks / http errors          0 / 0                 0 / 0
needle found                   1 of 2                2 of 2
failures                       0                     0

- fastest single agent (512 context, 1 stream): mlx at 430.1 tok/s, 1.38× llama (312.2)
- most combined output (1k context, 2 streams): llama at 518.2 tok/s; mlx at 498.3 is within noise
- tool accuracy: llama 100% (9/9), mlx 100% (9/9)
- needle retrieved: llama 1/2, mlx 2/2
- single run per cell: no spread to judge noise by, so treat differences under ~10% as unproven
```

Variants are rows: a backend with a twist (a speculative decoder, another
quantisation) can be added as another `Variant` without touching the suites.

## Why

The sizing constants come from one reference machine, a 256 GB Apple
Silicon workstation, and reproduce what was measured there:

- On the same q8 weights of a ~125B mixture-of-experts model, a current
  llama.cpp `llama-server` gave twice Ollama's combined output at 8
  concurrent sessions of 33k context (104 against 52 tokens/s), with every
  replayed tool call intact.
- oMLX with multi-token prediction was about 2× faster again for a single
  stream on that machine, which is why `auto` prefers it there; 8 concurrent
  ~65k-token requests ran it out of Metal memory and 6 did not, hence the
  cap.
- A 176 GiB GGUF on that machine gets 8 slots of 131,072 tokens with a 16 GiB
  prompt cache; a 182 GiB MLX model gets 6 concurrent requests. A 24 GB GPU
  with an 8B model gets 2 slots of 32k.

Those numbers are the motivation, not a benchmark suite: measure on your
own machine, and set the values the config lets you set when they differ.

## Status

Early. One model at a time; macOS and Linux; llama.cpp and oMLX. The tests
(`go test ./...`) run with no GPU, no model and no network. Things not done
yet: Windows, several models, an AMD or Vulkan machine actually tried.

## Contributing

Issues and pull requests are welcome. Keep tests plain (one set-up, call and
assertion per behaviour), run `go vet ./...` and `go test ./...` before a
pull request, and say what machine you measured on when a number changes.

## License

Apache-2.0. See [LICENSE](LICENSE).
