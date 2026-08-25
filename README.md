# llm-bench

Benchmarks for local LLM servers (colibri, llama.cpp) on consumer GPU hardware.

Measures throughput (tok/s), TTFT, and resource usage (CPU/GPU/VRAM, plus colibri's VRAM expert hit rate) against a standard prompt protocol.

## Protocol

30 prompts per file (Q1-10 EN, Q11-20 DE, Q21-30 FR), 100 max tokens:

- **Q1** = cold measurement
- **Q2-Q20** = heat buildup, discarded
- **Q21-Q30** = warm measurements (mean + stddev), resource sampler runs during warm

`prompts/` ships 6 domains (`math`, `geography`, `history`, `philosophy`, `physics`, `chemistry`), plus `math_repeat.txt` (30× identical question, for hit-rate-vs-speed runs).

## Run

```sh
uv sync

# launch a server in background
uv run bench start --backend llama --port 8888
uv run bench start --backend colibri --container int4 --port 8888   # or --container gs64

# benchmark a running server (serve mode)
uv run bench serve --mode llama --model qwen3.6-35b-a3b --prompts prompts/math.txt
uv run bench serve --mode colibri --model qwen3.6-colibri --prompts prompts/math.txt

# colibri standalone (fresh process per prompt, heat accumulates via HEAT_FILE, VRAM hit rate)
uv run bench standalone --prompts prompts/math.txt --snap $HOME/models/qwen36-35b-a3b-colibri-i4

# all 18 series (6 domains x 3 backends: llama / colibri-int4 / colibri-gs64)
uv run bench run-all

# custom battery: JSON list of Run definitions (name, backend, mode, model, prompts, ...)
uv run bench run-all --runs results/runs.json

# parse results/bench.log into a Markdown comparison table -> results/table.md
uv run bench analyze
```

All commands above are verified against the current CLI (`bench --help`).

## Results

**Hardware:** GTX 1660 Ti / 6 GB VRAM / sm_75 · Ryzen 7 5800X · 64 GB RAM · CPU: Qwen3.6-35B-A3B MoE (35B total / 3B active, 256 experts/layer)
**Full logs:** `results/*.txt` (single series) and `results/bench.log` (run-all, raw output + one `JSON:` line per run — the machine-readable contract `analyze` reads).

### Tier effect — CUDA tier budget vs speed (colibri int4, math domain)

| Run | Config | cold tok/s | warm tok/s | warm Hit% | TTFT (warm) | VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | CPU-only (`COLI_CUDA=0`) | 0.85 | **0.82** | — | ~38 s | 0.35 GB |
| A2 | Tier GB=1 (~675 experts) | 0.80 | **6.20** | 40.8% | ~2 s | 0.92 GB |
| A3 | Tier GB=5 (~3378 experts) | 5.19 | **6.57** | 81.1% | ~2 s | 3.43 GB |

**Finding:** the CUDA tier gives **~7.5× over CPU** already at 1 GB budget; 1 GB → 5 GB adds only +6%. GPU load stays below 8% in all configs — the GPU is never the bottleneck.

### Hit rate vs speed (colibri int4, GB=5, 30× identical question)

| | cold | warm |
| --- | --- | --- |
| tok/s | 6.10 | **6.91** (+13%) |
| VRAM hit | 33.3% | **95.3%** |

**Finding:** hit rate 33 → 95% yields only +13% speed. Hit rate is *not* the speed driver — the CPU path (int8 dense/attention) is the bottleneck.

### llama.cpp baseline (`--cpu-moe`, persistent server)

Mean **19.93 tok/s** (max 20.69, std 0.47) — flat across all 6 domains (`results/llama_baseline.txt`; regenerate via `bench serve --mode llama --model qwen3.6-35b-a3b ... > results/llama_baseline.txt`).

### Caveat: cold measurements are not reproducible

Same config gives cold 0.80 / 5.19 / 6.10 tok/s depending on whether weights are in the OS page cache. Cold measures page-cache temperature — discard the column unless you clear caches before Q1.

## Environment quirks (GTX 1660 Ti / Nix, for reproducibility)

### colibri build

```sh
cd ~/src/colibri
nix-shell -p cudaPackages_12.cuda_nvcc cudaPackages_12.cuda_cudart cudaPackages_12.cuda_cccl gmp gnumake \
  --run 'make -C c qwen36 CUDA=1 CUDA_ARCH=sm_75'
```

### Nix stub-libcuda workaround

Nix `cuda_cudart` ships a stub `libcuda.so` that shadows the real driver. Wrap the engine:

```sh
cd ~/src/colibri/c
mv qwen36 qwen36.real
cat > qwen36 <<'EOF'
#!/bin/sh
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1
exec "$(dirname "$0")/qwen36.real" "$@"
EOF
chmod +x qwen36
```

### colibri serve bug: `--gpu auto` fails for non-GLM engines

`cuda_binary()` in `c/coli` checks `GLM` not `engine_for(model)`. Workaround: set `COLI_CUDA=1` directly, no `--gpu` flag.

### ntfy push (optional)

Measurement summaries are auto-pushed to ntfy (XX after each `standalone` run). Topic/server come from `config/ntfy.env` (copy `config/ntfy.example.env`).
