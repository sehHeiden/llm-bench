# llm-bench

Benchmarks for local LLM servers (colibri, llama.cpp) on consumer GPU hardware.

## Setup (TuxedoOS + Nix, GTX 1660 Ti / 6 GB / sm_75)

- colibri: PR #712 + #713 (merged), built from `~/src/colibri`, CUDA 12.8 via `nix-shell`
- llama.cpp: `llama-server` with `--cpu-moe`, Qwen3.6-35B-A3B-Q4_K_M.gguf
- Model: Qwen3.6-35B-A3B

### colibri build

```sh
cd ~/src/colibri
nix-shell -p cudaPackages_12.cuda_nvcc cudaPackages_12.cuda_cudart cudaPackages_12.cuda_cccl gmp gnumake \
  --run 'make -C c qwen36 CUDA=1 CUDA_ARCH=sm_75'
```

### Nix stub-libcuda workaround (environment quirk, not colibri)

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

## Prompts

10 prompts each across 6 domains (60 total):

- `prompts/math.txt`
- `prompts/geography.txt`
- `prompts/history.txt`
- `prompts/philosophy.txt`
- `prompts/physics.txt`
- `prompts/chemistry.txt`

## Run

```sh
uv sync

# launch a server in background (llama serve / colibri serve)
uv run bench start --backend llama --port 8888
uv run bench start --backend colibri --container int4 --port 8888

# benchmark a running server (serve mode, Q1 cold / Q21-30 warm)
uv run bench serve --mode llama --model qwen3.6-35b-a3b --prompts prompts/math.txt
uv run bench serve --mode colibri --model qwen3.6-colibri --prompts prompts/math.txt

# colibri standalone (fresh process per prompt, heat accumulates, VRAM hit rate)
uv run bench standalone --prompts prompts/math.txt --snap ~/models/qwen36-35b-a3b-colibri-i4

# all 18 series (6 domains x 3 backends), results → results/bench.log
uv run run-all

# parse results into a Markdown comparison table → results/table.md
uv run analyze
```

## First results (Issue #1040, comment 2026-08-16)

GTX 1660 Ti / 6 GB / sm_75, `CUDA_EXPERT_GB=5` → 3378/10240 experts in VRAM (33%), RSS 42 GB.

| | tok/s | TTFT |
| --- | --- | --- |
| llama `--cpu-moe` (persistent, Ø) | 18.34 (max 19.56) | 0.70–0.81 s |
| colibri standalone cold (run 0) | 1.25 | 40.82 s |
| colibri standalone hot max (run 9) | 5.66 | 2.02 s |
| colibri serve (persistent, max req 8) | 3.82 (Ø 3.66) | n/a |

`results/colibri_1040_datapoint.md` — full comment posted to <https://github.com/JustVugg/colibri/issues/1040#issuecomment-5309768913>
