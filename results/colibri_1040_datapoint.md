## 1660 Ti Datapoint — CUDA expert tier (PR #713, branch `qwen36-cuda-tier`)

Following kreuzzelg's invitation for an under-represented card size. Setup + numbers below.

### Environment

- **GPU:** NVIDIA GeForce GTX 1660 Ti, 6 GB VRAM, sm_75, driver 580.178 (CUDA 13.0 capable)
- **OS:** TuxedoOS (based Ubuntu 24.04 LTS) + Nix — CUDA 12.8 toolkit (`cudaPackages_12.cuda_nvcc` + `cuda_cudart` + `cuda_cccl`) via `nix-shell`
- **Build:** `make -C c qwen36 CUDA=1 CUDA_ARCH=sm_75` — clean, only `-Wformat-truncation` warnings, no errors
- **Container:** `Kreuzzelg/qwen36-35b-a3b-colibri-i4` (per-row int4, ~20 GB, 47 safetensors) — note this is **per-row int4, not gs64**
- **RAM:** 62 GB total, 54 GB free
- **Engine flags:** `--cap 256` (required: `cap == n_experts` for tier), `COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=5`
- **Branch:** `qwen36-cuda-tier` from kreuzzelg's fork — PR #713 contains PR #712 (stacked, verified `git merge-base --is-ancestor`)

### Prompts used (10 math problems, 100 max tokens each, `stream=false`)

```
0  Solve: 3x + 7 = 22. Find x.
1  What is 17 * 23? Show the steps.
2  If a train travels 60 km/h for 2.5 hours, how far does it go?
3  Solve the quadratic equation x^2 - 5x + 6 = 0.
4  Compute the derivative of f(x) = 3x^3 - 2x + 5.
5  What is the integral of 2x dx from 0 to 4?
6  A rectangle has area 24 and width 4. Find the perimeter.
7  Solve: 2(x - 3) = 8. Find x.
8  What is 15% of 480?
9  Factor the polynomial x^2 + 7x + 12.
```

### Build note — Nix CUDA driver mismatch (environment quirk, not colibri)

On TuxedoOS with Nix packages, the Nix `cuda_cudart` package ships a stub `libcuda.so` in `…/lib/stubs/` that shadows the real driver lib on the engine's RUNPATH. This is a Nix packaging quirk on this host, **not a colibri bug**. Symptom:

```
[CUDA] device discovery: CUDA driver version is insufficient for CUDA runtime version
[qtier] coli_cuda_init failed -> CPU path
```

Driver 580.178 supports CUDA 13.0, so this is the stub talking, not the real driver. Workaround: wrap the engine binary in a shell script that `LD_PRELOAD`s the real driver lib:

```sh
#!/bin/sh
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1
exec "$(dirname "$0")/qwen36.real" "$@"
```

(`LD_LIBRARY_PATH` breaks the Nix glibc; `LD_PRELOAD` scoped to the engine only is clean.) Worth a README note for Nix users, but nothing for colibri to fix.

### `coli serve` note — `--gpu auto` does not enable CUDA for non-GLM engines (actual colibri bug)

`cuda_binary()` in `c/coli` checks `GLM` (= the colibri engine path), not `engine_for(model)` for the qwen36 engine. With `--gpu auto` on the qwen36 engine, `cuda_binary()` returns `False` → `coli` exits with `--gpu needs the CUDA build` even though `ldd c/qwen36` shows `libcudart.so.12` linked. This is a real colibri bug, independent of the Nix stub above. Workaround: set `COLI_CUDA=1` directly in the env (no `--gpu` flag). Fix: `cuda_binary()` should resolve against the actual engine binary for the model's arch, not just `GLM`.

### Tier sizing — explicit 5 GB VRAM budget

We raised the expert-tier budget to **`CUDA_EXPERT_GB=5`** (auto would be ~4 GB = free minus 1 GB headroom) to maximise residency on the 6 GB card. This is the largest budget that fits without OOM alongside the CUDA context.

```
[CUDA] device 0: NVIDIA GeForce GTX 1660 Ti, 6.0 GB VRAM, sm_75
[qtier] dev 0: 5.2 GB free, budget 5.0 GB (~3378 experts)
[qtier] CUDA VRAM expert tier active: 1 device(s), 1.52 MB/expert
[qtier] warmstart (parallel): all 10240 experts in RAM, 3378 in VRAM -- 34.4 s
```

3378/10240 = **33% of experts resident in VRAM**. All 10240 always in RAM (int4 packed + int8 dequant on demand). Peak RSS **42 GB** standalone (cap=256 KV + full int8 residency).

### colibri — standalone (fresh process per run, heat accumulates via `HEAT_FILE`)

Run 0 (cold): 1.25 tok/s, TTFT 40.82 s, VRAM-hit 33.3 %. Run 9 (hot): 5.66 tok/s, TTFT 2.02 s, VRAM-hit 85.6 %.

Representative `qt_stats` block (hot run 9):

```
[qtier] resident 3378/10240 experts | uploads 3384 | miss(CPU) 5312 | q_skips 0
[qtier]   dev 0: hits 31488 | 10134 tensors, 4.99 GB VRAM used (budget 5.00 GB)
[qtier] VRAM hit rate: 85.6 % | LFRU swaps 6
[qtier] group_stats: 4598 calls, 31488 experts | h2d 0 ms, kernel 0 ms, d2h 0 ms
Expert cache hit rate: 80.4% (hit=... miss=10240)
PEAK RSS: 41.98 GB
Speed: 5.66 tok/s (17.7s for 100 tokens)
```

`miss=10240` = 40 layers × 256 experts, i.e. every expert dequantised once per process (full RAM residency). RAM-cache-hit ~80% measures int8-materialised vs freshly-dequantised, **not** disk loading — disk I/O only happens at process start ("resident weights loaded in 7.5s").

### colibri — persistent serve (VRAM retained across requests, heat pre-seeded)

Ø 3.66 tok/s, max 3.82 tok/s (req 8), flat — no warmup curve because heat was pre-seeded. Per-request `qt_stats` is not emitted in serve mode (`qt_stats()` only fires on non-serve exit), so per-request VRAM-hit isn't directly visible.

### llama.cpp baseline — `--cpu-moe`, persistent server, same 10 prompts

`llama-server` with `Qwen3.6-35B-A3B-Q4_K_M.gguf`, `-ngl 99 --cpu-moe -c 131072 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0`.

Ø 18.34 tok/s, max 19.56 tok/s (req 8).

### Summary

Same 10 math prompts, 100 max tokens each, `stream=false`.

| | tok/s (gen) | TTFT |
| --- | --- | --- |
| llama `--cpu-moe` (persistent, Ø) | 18.34 (max 19.56) | 0.70–0.81 s |
| colibri standalone cold (run 0) | 1.25 | 40.82 s |
| colibri standalone hot max (run 9) | 5.66 | 2.02 s |
| colibri serve (persistent, heat-seeded, max req 8) | 3.82 (Ø 3.66) | n/a |

Happy to re-run with a gs64 container if someone points me at a pre-converted one (the `Kreuzzelg/qwen36-35b-a3b-colibri-i4` mirror is per-row int4, not gs64), or to test a patch that exposes per-request `qt_stats` in serve mode.
