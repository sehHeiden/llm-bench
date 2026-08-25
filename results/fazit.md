# Findings — Qwen3.6-35B-A3B: llama.cpp vs colibri CUDA tier

**Hardware:** GTX 1660 Ti / 6 GB VRAM / sm_75 · CPU: Ryzen 7 5800X · 64 GB RAM
**Model:** Qwen3.6-35B-A3B (MoE, 35B total / 3B active, 256 experts/layer, 40 layers)
**Prompts:** 30 per domain (Q1-10 EN, Q11-20 DE, Q21-30 FR), 100 max tokens
**Date:** August 2026
**Repo:** <https://github.com/sehHeiden/llm-bench> (benchmark code + full result logs)

---

## 1. Tier effect (colibri int4, math)

| Run | Config | cold tok/s | cold Hit | warm tok/s | warm Hit | CPU% | GPU% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | CPU-only (COLI_CUDA=0) | 0.85 | — | **0.82** | — | 77.5 | 3.1 |
| A2 | Tier GB=1 (~675 exp.) | 0.80 | 7.1% | **6.20** | 40.8% | 56.1 | 3.7 |
| A3 | Tier GB=5 (~3378 exp.) | 5.19 | 33.3% | **6.57** | 81.1% | 55.7 | 7.6 |

**Finding:** CUDA tier = **~7.5× speedup** (0.82 → 6.20 tok/s) already with 1 GB budget.
GB=1 → GB=5 adds only +6% (6.20 → 6.57). More VRAM residency is barely worth it on 6 GB.
GPU load stays below 8% in all configs — **the GPU is never the bottleneck**.

## 2. Hit rate vs speed (colibri int4, GB=5, 30× identical question)

| | cold | warm |
| --- | --- | --- |
| tok/s | 6.10 | **6.91** (+13%) |
| VRAM-Hit | 33.3% | **95.3%** |

**Finding:** Hit rate 33 → 95% produces only +13% speed. **Hit rate is NOT the speed driver.**
The bottleneck is the CPU path (int8 dense/attention), not the hit rates.

## 3. Cold measurement not reproducible

| Run | cold tok/s | TTFT cold |
| --- | --- | --- |
| A2 (after long pause) | 0.80 | 42.4 s |
| A3 (immediately after) | 5.19 | 2.1 s |
| B (immediately after) | 6.10 | 2.1 s |

**Finding:** 3× difference in "cold" with the same config — cause is the OS page cache
(first read from disk vs weights still in RAM). **Cold measures page-cache temperature;
without cache clearing before Q1 the column is unusable.** Cold hit rate was flat (~7-33%).

## 4. Domain comparison (all 6 domains, warm)

| Domain | llama tok/s | int4 tok/s | int4 Hit% | gs64 tok/s | gs64 Hit% |
| --- | --- | --- | --- | --- | --- |
| math | 19.29 | 6.92 | 81.1 | 6.93 | 78.1 |
| geography | 19.70 | 7.08 | 77.2 | 6.81 | 74.3 |
| history | 19.53 | 6.90 | 82.6 | 6.95 | 78.6 |
| philosophy | 19.53 | 6.76 | 81.6 | 6.65 | 79.8 |
| physics | 19.50 | 6.90 | 78.3 | 6.56 | 76.2 |
| chemistry | 19.33 | 6.61 | 79.4 | 5.85* | 75.6 |

*chemistry gs64 cold outlier; warm values stable.

**Finding:** Domains are irrelevant for speed — llama flat ~19.5 across all domains,
colibri flat ~6.8. Qwen's supposed math strength does NOT show in tok/s (routing
differences are too small against the CPU path). int4 ≈ gs64 (speed indistinguishable).

## 5. Overall comparison

| Backend | warm tok/s | Ratio |
| --- | --- | --- |
| llama.cpp (--cpu-moe, -ngl 99) | **19.93** | 1.0× |
| colibri tier GB=5 | 6.57 | 3.0× slower |
| colibri tier GB=1 | 6.20 | 3.2× slower |
| colibri CPU-only | 0.82 | 24× slower |

## Key takeaways

1. **llama.cpp ~3× faster** than best colibri (19.9 vs 6.9 tok/s).
   Reason: llama puts attention/dense on the GPU, colibri leaves them on CPU-int8.
2. **CUDA tier gives 7.5×** over CPU-only — but the win comes from the first GB;
   more residency (+0.4 tok/s) is not worth it on 6 GB.
3. **Hit rate ≠ speed**: 33→95% Hit = only +13%. The CPU path is the bottleneck.
4. **Cold measurement needs page-cache clearing** before Q1, otherwise unusable.
5. **Domains don't matter**: no domain dependency in performance.

## Implication

On a 6 GB card, colibri's CUDA tier for Qwen3.6-35B-A3B is **effectively without speed benefit**:
it accelerates only the small expert kernels (GPU <8% load), while the dominant CPU path
(dense + attention, int8) stays unchanged. llama.cpp --cpu-moe is the right choice for
this card/model combination.
