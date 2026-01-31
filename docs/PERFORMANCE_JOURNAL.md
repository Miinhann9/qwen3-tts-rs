# Performance Journal

## Baseline (Jan 2025, DGX Spark)

| Label | Words | Wall (ms) | Audio (s) | RTF | Tok/s | Prefill | Generate | Decode |
|-------|-------|-----------|-----------|-----|-------|---------|----------|--------|
| short | 13 | 5235 | 3.68 | 1.423 | 8.8 | 21ms (1%) | 2724ms (71%) | 1109ms (29%) |
| medium | 53 | 23786 | 34.00 | 0.700 | 17.9 | 20ms (0%) | 22694ms (95%) | 1057ms (4%) |
| long | 115 | 43797 | 60.96 | 0.718 | 17.4 | 19ms (0%) | 41861ms (96%) | 1886ms (4%) |

## Optimizations 1-3: GPU sync elimination (2026-01-31)

All three optimizations applied together:
1. Batch code_predictor argmax (15 fewer GPU→CPU syncs per frame)
2. GPU-side top-k/top-p filtering (no logit vector CPU transfer)
3. GPU-side repetition penalty & EOS suppression (on-device masking)

| Label | Words | Wall (ms) | ±stddev | Audio (s) | RTF | Tok/s | Mem (MB) | Prefill | Generate | Decode |
|-------|-------|-----------|---------|-----------|-----|-------|----------|---------|----------|--------|
| short | 13 | 2430 | 3 | 3.68 | 0.660 | 18.9 | 834 | 19ms (1%) | 2274ms (94%) | 136ms (6%) |
| medium | 53 | 22159 | 90 | 33.12 | 0.669 | 18.7 | 835 | 19ms (0%) | 21130ms (95%) | 1009ms (5%) |
| long | 115 | 41766 | 11 | 60.32 | 0.692 | 18.1 | 835 | 20ms (0%) | 39906ms (96%) | 1839ms (4%) |

## Optimization 4: Eliminate no-op causal masks (2026-01-31)

Skip creating all-zeros causal masks for single-token generation steps (both
talker and code_predictor). Also cached the token suppression mask to avoid
rebuilding it each frame.

| Label | Words | Wall (ms) | ±stddev | Audio (s) | RTF | Tok/s | Mem (MB) | Prefill | Generate | Decode |
|-------|-------|-----------|---------|-----------|-----|-------|----------|---------|----------|--------|
| short | 13 | 2412 | 7 | 3.68 | 0.655 | 19.1 | 835 | 19ms (1%) | 2258ms (94%) | 134ms (6%) |
| medium | 53 | 21935 | 94 | 33.12 | 0.662 | 18.9 | 836 | 20ms (0%) | 20903ms (95%) | 1010ms (5%) |
| long | 115 | 41320 | 59 | 60.32 | 0.685 | 18.2 | 847 | 20ms (0%) | 39476ms (96%) | 1823ms (4%) |

## Summary

| Label | Baseline RTF | Final RTF | Speedup | Baseline tok/s | Final tok/s |
|-------|-------------|-----------|---------|---------------|-------------|
| short | 1.423 | 0.655 | **2.17x** | 8.8 | 19.1 |
| medium | 0.700 | 0.662 | **1.06x** | 17.9 | 18.9 |
| long | 0.718 | 0.685 | **1.05x** | 17.4 | 18.2 |

## Analysis: Theoretical Ceiling

Chrome trace analysis of per-frame timing (long sentence, 756 frames):

| Span | Avg (ms) | % of frame |
|------|----------|------------|
| code_predictor | 25.88 | 50% |
| sampling (incl. GPU sync) | 15.78 | 30% |
| talker_step (CPU launch) | 10.15 | 20% |
| **Total** | **51.81** | |

The "sampling" span absorbs the GPU sync cost for the preceding async talker_step.
Actual GPU compute per frame is ~51ms, dominated by model forward passes (talker
28 layers + code_predictor 5 layers × 15 autoregressive steps).

At ~52ms/frame, theoretical max is ~19.2 tok/s. We're at 18.2-19.1 tok/s
(**~95% of theoretical throughput**). The remaining gap is framework overhead.

### What won't help further

- **Flash attention**: Tested — no improvement for single-token KV-cache steps
  (batch=1, seq_len=1). Flash attention benefits long-sequence prefill, not generation.
- **Further GPU sync reduction**: Only 1 unavoidable sync per frame remains
  (sampling the semantic token for the next iteration's embedding lookup).

## Final Numbers (v0.3.0, all variants, sequential isolated runs)

| Model | RTF (short) | RTF (long) | Tok/s | TTFA | Memory |
|-------|-------------|------------|-------|------|--------|
| 0.6B Base | 0.56 | 0.68 | 22.2 | 448 ms | 814 MB |
| 1.7B Base | 0.72 | 0.74 | 17.3 | 590 ms | 761 MB |
| 1.7B CustomVoice | 0.72 | 0.75 | 17.3 | 585 ms | 761 MB |
| 1.7B VoiceDesign | 0.72 | 0.75 | 17.3 | 585 ms | 761 MB |

### Remaining opportunities (diminishing returns)

- **Pre-allocated KV cache**: Replace `Tensor::cat` per step with pre-allocated
  buffer + index writes. Saves memory allocation overhead.
- **Quantization (INT8/INT4)**: Reduce memory bandwidth for matmuls.
- **Custom CUDA kernels**: Fused attention + MLP, fused embedding + projection.
- **Batched inference**: Process multiple utterances simultaneously.
