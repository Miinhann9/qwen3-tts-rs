# Benchmarks

Performance measurements for `qwen3-tts-rs` inference across CPU and GPU.

All results use default generation parameters
(temperature=0.9, top_k=50, top_p=0.9, repetition_penalty=1.05, seed=42).
2 warmup runs, 3 timed iterations, streaming mode enabled for TTFA measurement.

## Test Hardware

| | Spec |
|---|---|
| **Platform** | NVIDIA DGX Spark |
| **CPU** | ARM Cortex-X925 + Cortex-A725, 20 cores |
| **GPU** | NVIDIA GB10 (Blackwell) |
| **RAM** | 120 GB unified |
| **OS** | Linux 6.14 (aarch64) |
| **CUDA** | 13.0, Driver 580.95 |

## Test Corpus

| Label | Words | Text |
|-------|------:|------|
| Short | 13 | "The quick brown fox jumps over the lazy dog near the river bank." |
| Medium | 53 | "In a quiet village nestled between rolling hills and dense forests, there lived an old clockmaker who spent his days repairing timepieces from centuries past. His workshop, filled with the gentle ticking of a hundred clocks, was a place where time itself seemed to slow down and the outside world faded into silence." |
| Long | 115 | "The development of artificial intelligence has been one of the most transformative technological advances of the twenty-first century. From natural language processing to computer vision, machine learning models have achieved remarkable performance across a wide range of tasks that were once considered the exclusive domain of human intelligence. Speech synthesis, in particular, has seen dramatic improvements with the introduction of neural network architectures that can generate high-fidelity audio from text input. These systems learn complex patterns of prosody, intonation, and rhythm from large datasets of recorded speech, producing output that is increasingly difficult to distinguish from natural human speech. The implications of this technology extend across many fields, including accessibility, entertainment, education, and human-computer interaction." |

## End-to-End Synthesis

Real-time factor (RTF) = wall-clock time / audio duration. **Lower is better; < 1.0 means faster than real-time.**

Each cell shows the average of 3 timed iterations after 2 warmup runs, executed in isolation (no concurrent GPU workloads).

### 0.6B Base — CUDA (BF16)

| Text | Words | Wall Clock | Audio Duration | RTF | TTFA | Tok/s | Memory |
|------|-------|------------|----------------|-----|------|-------|--------|
| Short | 13 | 2.30 sec | 4.08 sec | **0.56** | 448 ms | 22.2 | 814 MB |
| Medium | 53 | 10.08 sec | 17.84 sec | **0.57** | 452 ms | 22.1 | 817 MB |
| Long | 115 | 110.63 sec | 163.84 sec | **0.68** | 456 ms | 18.5 | 841 MB |

> Note: The 0.6B Base model generates significantly more frames per word than 1.7B models,
> producing longer audio from the same text. The RTF increase on the long input reflects
> the higher frame count (2048 frames vs ~529 for 1.7B).

### 1.7B Base — CUDA (BF16)

| Text | Words | Wall Clock | Audio Duration | RTF | TTFA | Tok/s | Memory |
|------|-------|------------|----------------|-----|------|-------|--------|
| Short | 13 | 2.25 sec | 3.12 sec | **0.72** | 590 ms | 17.3 | 761 MB |
| Medium | 53 | 13.24 sec | 18.32 sec | **0.72** | 592 ms | 17.3 | 765 MB |
| Long | 115 | 31.12 sec | 42.32 sec | **0.74** | 591 ms | 17.0 | 771 MB |

### 1.7B CustomVoice — CUDA (BF16)

| Text | Words | Wall Clock | Audio Duration | RTF | TTFA | Tok/s | Memory |
|------|-------|------------|----------------|-----|------|-------|--------|
| Short | 13 | 2.65 sec | 3.68 sec | **0.72** | 585 ms | 17.3 | 761 MB |
| Medium | 53 | 24.11 sec | 33.12 sec | **0.73** | 588 ms | 17.2 | 766 MB |
| Long | 115 | 45.18 sec | 60.32 sec | **0.75** | 590 ms | 16.7 | 769 MB |

### 1.7B VoiceDesign — CUDA (BF16)

| Text | Words | Wall Clock | Audio Duration | RTF | TTFA | Tok/s | Memory |
|------|-------|------------|----------------|-----|------|-------|--------|
| Short | 13 | 3.01 sec | 4.16 sec | **0.72** | 585 ms | 17.3 | 761 MB |
| Medium | 53 | 14.73 sec | 20.48 sec | **0.72** | 585 ms | 17.4 | 764 MB |
| Long | 115 | 53.78 sec | 71.36 sec | **0.75** | 590 ms | 16.6 | 778 MB |

### CPU (F32, no MKL/BLAS)

| Text | Words | Frames | Wall Clock | Audio Duration | RTF | Tok/s | Memory |
|------|-------|--------|------------|----------------|-----|-------|--------|
| Short | 13 | 47 | 20.28 sec | 3.76 sec | 5.39 | 2.3 | 9.1 GB |
| Medium | 53 | 379 | 182.22 sec | 30.32 sec | 6.01 | 2.1 | 9.1 GB |
| Long | 115 | 703 | 364.17 sec | 56.24 sec | 6.48 | 1.9 | 9.1 GB |

### Summary

| Metric | CPU (1.7B) | 0.6B Base | 1.7B Base | 1.7B CustomVoice | 1.7B VoiceDesign |
|--------|----------:|---------:|---------:|----------------:|----------------:|
| RTF (avg) | 5.96 | **0.60** | 0.73 | 0.73 | 0.73 |
| Tokens/sec | 2.1 | **20.9** | 17.2 | 17.1 | 17.1 |
| TTFA | — | **452ms** | 591ms | 588ms | 587ms |
| Peak memory | 9.1 GB | 841 MB | 771 MB | 769 MB | 778 MB |

**CUDA delivers faster-than-real-time synthesis** across all text lengths.
CPU is ~6x slower than real-time without BLAS acceleration — expected for
a 1.7B parameter model in F32. Enabling MKL (x86) or Accelerate (macOS)
would improve CPU performance significantly.

TTFA (time to first audio) via streaming is stable at ~590ms (1.7B) or ~450ms (0.6B)
regardless of input length, making the streaming API suitable for interactive use cases.

The 0.6B model is ~20% faster than 1.7B variants with lower TTFA, at the cost
of reduced voice quality.

## Micro-Benchmarks

Component-level benchmarks run via [Criterion](https://bheisler.github.io/criterion.rs/book/).
No model weights required.

```
cargo bench
```

### Sampling (codec vocab = 3072)

| Operation | Time |
|-----------|-----:|
| Top-k sampling (k=50) | 53 µs |
| Top-p sampling (p=0.9) | 69 µs |
| Repetition penalty (500 prev tokens) | 834 ns |
| Token suppression | 684 ns |

Top-k with a large text vocab (32k) takes ~556 µs — the codec vocab (3k) keeps
per-step sampling overhead well under 100 µs.

### Audio Processing

| Operation | 0.5s | 2s | 10s |
|-----------|-----:|---:|----:|
| Mel spectrogram | 747 µs | 3.0 ms | 16.2 ms |
| Resample 12kHz → 24kHz | 691 µs | 1.4 ms | 5.4 ms |
| Resample 48kHz → 24kHz | 694 µs | 1.4 ms | 5.5 ms |

### Tensor Operations

| Operation | 1s (12 frames) | 5s (60 frames) | 20s (240 frames) |
|-----------|---------------:|----------------:|------------------:|
| codes_to_tensor | 162 ns | 420 ns | 1.4 µs |

## Reproducing

```bash
# Micro-benchmarks (no model weights needed)
cargo bench

# Single benchmark group
cargo bench -- sampling

# End-to-end (requires model weights)
cargo run --release --features cuda,cli --bin e2e_bench -- \
  --model-dir <path-to-model> --device cuda --iterations 3

# With streaming TTFA measurement and JSON export
cargo run --release --features cuda,cli --bin e2e_bench -- \
  --model-dir <path-to-model> --device cuda --streaming \
  --warmup 2 --json-output results.json

# Audio quality sanity check (optional)
python scripts/quality_check.py output.wav "expected transcription"
```

## Glossary

| Term | Definition |
|------|-----------|
| **RTF** | Real-time factor: wall-clock / audio duration. < 1.0 = faster than real-time. |
| **TTFA** | Time to first audio: latency until the first streaming chunk is available. |
| **Tok/s** | Semantic frames generated per second of wall-clock time. Each frame is one 12 Hz codec step (80ms of audio). |
