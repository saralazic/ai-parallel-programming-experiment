# CUDA Benchmark Results

## Experimental Environment

| Parameter | Value |
|-----------|-------|
| **OS** | Ubuntu 24.04.3 LTS |
| **CPU** | 11th Gen Intel(R) Core(TM) i7-11700K @ 3.60GHz |
| **GPU** | NVIDIA GeForce RTX 3080 Ti |
| **CUDA Toolkit** | 12.0V12.0.140 |
| **NVCC flags** | `-O2 -arch=sm_86` |
| **GCC (sequential)** | gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 |
| **GCC flags** | `-O2` |

## Methodology

- **Repetitions per configuration:** 10
- **Statistics:** arithmetic mean ± standard deviation (seconds)
- **Timing:** wall-clock time (`real` from bash `time` builtin) for the entire program execution
- **GPU architecture target:** sm_86 (compute capability 8.6)

### What CUDA timing includes

The measured time is **end-to-end wall-clock** time, which includes:
- Device memory allocation (`cudaMalloc`)
- Host-to-device memory transfers (`cudaMemcpy H→D`)
- Kernel execution (GPU computation)
- Device-to-host memory transfers (`cudaMemcpy D→H`)
- Device memory deallocation (`cudaFree`)
- Any CPU-side setup within the program

This gives a fair total-cost comparison with the sequential CPU baseline.

## Problem Parameters

| Problem | Input / Size | Source |
|---------|-------------|--------|
| SGEMM | Medium dataset: 1024×1056 matrices | Parboil benchmark suite |
| Hotspot | 1024×1024 grid, 10000 iterations | Rodinia benchmark suite |
| Mandelbrot | 500×500 image, max 2000 iterations, region [-2.25,1.25]×[-1.75,1.75] | https://people.math.sc.edu/Burkardt/f77_src/mandelbrot_openmp/mandelbrot_openmp.html |
| Moldyn | mm=20 → npart=32000, 20 timesteps | EPCC Training Examples |
| Feynman-Kac | n=10000 paths, 21 grid points, a=2.0 | https://people.math.sc.edu/burkardt/c_src/feynman_kac_1d/feynman_kac_1d.html |

## Accuracy Verification

CUDA output compared against a CPU baseline (see notes below).

| Problem | Tolerance | Comparison method |
|---------|-----------|-------------------|
| SGEMM | abs ≤ 0.01 or 1% relative | Output matrix vs sequential (Parboil-like) |
| Hotspot (AI models) | abs ≤ 0.1°C or 1% relative | Temperature grid vs educational sequential |
| Hotspot (Rodinia reference) | N/A | Not numerically comparable (see note) |
| Mandelbrot | ≥99% RGB match | PPM pixels (P3/P6); allows GPU boundary FP diffs |
| Moldyn | 1% relative | Energy values (KE, PE, total) per timestep |
| Feynman-Kac | 5% relative | Approximate solution values (stochastic) |

**Note — Hotspot Rodinia CUDA reference:** Accuracy is reported as **N/A**, not
Fail. Upstream Rodinia CUDA computes the integration step as
`PRECISION/max_slope`, while Rodinia OpenMP and the educational sequential code
use `PRECISION/max_slope/1000.0`. That 1000× timestep difference makes CUDA↔OMP
(and CUDA↔seq) temperature grids diverge by design; Rodinia ships no golden-file
checker for Hotspot. Timing still uses the stock Rodinia CUDA binary. AI Hotspot
solutions keep the `/1000.0` formulation and are validated against sequential.


---

## sgemm

**Sequential baseline (CPU):** 2.0728s ± 0.1366s

| Model | Accuracy | Mean (s) | Std Dev (s) | Speedup vs seq |
|-------|----------|----------|-------------|----------------|
| codex | Pass | 0.6672 | 0.0248 | 3.11x |
| opus | Pass | 0.6628 | 0.0103 | 3.13x |
| gemini | Pass | 0.6902 | 0.0192 | 3.00x |
| reference_base | Pass | 0.6634 | 0.0088 | 3.12x |
| reference_optimized | Pass | 0.6682 | 0.0249 | 3.10x |


---

## hotspot

**Sequential baseline (CPU):** 16.2238s ± 0.1297s

| Model | Accuracy | Mean (s) | Std Dev (s) | Speedup vs seq |
|-------|----------|----------|-------------|----------------|
| codex | Pass | 0.5723 | 0.0313 | 28.35x |
| opus | Pass | 0.5672 | 0.0021 | 28.60x |
| gemini | Pass | 0.5651 | 0.0032 | 28.71x |
| reference | N/A | 1.2426 | 0.0055 | 13.06x |

**Note (codex):** COMPILATION FAILED — the same solution was inserted twice into the file. The duplicate half was deleted manually; after that, compilation succeeded.


---

## mandelbrot

**Sequential baseline (CPU):** 0.1868s ± 0.0008s

| Model | Accuracy | Mean (s) | Std Dev (s) | Speedup vs seq |
|-------|----------|----------|-------------|----------------|
| codex | Pass | 0.1620 | 0.0133 | 1.15x |
| opus | Pass | 0.1764 | 0.0024 | 1.06x |
| gemini | Pass | 0.1758 | 0.0042 | 1.06x |
| reference | Pass | 0.1746 | 0.0018 | 1.07x |


---

## moldyn

**Sequential baseline (CPU):** 28.3633s ± 0.0302s

| Model | Accuracy | Mean (s) | Std Dev (s) | Speedup vs seq |
|-------|----------|----------|-------------|----------------|
| codex | Pass | 2.1867 | 0.0328 | 12.97x |
| opus | Pass | 2.1967 | 0.0045 | 12.91x |
| gemini | Pass | 2.1513 | 0.0027 | 13.18x |


---

## feynman-kac

**Sequential baseline (CPU):** 24.9184s ± 0.0230s

| Model | Accuracy | Mean (s) | Std Dev (s) | Speedup vs seq |
|-------|----------|----------|-------------|----------------|
| codex | Pass | 4.7788 | 0.1047 | 5.21x |
| opus | Pass | 4.0151 | 0.0617 | 6.21x |
| gemini | Pass | 29.5755 | 0.2263 | 0.84x |

