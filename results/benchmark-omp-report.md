# OpenMP Benchmark Results

## Experimental Environment

| Parameter | Value |
|-----------|-------|
| **Host OS** | macOS (Docker host) |
| **Container OS** | Ubuntu 22.04.5 LTS |
| **CPU** | Apple M4 |
| **CPU Cores** | 10 |
| **Compiler** | gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0 |
| **Optimization flags** | `-O2` (all implementations) |
| **OpenMP flags** | `-fopenmp` |

## Methodology

- **Repetitions per configuration:** 10
- **Statistics:** arithmetic mean ± standard deviation (seconds)
- **Thread counts tested:** 1 2 4 8 10
- **Timing method:** bash `time` builtin (`real` wall-clock value)
- **Execution:** Each run in a fresh Docker ubuntu:22.04 container

## Problem Parameters

| Problem | Input / Size | Source |
|---------|-------------|--------|
| SGEMM | Medium dataset: 1024×1056 matrices | Parboil benchmark suite |
| Hotspot | 1024×1024 grid, 10000 iterations | Rodinia benchmark suite |
| Mandelbrot | 500×500 image, max 2000 iterations, region [-2.25,1.25]×[-1.75,1.75] | https://people.math.sc.edu/Burkardt/f77_src/mandelbrot_openmp/mandelbrot_openmp.html |
| Moldyn | mm=20 → npart=32000, 20 timesteps | EPCC Training Examples |
| Feynman-Kac | n=10000 paths, 21 grid points, a=2.0 | https://people.math.sc.edu/burkardt/c_src/feynman_kac_1d/feynman_kac_1d.html |

## Accuracy Verification

Parallel output compared against sequential baseline output.

| Problem | Tolerance | Comparison method |
|---------|-----------|-------------------|
| SGEMM | abs ≤ 0.01 or 1% relative | Parboil `tools/compare-output` vs reference matrix3.txt |
| Hotspot | 1e-4 relative | Temperature grid numerical comparison |
| Mandelbrot | exact | PPM image byte-for-byte comparison |
| Moldyn | 1% relative | Energy values (KE, PE, total) per timestep |
| Feynman-Kac | 5% relative | Approximate solution values (stochastic) |


---

## sgemm

**Sequential baseline:** 3.9672s ± 0.0695s

### codex (Accuracy: Pass)

| Threads | Mean (s) | Std Dev (s) | Speedup vs seq |
|---------|----------|-------------|----------------|
| 1 | 6.2498 | 0.3262 | 0.63x |
| 2 | 3.9089 | 0.1563 | 1.01x |
| 4 | 2.8484 | 0.0766 | 1.39x |
| 8 | 2.6103 | 0.1609 | 1.52x |
| 10 | 2.4187 | 0.0398 | 1.64x |

### opus (Accuracy: Pass)

| Threads | Mean (s) | Std Dev (s) | Speedup vs seq |
|---------|----------|-------------|----------------|
| 1 | 6.0854 | 0.0484 | 0.65x |
| 2 | 6.0845 | 0.0740 | 0.65x |
| 4 | 6.1220 | 0.0478 | 0.65x |
| 8 | 6.2611 | 0.2478 | 0.63x |
| 10 | 5.9453 | 0.0845 | 0.67x |

### gemini (Accuracy: Pass)

| Threads | Mean (s) | Std Dev (s) | Speedup vs seq |
|---------|----------|-------------|----------------|
| 1 | 6.3565 | 0.1763 | 0.62x |
| 2 | 4.2423 | 0.1010 | 0.94x |
| 4 | 3.2188 | 0.0374 | 1.23x |
| 8 | 2.5444 | 0.0321 | 1.56x |
| 10 | 2.5902 | 0.0756 | 1.53x |

### reference (Accuracy: Pass)

| Threads | Mean (s) | Std Dev (s) | Speedup vs seq |
|---------|----------|-------------|----------------|
| 1 | 6.5960 | 0.1222 | 0.60x |
| 2 | 4.3892 | 0.3023 | 0.90x |
| 4 | 3.2629 | 0.0473 | 1.22x |
| 8 | 2.5738 | 0.0500 | 1.54x |
| 10 | 2.5925 | 0.1181 | 1.53x |


---

## hotspot

**Sequential baseline:** 17.2331s ± 0.9526s

### codex (Accuracy: Pass)

| Threads | Mean (s) | Std Dev (s) | Speedup vs seq |
|---------|----------|-------------|----------------|
| 1 | 17.3180 | 0.4581 | 1.00x |
| 2 | 12.4130 | 0.4938 | 1.39x |
| 4 | 8.0952 | 0.4251 | 2.13x |
| 8 | 7.2799 | 0.3056 | 2.37x |
| 10 | 11.5962 | 0.6030 | 1.49x |

### opus (Accuracy: Fail)

| Threads | Mean (s) | Std Dev (s) | Speedup vs seq |
|---------|----------|-------------|----------------|
| 1 | 19.6191 | 1.4983 | 0.88x |
| 2 | 13.3148 | 0.8273 | 1.29x |
| 4 | 9.2018 | 0.2758 | 1.87x |
| 8 | 9.0577 | 0.6047 | 1.90x |
| 10 | 12.6516 | 0.3904 | 1.36x |

### gemini (Accuracy: Pass)

| Threads | Mean (s) | Std Dev (s) | Speedup vs seq |
|---------|----------|-------------|----------------|
| 1 | 20.8350 | 1.8584 | 0.83x |
| 2 | 11.4310 | 0.4175 | 1.51x |
| 4 | 7.6832 | 0.1333 | 2.24x |
| 8 | 7.1900 | 0.1785 | 2.40x |
| 10 | 10.9944 | 0.2283 | 1.57x |

### reference (Accuracy: Pass)

| Threads | Mean (s) | Std Dev (s) | Speedup vs seq |
|---------|----------|-------------|----------------|
| 1 | 17.2949 | 0.8738 | 1.00x |
| 2 | 10.4402 | 0.6299 | 1.65x |
| 4 | 6.6791 | 0.2647 | 2.58x |
| 8 | 5.9046 | 0.5911 | 2.92x |
| 10 | 9.1738 | 0.2563 | 1.88x |


---

## mandelbrot

**Sequential baseline:** 0.2109s ± 0.0080s

### codex (Accuracy: Pass)

**Note:** Did not replace `clock()` with `omp_get_wtime()` on its own; only after an extra prompt: *"This result gives slower results with more threads than with 1 thread. What could be the issue?"*

| Threads | Mean (s) | Std Dev (s) | Speedup vs seq |
|---------|----------|-------------|----------------|
| 1 | 0.2185 | 0.0123 | 0.97x |
| 2 | 0.1358 | 0.0033 | 1.55x |
| 4 | 0.1317 | 0.0500 | 1.60x |
| 8 | 0.1019 | 0.0115 | 2.07x |
| 10 | 0.0888 | 0.0074 | 2.38x |

### opus (Accuracy: Pass)

**Note:** Solution did not compile on the first attempt. An extra prompt was needed that only pasted the compiler error:

```
mandelbrot.c: In function 'main':
mandelbrot.c:55:5: error: not enough perfectly nested loops before 'y'
   55 |     y = ( ( double ) (     i - 1 ) * y_max
      |     ^
```

After that prompt, compilation succeeded.

| Threads | Mean (s) | Std Dev (s) | Speedup vs seq |
|---------|----------|-------------|----------------|
| 1 | 0.2225 | 0.0073 | 0.95x |
| 2 | 0.1555 | 0.0139 | 1.36x |
| 4 | 0.1192 | 0.0050 | 1.77x |
| 8 | 0.0950 | 0.0023 | 2.22x |
| 10 | 0.1020 | 0.0146 | 2.07x |

### gemini (Accuracy: Pass)

| Threads | Mean (s) | Std Dev (s) | Speedup vs seq |
|---------|----------|-------------|----------------|
| 1 | 0.2132 | 0.0075 | 0.99x |
| 2 | 0.1380 | 0.0046 | 1.53x |
| 4 | 0.1077 | 0.0166 | 1.96x |
| 8 | 0.0859 | 0.0179 | 2.46x |
| 10 | 0.0796 | 0.0109 | 2.65x |

### reference (Accuracy: Pass)

| Threads | Mean (s) | Std Dev (s) | Speedup vs seq |
|---------|----------|-------------|----------------|
| 1 | 0.2122 | 0.0109 | 0.99x |
| 2 | 0.1385 | 0.0044 | 1.52x |
| 4 | 0.1397 | 0.0042 | 1.51x |
| 8 | 0.1190 | 0.0019 | 1.77x |
| 10 | 0.1143 | 0.0051 | 1.85x |


---

## moldyn

**Sequential baseline:** 19.5271s ± 1.1640s

### codex (Accuracy: Pass)

**Note:** Initial compile failed with:

```
moldyn.c: In function 'forces':
moldyn.c:161:7: error: 'stderr' not specified in enclosing 'parallel'
  161 |       fprintf(stderr, "Unable to allocate thread-local force buffer\n");
      |       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
moldyn.c:153:11: note: enclosing 'parallel'
  153 | #pragma omp parallel default(none)
      |           ^~~
```

Pasting this error as an extra prompt fixed the issue and compilation then succeeded.

| Threads | Mean (s) | Std Dev (s) | Speedup vs seq |
|---------|----------|-------------|----------------|
| 1 | 19.3236 | 0.4625 | 1.01x |
| 2 | 11.6521 | 0.2196 | 1.68x |
| 4 | 7.6886 | 0.4001 | 2.54x |
| 8 | 4.4805 | 0.0962 | 4.36x |
| 10 | 4.2100 | 0.0491 | 4.64x |

### opus (Accuracy: Pass)

**Note:** Did not voluntarily change the timing function to use `omp_get_wtime()`; that change was applied manually.

| Threads | Mean (s) | Std Dev (s) | Speedup vs seq |
|---------|----------|-------------|----------------|
| 1 | 44.9438 | 1.3324 | 0.43x |
| 2 | 26.4036 | 0.6970 | 0.74x |
| 4 | 16.2234 | 0.0947 | 1.20x |
| 8 | 9.4489 | 0.0619 | 2.07x |
| 10 | 9.3262 | 0.1488 | 2.09x |

### gemini (Accuracy: Pass)

| Threads | Mean (s) | Std Dev (s) | Speedup vs seq |
|---------|----------|-------------|----------------|
| 1 | 20.3740 | 0.4475 | 0.96x |
| 2 | 12.3017 | 0.0666 | 1.59x |
| 4 | 8.1750 | 0.0753 | 2.39x |
| 8 | 4.8924 | 0.0635 | 3.99x |
| 10 | 4.6374 | 0.0727 | 4.21x |

### reference (Accuracy: Pass)

| Threads | Mean (s) | Std Dev (s) | Speedup vs seq |
|---------|----------|-------------|----------------|
| 1 | 20.2722 | 0.4280 | 0.96x |
| 2 | 12.1098 | 0.0748 | 1.61x |
| 4 | 8.0517 | 0.0278 | 2.43x |
| 8 | 5.1100 | 0.4285 | 3.82x |
| 10 | 4.7913 | 0.0790 | 4.08x |


---

## feynman-kac

**Sequential baseline:** 26.5620s ± 0.3462s

### codex (Accuracy: Pass)

| Threads | Mean (s) | Std Dev (s) | Speedup vs seq |
|---------|----------|-------------|----------------|
| 1 | 26.3234 | 0.1294 | 1.01x |
| 2 | 15.2661 | 0.2137 | 1.74x |
| 4 | 8.6139 | 0.0357 | 3.08x |
| 8 | 4.5947 | 0.0339 | 5.78x |
| 10 | 4.1946 | 0.0488 | 6.33x |

### opus (Accuracy: Pass)

| Threads | Mean (s) | Std Dev (s) | Speedup vs seq |
|---------|----------|-------------|----------------|
| 1 | 27.0130 | 0.4843 | 0.98x |
| 2 | 15.9552 | 0.2002 | 1.66x |
| 4 | 9.1188 | 0.0385 | 2.91x |
| 8 | 5.0920 | 0.0542 | 5.22x |
| 10 | 5.0461 | 0.1342 | 5.26x |

### gemini (Accuracy: Pass)

| Threads | Mean (s) | Std Dev (s) | Speedup vs seq |
|---------|----------|-------------|----------------|
| 1 | 28.6279 | 0.3089 | 0.93x |
| 2 | 16.6799 | 0.0929 | 1.59x |
| 4 | 9.7118 | 0.1846 | 2.74x |
| 8 | 5.4381 | 0.0245 | 4.88x |
| 10 | 5.2341 | 0.0580 | 5.07x |

