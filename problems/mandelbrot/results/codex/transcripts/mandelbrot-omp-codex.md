# Code structural analysis
_Exported on 3/9/2026 at 19:03:38 GMT+1 from Cursor (2.4.31)_

---

**User**

Analyze this code step by step and explain what it does. Identify and analyze:
1. Control flow structure - conditionals and execution order
2. All loops, explain their purpose and iteration space
3. Data dependencies
4. Memory access patterns
5. Variables that may cause race conditions in a shared-memory parallel execution model
Do not propose optimizations, only provide structural analysis.

---

**Cursor**

I’ll inspect `mandelbrot.c` directly and provide a strict structural analysis focused on control flow, loops, dependencies, memory access, and shared-memory race-risk variables.

`mandelbrot.c` is a single-threaded Mandelbrot image generator that computes pixel colors into 2D arrays, then writes an ASCII PPM file.

## 1) Control flow structure (conditionals + execution order)

1. Program enters `main()`.
2. Declares image dimensions (`m=500`, `n=500`), iteration cap (`count_max=2000`), color/count arrays, and coordinate bounds.
3. Calls `timestamp()` and prints startup text.
4. Starts timing (`wtime = clock()/CLOCKS_PER_SEC`).
5. **Compute phase**: nested loops over all pixels `(i,j)`:
   - Map `(i,j)` to complex-plane coordinates `(x,y)`.
   - Initialize iterative state (`x1`, `y1`) and `count[i][j]=0`.
   - Run Mandelbrot recurrence in `k` loop until divergence or max iterations.
   - Color pixel based on parity of `count[i][j]`.
6. Stops timing and prints elapsed time.
7. Opens output file and writes PPM header.
8. **Output phase**: loops over rows and column blocks, writes `r/g/b` values with `fprintf`.
9. Closes file, prints completion messages, calls `timestamp()`, returns `0`.

Conditionals used:
- Divergence check in Mandelbrot iteration:
  - `if (x2 < -2 || x2 > 2 || y2 < -2 || y2 > 2)` then store iteration count and `break`.
- Color branch:
  - odd `count` => white pixel `(255,255,255)`.
  - even `count` => grayscale/blue-toned mapping via nested `sqrt`.
- `i4_min()` helper:
  - standard `if/else` minimum selection for output block upper bound.

---

## 2) All loops: purpose and iteration space

1. `for (i = 0; i < m; i++)`
   - Iteration space: `i in [0, 499]`.
   - Purpose: traverse image rows; compute row coordinate `y`; later write row data.

2. `for (j = 0; j < n; j++)` (inside compute phase)
   - Iteration space: `j in [0, 499]` per row.
   - Purpose: traverse columns; compute column coordinate `x`; compute Mandelbrot escape count and color for pixel `(i,j)`.

3. `for (k = 1; k <= count_max; k++)`
   - Iteration space: `k in [1, 2000]`, but may terminate early via `break`.
   - Purpose: apply recurrence:
     - `x2 = x1^2 - y1^2 + x`
     - `y2 = 2*x1*y1 + y`
     until divergence or max iterations.

4. `for (i = 0; i < m; i++)` (output phase)
   - Iteration space: rows again `i in [0, 499]`.
   - Purpose: emit precomputed color arrays to file.

5. `for (jlo = 0; jlo < n; jlo += 4)`
   - Iteration space: starts `0,4,8,...`.
   - Purpose: output columns in groups of 4 pixels per text line for formatting.
   - `jhi = i4_min(jlo + 4, n)` caps final block at `n`.

6. `for (j = jlo; j < jhi; j++)`
   - Iteration space: up to 4 columns in current block.
   - Purpose: print RGB triplets `r[i][j], g[i][j], b[i][j]`.

---

## 3) Data dependencies

### Per-pixel dependencies
- `count[i][j]` depends on the iterative sequence generated from `(x,y)`.
- `x2,y2` depend on prior iteration values `x1,y1` (**loop-carried dependency in `k` loop**).
- `x1,y1` are updated each iteration from `x2,y2`.
- `r/g/b[i][j]` depend on final `count[i][j]` (parity branch + continuous mapping).

### Cross-loop dependencies
- Output phase depends on compute phase completion:
  - `fprintf` reads `r/g/b` values that were filled earlier.
- Timing output depends on start time assignment before compute loops.

### Independence structure
- Different pixels `(i,j)` are algorithmically independent during compute phase (no neighbor reads/writes).
- No dependency from one row/column pixel to another pixel’s `count` or color.

---

## 4) Memory access patterns

- Arrays `count`, `r`, `g`, `b` are stack-allocated VLAs of shape `[m][n]`.
- Compute phase:
  - Writes `count[i][j]`, then writes `r/g/b[i][j]`.
  - Accesses are row-major friendly because `j` is inner loop and C stores second index contiguously.
  - Good spatial locality within each row for array stores.
- Mandelbrot recurrence uses scalar temporaries (`x1,y1,x2,y2`) in registers/stack, not array-heavy.
- Output phase:
  - Sequential row-major reads of `r/g/b[i][j]`.
  - `fprintf` introduces heavy I/O call overhead but structurally reads contiguous row elements.

---

## 5) Variables that may cause race conditions in shared-memory parallel execution

If this code is parallelized (e.g., OpenMP), race risk depends on variable sharing rules.

### Potential race-prone scalars (if shared)
- Loop indices/temporaries: `i`, `j`, `k`, `jlo`, `jhi`, `c`, `x`, `y`, `x1`, `y1`, `x2`, `y2`.
- These are rewritten every iteration; if threads share them, values can clobber each other.

### Arrays
- `count`, `r`, `g`, `b`:
  - Safe **only if** each thread gets disjoint `(i,j)` indices.
  - Race occurs if multiple threads write same element.

### I/O and global-library state
- `output_unit` / `fprintf` in output loops:
  - Concurrent writes to same `FILE*` can interleave/corrupt output ordering.
- `timestamp()`:
  - Uses static buffer `time_buffer` and `localtime()` (typically returns pointer to static internal storage), both non-thread-safe patterns in many C libraries; concurrent calls can race.

### Non-risk read-only values
- `m`, `n`, `count_max`, `x_min/x_max`, `y_min/y_max`, `output_filename` are initialized once and read-only afterward, so they are not data-race sources by themselves.

If you want, I can also give you a dependence graph view (producer/consumer per variable) in table form.

---

**User**

Based on your analysis of data dependencies, propose a parallelization strategy for this program.
Do not write code.
Explain:
•	which loop(s) can be parallelized,
•	which cannot and why,
•	what the unit of parallel work would be,
•	what kind of parallelism this represents (data, task, pipeline, etc.),
•	and how memory access patterns affect this decision.

---

**Cursor**

A good strategy is to parallelize the **pixel computation phase** and keep file output effectively serial (or logically ordered).

## Parallelizable loops

- **Best target:** outer image loop(s) over pixels:
  - `for i = 0..m-1`
  - inner `for j = 0..n-1`
- Reason: each pixel `(i,j)` is independent of other pixels. Its `count[i][j]`, `r[i][j]`, `g[i][j]`, `b[i][j]` are written exactly once and don’t depend on neighboring pixels.

Practical decomposition choices:
- **Row-based work**: one thread gets one or more full rows (`i` chunks).
- **2D block/tile work**: one thread gets a rectangular tile of pixels.
- **Pixel-level work**: finest grain, but usually too much scheduling overhead unless runtime handles it well.

## Loops that should not be parallelized directly

- **`k` loop (Mandelbrot iteration per pixel)** should stay serial per pixel.
  - It has a loop-carried dependency: `x1,y1` at step `k+1` depend on `x2,y2` from step `k`.
  - Also has early-exit (`break`) on divergence, which is inherently sequential for that pixel’s trajectory.

- **File write loops (`i`, `jlo`, `j` around `fprintf`)** are poor direct parallel targets.
  - Multiple threads writing to one `FILE*` can race/interleave output.
  - Even with synchronization, ordering constraints can dominate and reduce benefit.

## Unit of parallel work

Recommended unit: **a chunk of pixels**, typically:
- **Rows** (coarse grain, cache-friendly in row-major C layout), or
- **Tiles** (better load balancing when escape-time varies spatially).

Each work unit computes:
1. coordinate mapping (`x,y`),
2. serial `k` iterations,
3. final color write to `r/g/b` for that pixel set.

## Parallelism type

This is primarily **data parallelism** (SIMD/SPMD-style map over pixels), and in Mandelbrot specifically it is close to **embarrassingly parallel** for the compute stage.

It is not task parallelism in the sense of heterogeneous tasks; each worker runs the same computation on different data.

## Memory access pattern impact on strategy

- Arrays are row-major (`[i][j]`), and `j` is contiguous in memory.
- Assigning whole rows (or contiguous row segments) to a thread gives:
  - good spatial locality,
  - fewer cache-line ownership conflicts,
  - low risk of false sharing.
- Scattered/interleaved pixel assignment across threads can increase cache traffic and false sharing risk, especially when adjacent pixels on the same cache line are written by different threads.
- Compute phase is memory-light compared to arithmetic-heavy inner recurrence, so parallel efficiency is mainly governed by:
  - balanced work distribution (escape counts vary by region),
  - minimizing synchronization,
  - preserving contiguous write patterns.



---

**User**

Implement the parallelization strategy you proposed using OpenMP. Modify the given code. Explain every OpenMP directive you add.

---

**Cursor**

I’m going to update `mandelbrot.c` to parallelize the compute-phase outer loop with OpenMP while keeping the Mandelbrot inner iteration and file output structure intact. After editing, I’ll do a quick compile check to confirm the directives are valid.

Implemented OpenMP parallelization in `mandelbrot.c` by parallelizing the compute-stage outer loop over image rows, while leaving the per-pixel iteration and output-writing logic structurally unchanged.

```49:52:/Users/saradordevic/Desktop/mandelbrot/mandelbrot.c
#pragma omp parallel for default(none) schedule(dynamic) \
  shared(m,n,count_max,x_max,x_min,y_max,y_min,count,r,g,b) \
  private(i,j,k,c,x,y,x1,y1,x2,y2)
for ( i = 0; i < m; i++ )
```

I also added OpenMP header include:

```1:5:/Users/saradordevic/Desktop/mandelbrot/mandelbrot.c
# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <omp.h>
```

### What each OpenMP directive/clause does

- `#pragma omp parallel for`
  - Creates a team of threads and distributes iterations of the immediately following `for (i...)` loop across threads.
  - This is the core data-parallel mapping: different rows are computed concurrently.

- `default(none)`
  - Forces explicit data-sharing attributes for all variables used in the parallel region.
  - Prevents accidental sharing bugs by requiring every used variable to be classified as `shared` or `private`.

- `schedule(dynamic)`
  - Assigns loop iterations to threads dynamically at runtime.
  - Helps balance load because Mandelbrot rows can have uneven work (different escape iteration counts).

- `shared(m,n,count_max,x_max,x_min,y_max,y_min,count,r,g,b)`
  - These are read-only constants (`m`, `n`, bounds, `count_max`) or shared output arrays (`count`, `r`, `g`, `b`) indexed by distinct `(i,j)` per iteration.
  - Shared arrays are safe here because each pixel element is written by only one loop iteration.

- `private(i,j,k,c,x,y,x1,y1,x2,y2)`
  - Gives each thread its own instance of loop counters and per-pixel temporaries.
  - Prevents races on scalar state used during coordinate mapping and recurrence updates.

### Notes

- I did **not** parallelize the inner `k` loop because it has loop-carried dependency (`x1,y1` depend on prior iteration values).
- I did **not** parallelize file output loops to avoid unsafe/interleaved writes to a single `FILE*`.
- Lint check reports no issues in the edited file.
- Build check on your machine’s default `clang` failed with `unsupported option '-fopenmp'`, which indicates toolchain/runtime setup issue rather than code syntax in the OpenMP directive.

---

**User**

This result gives slower results with more threads than with 1 thread. What could be the issue?

---

**Cursor**

Most likely the main issue is your **timer**, not the parallel loop itself.

- You use `clock()`:
  - `clock()` reports **CPU time consumed by the process**, not wall-clock elapsed time.
  - In multithreaded runs, CPU time is roughly summed across threads, so it can grow with thread count even when real elapsed time drops.
  - That makes it look “slower with more threads” even if parallelism is helping.

Other factors that can also reduce real speedup:

- `schedule(dynamic)` overhead:
  - For `m = 500`, dynamic scheduling may add unnecessary runtime overhead vs `static` because work is distributed in tiny chunks.
- Problem size may be too small:
  - `500x500` with this compute intensity may not amortize OpenMP startup/scheduling costs well.
- Too many threads:
  - If thread count exceeds physical cores (or causes heavy SMT contention), performance can degrade.
- CPU frequency scaling / thermal throttling:
  - Many-core load can reduce per-core turbo frequency, hurting per-thread progress.

So the first thing to verify is: compare using a **wall-clock timer** (e.g., OpenMP wall time) rather than `clock()`. If wall time improves but `clock()` worsens, that confirms measurement artifact.

