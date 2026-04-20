# Code structural analysis
_Exported on 3/3/2026 at 23:08:57 GMT+1 from Cursor (2.4.31)_

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

Here is a complete structural analysis.

---

## Overview

This program computes the **Mandelbrot set** over a 500x500 grid of complex-plane sample points and writes the result as an ASCII PPM image file. The program has three distinct phases: computation, file output, and two helper functions.

---

## 1. Control Flow Structure

The program's `main` function executes in strict sequential order across three phases:

**Phase 1 — Initialization (lines 14–46):** Variable declarations, stack allocation of four 500x500 integer arrays (`b`, `count`, `g`, `r`), and a timing start via `clock()`.

**Phase 2 — Mandelbrot computation (lines 48–94):** A doubly-nested loop over the 2D grid, with a triply-nested inner escape-time iteration, followed by conditional coloring. This is the computational core.

**Phase 3 — File I/O (lines 103–132):** A doubly-nested loop writes pixel data to a PPM file, followed by cleanup prints and a timestamp.

Within the computational core (Phase 2), the control flow per pixel is:

1. Compute the complex coordinate `(x, y)` for pixel `(i, j)`.
2. Initialize `count[i][j] = 0`, then enter the escape-time loop (up to 2000 iterations).
3. **Conditional break (line 70–74):** If the iterated point escapes the bounding box `[-2, 2] x [-2, 2]`, record the escape iteration count and `break`.
4. **Conditional coloring (lines 79–92):** After the escape loop, branch on whether `count[i][j]` is odd or even:
   - **Odd:** set `r`, `g`, `b` to 255 (white).
   - **Even:** compute a nonlinear grayscale intensity `c` via nested square roots, and assign a blue-tinted color `(3c/5, 3c/5, c)`.

The `i4_min` helper (lines 138–153) is a simple `if/else` returning the minimum of two integers.

---

## 2. All Loops

### Loop A — Outer row loop (line 48)

```48:48:/Users/saradordevic/Desktop/mandelbrot/mandelbrot.c
  for ( i = 0; i < m; i++ )
```

- **Purpose:** Iterates over each row of the image grid.
- **Iteration space:** `i` in `[0, 499]` (500 iterations).
- **Per iteration:** Computes the imaginary component `y` of the complex coordinate for row `i`, then enters Loop B.

### Loop B — Inner column loop (line 54)

```54:54:/Users/saradordevic/Desktop/mandelbrot/mandelbrot.c
    for ( j = 0; j < n; j++ )
```

- **Purpose:** Iterates over each column within the current row.
- **Iteration space:** `j` in `[0, 499]` (500 iterations, 250,000 total with Loop A).
- **Per iteration:** Computes the real component `x`, initializes the iteration seed, enters Loop C, then assigns colors.

### Loop C — Escape-time iteration (line 65)

```65:65:/Users/saradordevic/Desktop/mandelbrot/mandelbrot.c
      for ( k = 1; k <= count_max; k++ )
```

- **Purpose:** Iterates the Mandelbrot recurrence \(z_{n+1} = z_n^2 + c\) to determine whether the point escapes.
- **Iteration space:** `k` in `[1, 2000]` (up to 2000 iterations). Terminates early via `break` if the point escapes the `[-2, 2]` bounding box.
- **Per iteration:** Computes `x2 = x1^2 - y1^2 + x` and `y2 = 2*x1*y1 + y`, tests escape condition, and either breaks or advances `(x1, y1)`.

### Loop D — Output row loop (line 109)

```109:109:/Users/saradordevic/Desktop/mandelbrot/mandelbrot.c
  for ( i = 0; i < m; i++ )
```

- **Purpose:** Iterates over rows for file output.
- **Iteration space:** `i` in `[0, 499]`.

### Loop E — Output column-chunk loop (line 111)

```111:111:/Users/saradordevic/Desktop/mandelbrot/mandelbrot.c
    for ( jlo = 0; jlo < n; jlo = jlo + 4 )
```

- **Purpose:** Breaks each row into chunks of 4 columns for PPM line formatting (so each output line has at most 4 pixel triplets).
- **Iteration space:** `jlo` in `{0, 4, 8, ..., 496}` (125 iterations per row).

### Loop F — Output pixel-within-chunk loop (line 114)

```114:114:/Users/saradordevic/Desktop/mandelbrot/mandelbrot.c
      for ( j = jlo; j < jhi; j++ )
```

- **Purpose:** Writes the R, G, B values for up to 4 pixels per output line.
- **Iteration space:** `j` in `[jlo, jhi)`, where `jhi = min(jlo + 4, n)`. Always 4 iterations except possibly the last chunk if `n` is not divisible by 4 (here `500 % 4 == 0`, so always 4).

---

## 3. Data Dependencies

### Within the escape-time loop (Loop C):

There is a **loop-carried dependency** on `(x1, y1)`. Each iteration depends on the values produced by the previous iteration:

```67:68:/Users/saradordevic/Desktop/mandelbrot/mandelbrot.c
        x2 = x1 * x1 - y1 * y1 + x;
        y2 = 2 * x1 * y1 + y;
```

followed by:

```75:76:/Users/saradordevic/Desktop/mandelbrot/mandelbrot.c
        x1 = x2;
        y1 = y2;
```

This creates a strict sequential chain: iteration `k` cannot begin until iteration `k-1` has completed. The recurrence is inherently serial per pixel.

### Between Loop C and coloring logic:

The coloring block (lines 79–92) depends on `count[i][j]`, which is only finalized after Loop C completes (either via the `break` on line 73 or by remaining 0 if the point never escapes).

### Between pixels (Loop A x Loop B iterations):

Each `(i, j)` pixel computation is **independent** of all other pixels. The values written to `count[i][j]`, `r[i][j]`, `g[i][j]`, `b[i][j]` depend only on the fixed complex coordinate `(x, y)` derived from `(i, j)` and the fixed grid parameters. No pixel reads another pixel's results.

### Between Phase 2 and Phase 3:

Phase 3 (file output) reads `r[i][j]`, `g[i][j]`, `b[i][j]` for all `(i, j)`. It depends on the entire Phase 2 computation being complete.

---

## 4. Memory Access Patterns

### Stack-allocated arrays

Four 500x500 `int` arrays (`r`, `g`, `b`, `count`) are allocated on the stack. At `sizeof(int) == 4`, this totals 4 MB of stack usage, which is notable and may exceed default stack limits on some systems.

### Computation phase (Loops A/B/C):

- `count[i][j]`, `r[i][j]`, `g[i][j]`, `b[i][j]` are written **once** each per `(i, j)` pair.
- Access is **row-major**: the inner loop iterates over `j` (columns), so consecutive iterations access consecutive memory locations within the same row. This is **cache-friendly** for C's row-major array layout.
- The scalar temporaries `x1`, `y1`, `x2`, `y2`, `x`, `y` live in registers or on the stack — no array access for the hot inner loop.

### Output phase (Loops D/E/F):

- `r[i][j]`, `g[i][j]`, `b[i][j]` are **read-only** in row-major order, again cache-friendly.
- File I/O via `fprintf` dominates this phase.

### Spatial locality summary:

All array accesses exhibit strong spatial locality. There is no strided or random access. Temporal locality is limited — each element is written once and read once (during output).

---

## 5. Race Condition Analysis for Shared-Memory Parallelism

If the outer loops (A and B) were parallelized (e.g., via OpenMP, as the output filename `mandelbrot_openmp.ppm` suggests was intended):

### Safe (no race) — Array accesses:

`count[i][j]`, `r[i][j]`, `g[i][j]`, `b[i][j]` are indexed by the loop variables `(i, j)`. Each thread operating on a distinct `(i, j)` pair writes to a unique memory location. **No races on these arrays.**

### Unsafe — Shared scalar temporaries:

The following variables are declared in `main`'s scope and would be **shared** by default in an OpenMP parallel region:

| Variable | Issue |
|----------|-------|
| **`x`** | Written on line 56, read on lines 62, 67, 68. Each thread needs its own value derived from its own `j`. **Race condition.** |
| **`y`** | Written on line 50, read on lines 63, 68. Each thread needs its own value derived from its own `i`. **Race condition.** |
| **`x1`** | Written on lines 62, 75. Loop-carried dependency within Loop C. **Race condition.** |
| **`y1`** | Written on lines 63, 76. Loop-carried dependency within Loop C. **Race condition.** |
| **`x2`** | Written on line 67, read on lines 70, 75. **Race condition.** |
| **`y2`** | Written on line 68, read on lines 70, 76. **Race condition.** |
| **`k`** | Loop induction variable for Loop C. **Race condition.** |
| **`j`** | If the outer loop on `i` is parallelized, `j` is used as the inner loop induction variable. **Race condition.** |
| **`c`** | Written and read in the even-count branch (line 87–91). **Race condition.** |

All of these would need to be declared **private** to each thread in a parallel context.

### Safe scalars:

- `i` — if used as the parallelized loop index, it is implicitly private.
- `m`, `n`, `count_max`, `x_max`, `x_min`, `y_max`, `y_min` — read-only constants. No write after initialization. **No race.**

### Phase 3 (file output):

The output phase performs sequential `fprintf` calls in a specific pixel order. This is **inherently serial** — parallelizing it would produce garbled or interleaved output. It must execute after all of Phase 2 has completed (an implicit barrier).

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

## Parallelization Strategy

### Which loops can be parallelized

**Loop A (outer row loop, line 48) and Loop B (inner column loop, line 54) — both are parallelizable.**

As established in the dependency analysis, every pixel `(i, j)` is computed independently. The complex coordinate is derived solely from `(i, j)` and the fixed grid parameters. The results are written to unique locations `count[i][j]`, `r[i][j]`, `g[i][j]`, `b[i][j]`. There are no cross-pixel data dependencies. This makes the full `(i, j)` iteration space — all 250,000 pixels — embarrassingly parallel.

You can parallelize at either level:

- **Parallelizing Loop A alone** (the outer loop over rows): Each thread gets a subset of rows. This is the simplest and most practical approach. The inner loop over columns and the escape-time loop remain serial within each thread. This yields up to 500 independent units of work.

- **Parallelizing both Loop A and Loop B** (collapsing the two loops into a single flattened iteration space of 250,000 pixels): This increases the available parallelism by a factor of 500 and improves load balancing, since the amount of work per row is non-uniform (some rows contain more points near the set boundary, which require more escape iterations).

### Which loops cannot be parallelized

**Loop C (escape-time iteration, line 65) — cannot be parallelized.**

This loop carries a strict recurrence dependency:

- `x2` at iteration `k` depends on `x1` and `y1`, which are the outputs of iteration `k-1`.
- `y2` at iteration `k` depends on `x1` and `y1` from iteration `k-1`.
- The early exit (`break` on escape) depends on the values computed in that same iteration.

Each step of the Mandelbrot recurrence \(z_{n+1} = z_n^2 + c\) is inherently sequential. There is no known algebraic shortcut to compute iteration `k` without first computing iterations `1` through `k-1`. This loop must remain serial.

**Loops D/E/F (file output, lines 109–119) — cannot be parallelized.**

The output phase performs ordered sequential writes to a file stream via `fprintf`. PPM is a format where pixel order is semantically meaningful — row 0 must be written before row 1, and so on. Parallel writes would produce interleaved or out-of-order output, corrupting the file. This phase is inherently serial, and must also execute strictly after all of Phase 2 (an implicit barrier between computation and output).

### Unit of parallel work

The natural unit of parallel work is **one pixel** — a single `(i, j)` pair. Each pixel involves:

1. Computing the complex coordinate (trivial arithmetic),
2. Running the escape-time loop (1 to 2000 iterations of floating-point arithmetic),
3. Assigning the RGB color values.

In practice, the granularity decision is between:

- **One row** (500 pixels): simpler to express, lower scheduling overhead, 500-way parallelism.
- **One pixel** (1 pixel): maximum parallelism (250,000-way), better load balance, but higher scheduling overhead.

The row-level granularity is likely too coarse for good load balancing because of the **irregular workload**: pixels near the Mandelbrot set boundary require close to 2000 escape iterations, while pixels far from the boundary escape in fewer than 10. Rows passing through the boundary are far more expensive than rows in the exterior. Collapsing to per-pixel granularity with dynamic scheduling (or using chunked dynamic scheduling at the row level) addresses this imbalance.

### Type of parallelism

This is **data parallelism** — specifically, the SPMD (Single Program, Multiple Data) variant.

- The same computation (escape-time iteration + coloring) is applied independently to every element of a 2D data domain.
- There is no inter-element communication during the computation phase.
- There is no task heterogeneity — every thread runs the same code on different data.
- There is no pipeline structure — Phase 2 must fully complete before Phase 3 begins; there is no streaming overlap between computation and output.

It is not task parallelism (all tasks are identical), and it is not pipeline parallelism (there is no producer-consumer chain between stages that could overlap in time).

### How memory access patterns affect this decision

**Favoring row-level parallelization:** C stores 2D arrays in row-major order. When a thread processes row `i`, it writes sequentially through `count[i][0..499]`, `r[i][0..499]`, `g[i][0..499]`, `b[i][0..499]`. These are contiguous memory regions. Assigning entire rows (or contiguous blocks of rows) to threads ensures that each thread operates on a contiguous memory region, maximizing spatial locality and L1/L2 cache utilization.

**False sharing considerations:** If parallelizing at the pixel level with fine granularity, adjacent pixels in the same row may reside on the same cache line (a 64-byte cache line holds 16 `int` values, covering 16 adjacent columns). If two threads work on adjacent pixels in the same row, they write to the same cache line, causing false sharing — the cache line bounces between cores even though the threads write to distinct addresses. Row-level distribution eliminates this entirely, since each thread's write region is row-aligned and spatially separated. Collapsing to pixel-level parallelism is still safe if the scheduler assigns contiguous chunks of pixels rather than interleaving individual pixels across threads.

**Read-only data stays local:** The grid parameters (`x_min`, `x_max`, `y_min`, `y_max`, `m`, `n`, `count_max`) are read-only after initialization. They will replicate into each core's cache and remain there without invalidation.

**Stack allocation concern:** The four 500x500 `int` arrays total approximately 4 MB on the stack. In a threaded execution model, each thread has its own stack, but these arrays are in the main thread's stack and shared across threads. This is fine for shared-memory parallelism (all threads can access the main thread's stack), but the total footprint must fit in the main thread's stack limit. This is a deployment concern rather than a correctness issue, but it is relevant to the parallelization decision: heap allocation would be more robust for larger problem sizes.

**Output phase is memory-bound anyway:** Phase 3 reads the arrays sequentially in row-major order and writes to a file. It is I/O-dominated. Even if the arrays are cold in cache after Phase 2 (if different threads' caches have different rows), the sequential read pattern will prefetch effectively. There is no benefit to parallelizing this phase.

### Summary

| Aspect | Decision |
|--------|----------|
| Parallelizable loops | Loop A (rows) and Loop B (columns) — both, independently or collapsed |
| Non-parallelizable loops | Loop C (recurrence dependency), Loops D/E/F (ordered I/O) |
| Work unit | One pixel `(i, j)`, or one row `i` for lower overhead |
| Parallelism type | Data parallelism (SPMD) |
| Scheduling | Dynamic (or guided) to handle irregular per-pixel workload |
| Memory implication | Row-level distribution aligns with row-major layout and avoids false sharing |

---

**User**

Implement the parallelization strategy you proposed using OpenMP. Modify the given code. Explain every OpenMP directive you add.

---

**Cursor**

I'm currently in **Ask mode**, which means I can only provide information and analysis — I cannot edit files. To have me implement the changes directly, switch to **Agent mode** (you can do this from the mode selector in the chat input).

That said, here is the exact implementation plan with every directive explained, so you can either apply it yourself or let me do it once you switch modes.

---

## Required Changes

### 1. Add the OpenMP header

At the top, alongside the other includes, add:

```c
#include <omp.h>
```

This provides the OpenMP runtime API, specifically needed for `omp_get_wtime()`.

### 2. Replace the timing mechanism

The current code uses `clock()`, which measures **CPU time** (summed across all threads). In a parallel program, this makes the elapsed time appear *longer* with more threads. Replace it with OpenMP's wall-clock timer.

Replace line 46:

```c
wtime = omp_get_wtime();
```

Replace line 97:

```c
wtime = omp_get_wtime() - wtime;
```

`omp_get_wtime()` returns wall-clock seconds as a `double`. The difference gives true elapsed time, which is what you want for measuring parallel speedup.

### 3. Add the parallel directive on the computation loop

Immediately before the outer `for` loop on line 48, add:

```c
#pragma omp parallel for collapse(2) schedule(dynamic, 1) \
    default(none) \
    shared(count, r, g, b, m, n, count_max, x_max, x_min, y_max, y_min) \
    private(i, j, k, x, y, x1, y1, x2, y2, c)
```

Here is what each clause does:

**`#pragma omp parallel for`** — This is a combined directive. `parallel` creates a team of threads. `for` distributes the iterations of the immediately following `for` loop across those threads. The implicit barrier at the end ensures all threads finish before execution continues to Phase 3.

**`collapse(2)`** — Fuses Loop A (`i`) and Loop B (`j`) into a single iteration space of 250,000 pixels. Without this, only the 500 row iterations would be distributed. Collapsing both loops is critical here for two reasons: (a) it exposes 500x more parallelism, and (b) it allows the scheduler to interleave rows, balancing the workload since rows near the set boundary are far more expensive.

**`schedule(dynamic, 1)`** — Assigns iterations to threads dynamically, one pixel at a time. When a thread finishes one pixel, it grabs the next available one from the shared work queue. This handles the **irregular workload**: a pixel deep in the Mandelbrot set interior runs all 2000 escape iterations, while a pixel far in the exterior escapes in under 10. Static scheduling would leave some threads idle while others grind through expensive boundary pixels. A chunk size of 1 maximizes balance at the cost of some scheduling overhead; in practice a small chunk size like `(dynamic, 4)` or `(dynamic, 8)` would reduce overhead while retaining good balance — but `1` is the simplest correct choice.

**`default(none)`** — Forces every variable to be explicitly classified as `shared` or `private`. This is a safety discipline: if you forget a variable, the compiler errors out rather than silently sharing it and creating a race condition.

**`shared(count, r, g, b, m, n, count_max, x_max, x_min, y_max, y_min)`** — These variables are shared across all threads:
- `count`, `r`, `g`, `b` — the output arrays. Safe to share because each thread writes to a unique `[i][j]` location.
- `m`, `n`, `count_max`, `x_max`, `x_min`, `y_max`, `y_min` — read-only constants. No write after initialization, so no race.

**`private(i, j, k, x, y, x1, y1, x2, y2, c)`** — Each thread gets its own private copy of these variables. This is mandatory because:
- `i`, `j` — loop induction variables for the parallelized loops. Each thread works on different `(i, j)` values simultaneously.
- `k` — loop induction variable for the escape-time loop. Each thread runs its own independent escape iteration.
- `x`, `y` — the complex coordinate derived from `(i, j)`. Different threads process different pixels, so they need different coordinates.
- `x1`, `y1`, `x2`, `y2` — the recurrence temporaries. These carry the loop dependency within a single pixel's escape iteration and must not be shared.
- `c` — the color intensity, computed per-pixel in the even-count branch.

Without privatization, all nine of these variables would be race conditions, as identified in the earlier analysis.

### 4. No changes to Phase 3

The file output loops (lines 109–120) remain serial. They perform ordered sequential `fprintf` calls that must produce pixels in row-major order. Parallelizing I/O here would corrupt the PPM file.

---

## Summary of the complete modified computation section

The modified section would look like this (showing only the changed region):

```c
wtime = omp_get_wtime();

#pragma omp parallel for collapse(2) schedule(dynamic, 1) \
    default(none) \
    shared(count, r, g, b, m, n, count_max, x_max, x_min, y_max, y_min) \
    private(i, j, k, x, y, x1, y1, x2, y2, c)
  for ( i = 0; i < m; i++ )
  {
    /* ... body unchanged ... */
  }

wtime = omp_get_wtime() - wtime;
```

### Compilation

The program must be compiled with OpenMP enabled:

```bash
gcc -fopenmp -O2 -lm -o mandelbrot mandelbrot.c
```

`-fopenmp` enables the OpenMP directives and links the runtime. Without it, the `#pragma` lines are silently ignored and the program runs serially.

---

Switch to **Agent mode** and I will apply all of these changes to your file.

---

**User**

Implement the parallelization strategy you proposed using OpenMP. Modify the given code. Explain every OpenMP directive you add.

---

**Cursor**

Let me re-read the file to confirm the current state, then apply the changes.

Now I'll apply the four changes: add the OpenMP header, replace the timing calls, and insert the parallel directive.

Let me verify the final state of the file.

The file is correct. Here is a summary of every change and why.

---

## Changes made

### 1. `#include <omp.h>` (line 5)

Provides the OpenMP runtime API. Required for `omp_get_wtime()` and for the compiler to resolve the `#pragma omp` directives.

### 2. `omp_get_wtime()` (lines 47 and 102)

Replaced `clock() / CLOCKS_PER_SEC` with `omp_get_wtime()`. The original `clock()` measures **total CPU time across all threads**, which increases with thread count and gives a misleading "slowdown." `omp_get_wtime()` measures **wall-clock time** — the actual elapsed time the user experiences — which is the correct metric for evaluating parallel speedup.

### 3. The parallel directive (lines 49–52)

```49:52:/Users/saradordevic/Desktop/mandelbrot/mandelbrot.c
# pragma omp parallel for collapse(2) schedule(dynamic, 1) \
    default(none) \
    shared(count, r, g, b, m, n, count_max, x_max, x_min, y_max, y_min) \
    private(i, j, k, x, y, x1, y1, x2, y2, c)
```

Each clause:

**`parallel for`** — Combined directive that creates a thread team *and* distributes the loop iterations across those threads. An implicit barrier at the closing brace of the loop ensures all threads complete before execution proceeds to the file output phase.

**`collapse(2)`** — Fuses the `i` and `j` loops into a single iteration space of 250,000 pixels. Without this, only 500 row iterations would be distributed, limiting parallelism and causing poor load balance (rows near the set boundary are much more expensive). Collapsing exposes the full pixel-level parallelism.

**`schedule(dynamic, 1)`** — Iterations are assigned to threads on demand, one at a time. This is essential because the per-pixel workload is highly irregular: exterior pixels escape in <10 iterations while boundary pixels may run all 2000. Static scheduling would leave fast-finishing threads idle. Dynamic scheduling ensures all threads stay busy.

**`default(none)`** — Every variable must be explicitly listed as `shared` or `private`. This is a safety net: if a variable is accidentally omitted, the compiler emits an error rather than silently sharing it and creating a race condition.

**`shared(count, r, g, b, m, n, count_max, x_max, x_min, y_max, y_min)`** — These are safely shared:
- `count`, `r`, `g`, `b` — output arrays. Each thread writes to a unique `[i][j]` element, so no two threads ever write to the same location.
- `m`, `n`, `count_max`, `x_max`, `x_min`, `y_max`, `y_min` — read-only constants after initialization. Concurrent reads with no writes are always safe.

**`private(i, j, k, x, y, x1, y1, x2, y2, c)`** — Each thread gets its own independent copy. These are the nine variables identified in the race condition analysis:
- `i`, `j` — parallelized loop indices. Each thread works on different pixels.
- `k` — the escape-time loop counter. Each thread runs its own independent escape iteration.
- `x`, `y` — complex-plane coordinates derived from `(i, j)`. Different threads process different pixels.
- `x1`, `y1`, `x2`, `y2` — recurrence temporaries for the escape iteration. Each thread iterates its own sequence.
- `c` — color intensity computed per pixel.

### What was *not* changed

The file output phase (lines 114–125) remains serial. It performs ordered sequential `fprintf` calls that must emit pixels in exact row-major order to produce a valid PPM file. Parallelizing this would corrupt the output.

### Compilation

To build with OpenMP enabled:

```bash
gcc -fopenmp -O2 -lm -o mandelbrot mandelbrot.c
```

