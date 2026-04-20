# Code structural analysis
_Exported on 3/28/2026 at 13:59:16 GMT+1 from Cursor (2.4.31)_

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

## Structural Analysis of `feynman-kac.c`

This program is a 1D Feynman-Kac solver. It uses Monte Carlo random walks to numerically estimate the solution to a boundary value problem on the interval \([-a, a]\), then compares the estimate against a known exact solution.

---

### 1. Control Flow Structure

The program has a clear top-down flow with three levels of nesting:

**Top-level sequence (lines 43–193):**
1. Print configuration/banner
2. Initialize constants (`ni`, `rth`, `err`, `k`, `n_int`)
3. Enter the main spatial loop (line 88)
4. After the loop: compute RMS error, print summary, terminate

**Inside the spatial loop** (lines 88–176), there is a conditional branch at line 98:

```98:106:/Users/saradordevic/Desktop/feynman-kac/feynman-kac.c
    if ( test < 0.0 )
    {
      w_exact = 1.0;
      wt = 1.0;
      steps_ave = 0;
      printf ( "  %4d  %4d  %12g  %12g  %12g  %12g  %8d  %8g\n",
        i, k, x, w_exact, wt, fabs ( w_exact - wt ), steps_ave, test );
      continue;
    }
```

- When `test < 0.0`, the point `x` is **outside** the domain. The solution is trivially 1.0; the iteration prints results and `continue`s, skipping all Monte Carlo work.
- When `test >= 0.0`, the point is **on or inside** the domain. Execution falls through to the Monte Carlo estimation block (lines 108–175).

**Inside the Monte Carlo block**, there is a nested conditional at line 138:

```138:145:/Users/saradordevic/Desktop/feynman-kac/feynman-kac.c
        if ( us < 0.0 )
        {
          dx = - rth;
        }
        else
        {
          dx = + rth;
        }
```

This implements a symmetric random walk: a uniform random number determines whether the walker steps left (`-rth`) or right (`+rth`).

**In `r8_uniform_01`** (line 216), there is a conditional to keep the seed positive:

```216:219:/Users/saradordevic/Desktop/feynman-kac/feynman-kac.c
  if ( *seed < 0 )
  {
    *seed = *seed + 2147483647;
  }
```

This wraps the seed modulo \(2^{31} - 1\), which is the modulus of the linear congruential generator (a Park-Miller style PRNG).

**Overall execution order:**
`main` entry → banner → spatial loop { boundary check → [exact solution → trial loop { random walk loop }] → accumulate error } → RMS error → exit.

---

### 2. All Loops

There are exactly **three loops**, with two nested inside the first:

#### Loop 1: Spatial discretization loop (outermost)

```88:88:/Users/saradordevic/Desktop/feynman-kac/feynman-kac.c
  for ( i = 0; i <= ni + 1; i++ )
```

- **Purpose:** Iterates over each grid point in the spatial domain. It visits `ni + 2 = 23` points spanning from slightly beyond \(-a\) to slightly beyond \(+a\).
- **Iteration space:** `i = 0, 1, 2, ..., 22` (23 iterations).
- **Grid coordinate formula** (lines 90–92): `x` is computed as a linear interpolation. When `i = 0`, `x` is slightly below \(-a\); when `i = 22`, `x` is slightly above \(+a\). The boundary points at the extremes have `test < 0` and are handled by the early-exit branch.

#### Loop 2: Monte Carlo trial loop (middle)

```119:163:/Users/saradordevic/Desktop/feynman-kac/feynman-kac.c
    for ( it = 1; it <= n; it++ )
    {
      // ... random walk ...
      wt = wt + w;
    }
```

- **Purpose:** Runs `n = 10,000` independent random walk trials from the same starting point `x`. Each trial simulates a Brownian path and accumulates a weighted functional along it. The results are averaged to produce the Monte Carlo estimate.
- **Iteration space:** `it = 1, 2, ..., 10000` (10,000 iterations).
- **Only entered** for interior/boundary points (when `test >= 0.0`).

#### Loop 3: Random walk loop (innermost)

```132:161:/Users/saradordevic/Desktop/feynman-kac/feynman-kac.c
      while ( chk < 1.0 )
      {
        us = r8_uniform_01 ( &seed ) - 0.5;
        // ... step and accumulate ...
        chk = pow ( x1 / a, 2 );
      }
```

- **Purpose:** Simulates a single random walk trajectory starting from `x1 = x`. At each step it moves left or right by `rth = sqrt(h)`, evaluates the potential, and updates the path weight `w` using a trapezoidal-like discretization of the Feynman-Kac formula. The walk terminates when the particle leaves the interval, i.e., when \((x_1/a)^2 \geq 1.0\).
- **Iteration space:** Variable. The number of steps depends on how quickly the random walker reaches the boundary. For points near the center of the interval, the expected number of steps is large; for points near the boundary, it is small. The step count is accumulated in `steps`.
- **Termination condition:** `chk >= 1.0`, meaning `|x1| >= |a|`.

---

### 3. Data Dependencies

#### Within the spatial loop (across iterations of `i`):

| Variable | Dependency Type |
|---|---|
| `k` | **Loop-carried.** Incremented by 1 each iteration: `k = k + 1`. |
| `n_int` | **Loop-carried.** Conditionally incremented: `n_int = n_int + 1` (only for interior points). |
| `err` | **Loop-carried.** Accumulated across iterations: `err = err + pow(w_exact - wt, 2)`. |
| `seed` | **Loop-carried.** Mutated by every call to `r8_uniform_01(&seed)`. The PRNG state from one spatial point feeds into the next. |
| `x`, `test`, `w_exact`, `wt`, `steps_ave` | **Loop-private.** Recomputed from scratch each iteration; no dependency on prior `i`. |

#### Within the trial loop (across iterations of `it`):

| Variable | Dependency Type |
|---|---|
| `wt` | **Loop-carried.** Accumulated: `wt = wt + w`. |
| `steps` | **Loop-carried.** Accumulated: `steps = steps + 1` (inside the while loop, but the total spans across all trials for a given point). |
| `seed` | **Loop-carried.** Each call to `r8_uniform_01` mutates `seed`, and the next trial begins with the seed left by the previous trial. |
| `x1`, `w`, `chk`, `dx`, `us`, `vs`, `vh`, `we` | **Loop-private.** Reset at the start of each trial iteration. |

#### Within the random walk loop (across steps):

| Variable | Dependency Type |
|---|---|
| `x1` | **Loop-carried.** `x1 = x1 + dx` — the walker's position is updated each step. |
| `w` | **Loop-carried.** Updated via `w = w - 0.5 * h * (vh * we + vs * w)`, where `we` itself depends on the prior value of `w`. |
| `seed` | **Loop-carried.** Mutated each step by `r8_uniform_01`. |
| `chk` | **Loop-carried (termination).** Recomputed from `x1` each step; controls loop exit. |
| `steps` | **Loop-carried.** Incremented each step. |
| `us`, `dx`, `vs`, `vh`, `we` | **Step-private.** Recomputed each step, though `vs` at step \(k\) uses `x1` from step \(k-1\) and `vh` uses `x1` from step \(k\). |

#### Inter-function dependencies:

- `potential(a, x1)` is a **pure function** — it reads `a` and `x1`, computes a value, and has no side effects.
- `r8_uniform_01(&seed)` is **stateful** — it reads and writes `*seed`. Every call depends on the result of the previous call.

---

### 4. Memory Access Patterns

This program operates almost entirely in **scalar registers/stack variables**. There are no heap allocations, no arrays used for computation, and no dynamic data structures.

- **All variables in `main`** are local scalars on the stack. Access is strictly sequential and trivially cache-friendly (they likely all reside in registers or L1 cache).
- **`r8_uniform_01`** receives a pointer to `seed` (a stack variable in `main`). This is an indirect write to a single integer — effectively a register-to-register operation after inlining.
- **`potential`** takes two doubles by value and returns a double. No memory side effects.
- **`timestamp`** uses a `static char` buffer (`time_buffer[40]`). This is the only piece of static/global mutable state in the entire program. It also calls `time()` and `localtime()`, which access system state.
- **I/O:** `printf` calls are scattered throughout. These write to `stdout`, which involves buffered I/O (library-internal state).

There are **no stride patterns, no array indexing, and no indirect (pointer-chasing) memory accesses** in the computational core. The entire computational hot path (the three nested loops) is purely scalar.

---

### 5. Variables at Risk of Race Conditions Under Shared-Memory Parallelism

If one were to parallelize this code (e.g., with OpenMP), the following variables would be sources of **data races** depending on which loop is parallelized:

#### If parallelizing the **outer spatial loop** (`for i`):

| Variable | Issue |
|---|---|
| `seed` | **Critical.** The PRNG is a single sequential state machine. Every iteration reads and writes `seed`. Sharing it across threads produces correlated or corrupted random numbers and a data race on every PRNG call. |
| `err` | **Reduction variable.** Multiple iterations write `err = err + ...`. Concurrent read-modify-write is a race. |
| `n_int` | **Reduction variable.** Conditionally incremented. Same concurrent accumulation issue. |
| `k` | **Sequentially incremented.** Although `k` simply mirrors `i + 1` (it's incremented by 1 each iteration), it is written as a loop-carried dependency. Concurrent writes are a race. However, `k` could be trivially recomputed from `i`. |
| `wt`, `steps`, `steps_ave`, `w_exact`, `x`, `x1`, `w`, `chk`, `test`, `dx`, `us`, `vs`, `vh`, `we` | Would need to be **private** to each thread (they are recomputed per iteration). If not declared private, concurrent writes from different `i` values would collide. |

#### If parallelizing the **trial loop** (`for it`) for a given spatial point:

| Variable | Issue |
|---|---|
| `seed` | **Critical** — same issue as above. Each trial calls `r8_uniform_01` multiple times. |
| `wt` | **Reduction variable.** Accumulated across trials: `wt = wt + w`. |
| `steps` | **Reduction variable.** Accumulated across trials (via the inner while loop). |
| `x1`, `w`, `chk`, `dx`, `us`, `vs`, `vh`, `we` | Would need to be **thread-private**. |

#### If parallelizing the **innermost while loop** (random walk steps):

This loop is **not parallelizable** due to tight sequential dependencies: `x1`, `w`, `chk`, and `seed` are all loop-carried with step-to-step data dependencies. There is no independent work to distribute.

#### Summary of race-prone variables by severity:

| Severity | Variables | Nature |
|---|---|---|
| **High** | `seed` | Sequential PRNG state; cannot be shared without producing incorrect results even if the race is "resolved" by a lock, because the statistical properties of the sequence depend on serial ordering. |
| **Medium** | `err`, `wt`, `steps`, `n_int` | Accumulation variables; addressable via reduction clauses, but require explicit handling. |
| **Low** | `k` | Trivially replaceable by an expression of the loop index (`i + 1`), but as written it is loop-carried. |
| **Privatization required** | `x`, `x1`, `w`, `chk`, `test`, `dx`, `us`, `vs`, `vh`, `we`, `w_exact`, `wt`, `steps`, `steps_ave` | Must be declared thread-private if the enclosing loop is parallelized; otherwise concurrent writes from different threads to the same stack location. |

The `static` buffer in `timestamp` (line 234) is also technically a shared mutable, but `timestamp` is only called outside the computational loops so it poses no practical risk during parallel execution of the hot path.

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

**The trial loop (`for it = 1..n`) is the best candidate for parallelization.**

This loop runs 10,000 independent Monte Carlo trials from the same starting point `x`. Each trial simulates a random walk, accumulates a weight `w`, and contributes that weight to the sum `wt`. The trials are *conceptually* independent — the physics of one random walk does not depend on another. The only thing coupling them is the shared PRNG state (`seed`), which is an implementation artifact, not a mathematical dependency.

To parallelize this loop, you would need to:

1. **Eliminate the `seed` dependency** by giving each thread (or each trial) its own independent PRNG state. This is not merely a synchronization concern — even if you serialized access to a shared seed with a lock, the *order* of draws would become nondeterministic, changing the numerical result and potentially degrading the statistical quality of the ensemble. Each thread needs a properly seeded, independent generator.
2. **Handle `wt` as a reduction.** Each trial produces a value of `w` that is summed into `wt`. This is a classic sum-reduction: each thread accumulates a private partial sum, and the partial sums are combined after all trials complete.
3. **Handle `steps` as a reduction.** Same structure — each trial contributes a step count, and the total is summed.
4. **Privatize all per-trial working variables:** `x1`, `w`, `chk`, `dx`, `us`, `vs`, `vh`, `we`. These are all written and read within a single trial and carry no information between trials. Each thread needs its own copy.

**The outer spatial loop (`for i = 0..ni+1`) can also be parallelized, but with more caveats.**

Each spatial point `i` computes an independent Monte Carlo estimate. The points do not interact mathematically. However, the implementation couples iterations through:

- `seed` — the PRNG state flows serially from one spatial point to the next. Same resolution as above: independent PRNG per thread.
- `err` — a sum-reduction across spatial points.
- `n_int` — a count-reduction (conditionally incremented).
- `k` — a trivial loop-carried counter that equals `i + 1` and can be rewritten as such.

If you parallelize this loop, each thread handles one or more spatial grid points, and within each point it runs all 10,000 trials serially. This is a coarser granularity than parallelizing the trial loop.

**Both loops can be parallelized simultaneously** (nested parallelism), but with only 23 outer iterations and 10,000 inner iterations, the practical benefit of outer parallelism is limited. The inner trial loop offers far more parallel work.

### Which loop cannot be parallelized

**The innermost while loop (random walk steps) cannot be parallelized.**

Each step of the walk depends on the result of the previous step through tight, unavoidable data dependencies:

- `x1` at step \(k\) is computed from `x1` at step \(k-1\): `x1 = x1 + dx`
- `w` at step \(k\) depends on `w` at step \(k-1\)  and on the potential evaluated at the *new* `x1`: `w = w - 0.5 * h * (vh * we + vs * w)` where `we` itself is a function of the prior `w`
- `chk` (the termination condition) is computed from the *current* `x1`
- `seed` is mutated on every step

These are all **loop-carried, serial-order dependencies**. Step \(k\) cannot begin until step \(k-1\) completes. There is no way to partition the iteration space because the number of steps is not even known in advance — it is determined dynamically by when the walker exits the domain. This loop is fundamentally sequential.

### Unit of parallel work

The natural unit of parallel work is **one Monte Carlo trial**: a single execution of the entire while-loop random walk from the starting point `x` to the boundary. This is an indivisible unit because the walk steps within a trial are sequential, but distinct trials are independent.

Each work unit consists of:
- Initializing `x1 = x`, `w = 1.0`, `chk = 0.0`
- Executing the full random walk (variable number of steps, on the order of \(a^2/h\) on average for central points)
- Returning the final weight `w` and the step count

With `n = 10,000` trials per spatial point and 21 interior spatial points, there are roughly **210,000 independent work units** in total.

### Type of parallelism

This is **data parallelism**, specifically of the *embarrassingly parallel* / Monte Carlo variety.

- Every trial executes the *same* computation (a random walk with weight accumulation) on *different* random data (different PRNG streams).
- There is no communication between trials during execution.
- The only synchronization point is the reduction at the end: summing the per-trial `w` values into `wt` and the per-trial step counts into `steps`.
- There is no pipeline structure (no stages feeding into subsequent stages) and no task heterogeneity (every work unit is the same kind of computation). All trials run the same code; only the random numbers differ.

If both the outer spatial loop and the inner trial loop are parallelized, you get **two-level data parallelism**: the outer level partitions grid points across threads, and the inner level partitions trials across threads. But both levels are the same kind — independent replicas of the same computation.

### How memory access patterns affect the decision

The memory access patterns in this code are **exceptionally favorable** for parallelization:

1. **All computation is scalar.** There are no arrays, no shared data structures, no heap allocations. Every variable in the hot path is a stack-local scalar. When threads are given private copies of the working variables, there is **zero shared mutable state** during trial execution (after resolving the PRNG). This means:
   - No false sharing (no adjacent memory locations accessed by different threads)
   - No cache coherence traffic during the parallel phase
   - No need for locks, atomics, or memory fences during trial execution

2. **No stride or indexing patterns.** Since there are no arrays, there are no concerns about spatial locality, cache line alignment, or memory bandwidth. Each thread's working set is a handful of doubles and ints that fit trivially in registers or L1 cache.

3. **The only shared-memory write points are reductions.** `wt` and `steps` are accumulated at the end of each trial, and `err` and `n_int` at the end of each spatial point. These are single-scalar reductions — the cheapest possible synchronization pattern. They can be implemented as thread-local accumulators with a single combine step, adding effectively zero overhead.

4. **The PRNG state is the only complication.** `r8_uniform_01` mutates `seed` through a pointer, which would create a serial bottleneck and a correctness hazard if shared. But because the state is a single integer, giving each thread its own PRNG instance is trivial in terms of memory — it adds one `int` per thread. The concern here is not memory but **statistical independence**: the per-thread generators must produce uncorrelated sequences.

5. **`timestamp`'s static buffer** is the only piece of static mutable memory in the program, but it is called only outside the computational loops and poses no issue.

In summary: the memory profile of this program is about as clean as it gets for parallelization. The entire cost of going parallel is (a) replicating the PRNG state and (b) performing scalar reductions — both negligible. The dominant constraint is the sequential nature of each individual random walk, which sets the floor on the grain size of parallel work.

---

**User**

Implement the parallelization strategy you proposed using OpenMP. Modify the given code. Explain every OpenMP directive you add.

---

**Cursor**

Let me re-read the file to have the exact current state before making edits.

Now I'll make the changes step by step.

**Step 1 — Add the OpenMP header.**

**Step 2 — Print the thread count in the banner** so the user can verify OpenMP is active.

**Step 3 — Eliminate the loop-carried dependency on `k`.** Currently `k` is incremented each iteration, but it always equals `i + 1`. Recomputing it removes a false dependency that would otherwise require serialization or special handling.

**Steps 3 & 4 — Add the OpenMP parallel directive on the trial loop, with reduction and private clauses, and replace the shared `seed` with per-trial independent PRNG state.**

Now let me verify the final state of the file and check for linter errors.

No linter errors. The file is clean. Here is a complete explanation of every change and every OpenMP directive.

---

## Summary of Changes

Four modifications were made. I'll walk through each one and the rationale from the dependency analysis.

---

### 1. `#include <omp.h>` (line 5)

```5:5:/Users/saradordevic/Desktop/feynman-kac/feynman-kac.c
# include <omp.h>
```

Required to use any OpenMP API function (`omp_get_max_threads`) and to have the compiler recognize `#pragma omp` directives. Without this header, calls like `omp_get_max_threads()` would fail to compile.

---

### 2. `omp_get_max_threads()` diagnostic print (line 65)

```65:65:/Users/saradordevic/Desktop/feynman-kac/feynman-kac.c
  printf ( "  Number of OpenMP threads = %d\n", omp_get_max_threads ( ) );
```

This is not a directive but a runtime query. It prints the thread pool size so the user can confirm that OpenMP is active and how many threads are in play. This is called *before* any parallel region, so it reports the environment's configured thread count (controlled by `OMP_NUM_THREADS` or the implementation default).

---

### 3. Elimination of the loop-carried dependency on `k` (line 95)

```95:95:/Users/saradordevic/Desktop/feynman-kac/feynman-kac.c
    k = i + 1;
```

The original code had `k = 0` before the loop and `k = k + 1` inside it — a loop-carried dependency. Since `k` is always equal to `i + 1`, recomputing it from the loop index makes each iteration of the outer spatial loop independent with respect to `k`. This change is not OpenMP-specific, but it removes a serial dependency that would have been an obstacle if the outer loop were ever parallelized too.

---

### 4. The OpenMP directive on the trial loop (lines 120–122)

This is the core parallelization. Here is the directive:

```120:122:/Users/saradordevic/Desktop/feynman-kac/feynman-kac.c
    # pragma omp parallel for \
        reduction ( + : wt, steps ) \
        private ( x1, w, chk, dx, us, vs, vh, we )
```

This single compound directive does three things. I'll break down each clause:

#### `#pragma omp parallel for`

This is a *combined* directive. It fuses two operations:

- **`parallel`** — creates a team of threads. Each thread gets its own stack frame. The team exists for the duration of the structured block (the `for` body).
- **`for`** — distributes the iterations of the immediately following `for` loop across the thread team. By default, OpenMP uses static scheduling: the 10,000 iterations are divided into roughly equal contiguous chunks, one per thread.

The loop variable `it` is **automatically made private** by the `for` work-sharing construct — each thread gets its own copy and iterates over its assigned range.

#### `reduction ( + : wt, steps )`

This clause addresses the two **loop-carried accumulation dependencies** identified in the analysis:

- `wt` — each trial adds its final path weight `w` to `wt` (`wt = wt + w`).
- `steps` — each random walk step increments `steps` (`steps = steps + 1`).

Without this clause, multiple threads would perform unsynchronized read-modify-write on the same shared variables — a data race.

The `reduction(+:...)` clause instructs OpenMP to:
1. Create a **thread-private copy** of each listed variable, initialized to the identity element for `+` (which is `0`).
2. Let each thread accumulate into its private copy throughout its portion of the loop.
3. After all threads complete, **combine** all private copies into the original shared variable using the `+` operator.

This eliminates the race and avoids the overhead of per-iteration locking.

#### `private ( x1, w, chk, dx, us, vs, vh, we )`

These eight variables were identified in the analysis as **loop-private** (per-trial working state). In the serial code they are declared in `main`'s scope and reused across iterations. In a parallel context, if left shared, multiple threads would write to the same memory locations simultaneously — a data race on every variable in the list.

The `private` clause instructs OpenMP to create a **separate, uninitialized copy** of each listed variable for every thread. Each thread reads and writes only its own copy. The original shared variable is not touched during the parallel region.

These variables do not need `firstprivate` (initialized from the original) because every one of them is assigned before it is read within each trial iteration:
- `x1 = x` (line 127)
- `w = 1.0` (line 131)
- `chk = 0.0` (line 135)
- `dx` is set inside the `if/else` (lines 144–149)
- `us` is set from `r8_uniform_01` (line 142)
- `vs`, `vh` are set from `potential()` calls (lines 152, 160)
- `we` is computed before use (line 162)

---

### 5. Per-trial PRNG state: `seed_local` (line 125)

```125:125:/Users/saradordevic/Desktop/feynman-kac/feynman-kac.c
      int seed_local = seed + it + i * n;
```

This addresses the most critical dependency from the analysis: `seed`. The original PRNG (`r8_uniform_01`) is a sequential state machine — each call mutates `*seed` and the next call depends on the result. Sharing it across threads would be both a data race and a statistical error (correlated or corrupted random streams).

The solution is to give each trial its own deterministic seed. `seed_local` is declared *inside* the loop body, making it **automatically private** to each thread (block-scoped). The formula `seed + it + i * n` ensures:

- Every trial across all spatial points gets a unique seed (`it` ranges 1..10000, and `i * n` offsets each spatial point by 10000).
- The seeds are deterministic and reproducible regardless of thread count.

All calls to `r8_uniform_01` inside the loop now use `&seed_local` instead of `&seed`. The shared `seed` variable is only **read** (never written) inside the parallel region, which is safe.

Note that the numerical results will differ from the original serial program because the trials receive different random number sequences. This is inherent to any parallelization of Monte Carlo — what is preserved is the statistical correctness of the estimator, not the specific random realization.

---

### Variables not listed in any clause — and why

| Variable | Why it's safe as shared (implicit default) |
|---|---|
| `a`, `h`, `rth`, `n` | Constants. Only read, never written inside the parallel region. |
| `x` | Set before the parallel region, only read inside it (each trial copies it to the private `x1`). |
| `seed` | Only read (to initialize `seed_local`), never written. |
| `i` | Outer loop variable, only read inside the inner parallel loop. |
| `it` | Automatically private (OpenMP `for` loop variable). |
| `seed_local` | Automatically private (declared inside the loop body). |

---

### Compilation

To compile with OpenMP support:

```bash
gcc -fopenmp -o feynman-kac feynman-kac.c -lm
```

The `-fopenmp` flag enables OpenMP directive processing and links the OpenMP runtime. Without it, the `#pragma omp` lines are silently ignored by the compiler and the program runs serially (with the `omp_get_max_threads` call failing to link).

