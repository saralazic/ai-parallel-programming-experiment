# Code structural analysis
_Exported on 3/10/2026 at 19:21:44 GMT+1 from Cursor (2.4.31)_

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

This is a classical **Lennard-Jones molecular dynamics (MD) simulation** implementing the velocity Verlet integration scheme with periodic boundary conditions. Here is the full structural analysis.

---

## 1. Control Flow Structure

### Top-Level Execution Order (`main`, lines 203-308)

The program follows a strict sequential pipeline:

1. **Initialization phase** (lines 205-232): Parse `mm` from `argv[1]`, derive all simulation parameters (`npart = 4*mm^3`, `side`, `rcoff`, `h`, etc.), and allocate VLAs for positions `x`, velocities `vh`, and forces `f`.
2. **Lattice generation** (line 253): `fcc()` fills `x` with atom positions on a face-centered cubic lattice.
3. **Velocity/force initialization** (lines 257-258): `mxwell()` samples initial velocities from a Maxwell-Boltzmann distribution; `dfill()` zeroes the force array.
4. **Main timestep loop** (lines 268-303): `for move = 1..movemx`, executing a fixed pipeline each iteration:

```
domove → forces → mkekin → velavg → [conditional scaling] → [conditional print]
```

5. **Timing output** (lines 305-307).

### Conditionals

- **`domove`** (lines 64-65): Periodic boundary wrapping — if a coordinate leaves the box `[0, side]`, it is shifted back. Two independent `if` tests per coordinate component.
- **`forces`** (line 172): Distance cutoff test `rd <= rcoffs` — the Lennard-Jones interaction is computed only if the squared distance between a pair is within the cutoff radius squared.
- **`forces`** (lines 158-169): Minimum-image convention — six independent `if` tests wrap each displacement component into `[-sideh, +sideh]`.
- **`main`** (line 291): Temperature rescaling is applied only when `move < istop` AND `move` is a multiple of `irep`.
- **`main`** (line 301): Output is printed only every `iprint` steps.
- **`dscal`** (line 84): Branch on `incx == 1` to select a unit-stride vs. general-stride loop.
- **`mxwell`** (line 357): `while (s >= 1.0)` rejection loop for Box-Muller sampling — non-deterministic iteration count.

---

## 2. All Loops

### `dfill` (line 48)

```c
for (i=0; i<(n-1)*ia+1; i+=ia)
```

- **Purpose**: Fill array `a` with scalar `val` at stride `ia`.
- **Iteration space**: `ceil(n)` iterations. When called with `ia=1`, this is a contiguous fill of `n` elements.

### `domove` (line 59)

```c
for (i=0; i<n3; i++)
```

- **Purpose**: Advance positions via `x[i] += vh[i] + f[i]`, apply periodic boundary conditions, do a partial velocity update `vh[i] += f[i]`, and zero forces.
- **Iteration space**: `3 * npart` iterations (one per coordinate component).

### `dscal` (lines 85-93)

Two alternative loops depending on `incx`:

```c
// incx == 1 path:
for (i=0; i<n; i++)           // unit stride
// incx != 1 path:
for (i=0; i<n; i++) { j+=incx; }  // general stride
```

- **Purpose**: Multiply every element of `sx` by scalar `sa`.
- **Iteration space**: `n` iterations in both cases.

### `fcc` (lines 104-122)

Two 4-deep nested loop nests:

**First nest** (lines 104-112):
```
lg ∈ [0,1] × i ∈ [0,mm) × j ∈ [0,mm) × k ∈ [0,mm)
```
- Iteration space: `2 × mm³` iterations, generating `2 × mm³` atoms (6 × mm³ coordinate values).

**Second nest** (lines 114-122):
```
lg ∈ [1,2] × i ∈ [0,mm) × j ∈ [0,mm) × k ∈ [0,mm)
```
- Iteration space: `2 × mm³` iterations, generating the remaining `2 × mm³` atoms.

**Total**: `4 × mm³ = npart` atoms, consistent with the FCC unit cell (4 atoms per cell).

- **Purpose**: Place atoms at the four sublattice positions of an FCC lattice within the simulation box.

### `forces` — outer loop (line 144)

```c
for (i = 0; i < npart * 3; i += 3)
```

- **Purpose**: Iterate over each particle `i` (stride-3 indexing into the flat coordinate array).
- **Iteration space**: `npart` iterations.

### `forces` — inner loop (line 153)

```c
for (j = i + 3; j < npart * 3; j += 3)
```

- **Purpose**: For each particle `i`, iterate over all particles `j > i` to compute pairwise forces. This is a **triangular iteration** exploiting Newton's third law — each pair `(i,j)` is visited exactly once, and both `f[i]` and `f[j]` are updated.
- **Iteration space**: For a given outer index `i` (0-indexed particle `i/3`), the inner loop runs `npart - i/3 - 1` iterations. Total across both loops: `npart*(npart-1)/2` pair evaluations (though the cutoff test prunes many).

### `main` — timestep loop (line 268)

```c
for (move = 1; move <= movemx; move++)
```

- **Purpose**: Execute `movemx` timesteps of the MD simulation.
- **Iteration space**: `movemx` (=20) iterations. Each iteration is a complete simulation step.

### `mkekin` (line 328)

```c
for (i=0; i<3*npart; i++)
```

- **Purpose**: Scale forces by `hsq2`, complete the velocity update `vh[i] += f[i]`, and accumulate kinetic energy `sum += vh[i]²`.
- **Iteration space**: `3 * npart` iterations (unit stride).

### `mxwell` — multiple loops (lines 355-391)

1. **Box-Muller sampling** (line 355):

```c
for (i=0; i<n3; i+=2) {
    while (s>=1.0) { ... }  // rejection loop (non-deterministic)
}
```
   - Iteration space: `n3/2` outer iterations, each with a geometrically-distributed number of inner rejection iterations. Generates pairs of Gaussian random numbers.

2. **Center-of-mass velocity removal** — six loops in three groups (lines 367-388), one group per coordinate axis (x, y, z):

```c
// For each axis offset ∈ {0, 1, 2}:
for (i=offset; i<n3; i+=3) sp += vh[i];   // compute mean
for (i=offset; i<n3; i+=3) { vh[i] -= sp; ekin += vh[i]*vh[i]; }  // subtract mean, accumulate KE
```
   - Iteration space: `npart` iterations each, stride 3.

3. **Final scaling** (line 391):

```c
for (i=0; i<n3; i++) vh[i] *= sc;
```
   - Iteration space: `3 * npart`, unit stride.

### `velavg` (line 426)

```c
for (i=0; i<npart*3; i+=3)
```

- **Purpose**: Compute the magnitude of each particle's velocity vector, count how many exceed the threshold `vaverh`, and accumulate total speed.
- **Iteration space**: `npart` iterations, stride 3.

---

## 3. Data Dependencies

### Inter-function dependencies within each timestep

The main loop pipeline has strict serial data-flow dependencies:

```
domove(x, vh, f) ──→ forces(x, f) ──→ mkekin(f, vh) ──→ velavg(vh)
  writes: x, vh, f     writes: f,         writes: f, vh      reads: vh
  (zeros f)             epot, vir          reads: f           writes: count, vel
```

- **`domove` → `forces`**: `domove` produces the updated positions `x[]` and zeros `f[]`. `forces` reads `x[]` and writes into `f[]`.
- **`forces` → `mkekin`**: `mkekin` reads `f[]` (the raw forces), scales them, and uses them to update `vh[]`.
- **`mkekin` → `velavg`**: `velavg` reads the updated `vh[]`.
- **Conditional scaling** (`dscal`): Reads and writes `vh[]`, depends on `ekin` from `mkekin`.
- **`prnout`**: Reads `ekin` (from `mkekin`), `epot` and `vir` (globals set by `forces`), `vel` and `count` (from `velavg`). Pure consumer, no write-back.

### Inter-timestep dependencies

Each timestep's `domove` reads `x`, `vh`, and `f` produced by the *previous* timestep's `mkekin` (for `vh` and `f`) and `domove` itself (for `x`). This creates a loop-carried dependency chain across the timestep loop.

### Intra-function dependencies in `forces`

Within the inner loop body:
- `xx, yy, zz` depend on `xi, yi, zi` (set once per outer iteration) and `x[j], x[j+1], x[j+2]`.
- `rd` depends on `xx, yy, zz`.
- `rrd → rrd2 → rrd3, rrd4 → rrd6, rrd7 → r148` is a strict sequential chain.
- `epot` and `vir` carry a **loop-carried accumulation dependency** across all inner iterations.
- `fxi, fyi, fzi` carry a **loop-carried accumulation dependency** across the inner loop, then are flushed to `f[i]` after the inner loop completes.
- `f[j]` elements are read-modify-written in the inner loop, creating a dependency between any two outer iterations `i₁` and `i₂` that share a common `j`.

### Intra-function dependencies in `mkekin`

- `f[i] *= hsq2` must complete before `vh[i] += f[i]`, which must complete before `sum += vh[i]*vh[i]`. This is a strict per-element chain, plus `sum` is loop-carried.

---

## 4. Memory Access Patterns

### `domove`

- **Unit-stride sequential** reads and writes to `x[0..n3-1]`, `vh[0..n3-1]`, `f[0..n3-1]`.
- Maximally cache-friendly. Three arrays are interleaved in access but each is accessed sequentially.

### `forces`

- **Outer loop**: Reads `x[i], x[i+1], x[i+2]` (stride-3 sequential). These are loaded once and cached in registers (`xi, yi, zi`).
- **Inner loop**: Reads `x[j], x[j+1], x[j+2]` sequentially (stride-3). Reads *and writes* `f[j], f[j+1], f[j+2]` (stride-3, read-modify-write).
- **Triangular access pattern**: For `i=0`, the inner loop scans nearly the full `x[]` and `f[]` arrays. For `i` near `npart`, the inner loop is very short. This creates an asymmetric, shrinking sweep pattern.
- **Spatial locality on `f[]`**: The inner loop modifies `f[j]` for `j > i`. Additionally, after the inner loop, `f[i]` is updated. Since different outer iterations `i` write to overlapping regions of `f[]`, there is **no spatial partitioning** of writes to `f[]` across outer iterations.
- **`epot` and `vir`**: Scalar globals, repeatedly written. Likely register-resident in a serial execution.

### `mkekin`

- **Unit-stride sequential** read-modify-write on `f[0..3*npart-1]` and `vh[0..3*npart-1]`.
- `sum` is a scalar accumulator.

### `fcc`

- **Sequential write** to `x[0..3*npart-1]` via the `ijk` counter (always incremented by 3). Write-only, maximally cache-friendly.
- Read pattern on loop indices is trivial (registers).

### `mxwell`

- **Stride-2** writes in the Box-Muller loop.
- **Stride-3** reads/writes in the center-of-mass removal loops (three passes, each at a different offset — accesses `vh[0], vh[3], vh[6],...` then `vh[1], vh[4], vh[7],...` then `vh[2], vh[5], vh[8],...`). This is stride-3 with poor spatial locality per pass, though each pass covers the whole array.
- **Unit-stride** final scaling pass.

### `velavg`

- **Stride-3** reads of `vh[]`, accessing three consecutive elements `vh[i], vh[i+1], vh[i+2]` per iteration (effectively reading every element, but grouped in triplets with a stride-3 outer loop). Good spatial locality.

### `dscal` / `dfill`

- Unit-stride (when `incx`/`ia` = 1): sequential, cache-friendly.
- General stride: potentially poor locality for large strides.

---

## 5. Variables Subject to Race Conditions Under Shared-Memory Parallelism

### `forces` (the most critical function)

If the **outer loop** (`i`) is distributed across threads:

| Variable | Type | Race? | Reason |
|---|---|---|---|
| `epot` | global, accumulated | **Yes** | Every outer iteration's inner loop adds to `epot`. Concurrent `+=` on a shared scalar. |
| `vir` | global, accumulated | **Yes** | Same as `epot` — concurrent `-=` from multiple threads. |
| `f[j], f[j+1], f[j+2]` | array, write in inner loop | **Yes** | The inner loop writes `f[j] -= forcex` for `j > i`. Two different outer iterations `i₁` and `i₂` (where `i₁ < i₂`) both iterate over `j = i₂`, meaning thread handling `i₁` writes to `f[i₂]` at the same time thread handling `i₂` writes to `f[i₂]` (line 194-196). The write sets from different outer iterations overlap on the `f[]` array. |
| `f[i], f[i+1], f[i+2]` | array, write after inner loop | No (if `i` uniquely assigned) | Each outer iteration writes to its own `f[i]` block, and no other outer iteration writes to `f[i]` because the inner loop only writes to `f[j]` where `j > i`. |
| `xi, yi, zi, fxi, fyi, fzi` | local scalars | No | Private to each outer iteration (can be made thread-private). |

If the **inner loop** (`j`) is parallelized for a fixed `i`:

| Variable | Type | Race? | Reason |
|---|---|---|---|
| `fxi, fyi, fzi` | accumulated over `j` | **Yes** | Multiple threads would concurrently `+=` to these accumulators. |
| `epot`, `vir` | accumulated over `j` | **Yes** | Same as above. |
| `f[j]` | per-`j` write | No | Each `j` is unique, so each thread writes to distinct `f[j]` elements. |

### `mkekin`

If the loop (line 328) is parallelized:

| Variable | Race? | Reason |
|---|---|---|
| `sum` | **Yes** | Scalar accumulator, concurrent `+=` from all iterations. |
| `f[i]`, `vh[i]` | No | Each iteration touches a distinct index. |

### `velavg`

If the loop (line 426) is parallelized:

| Variable | Race? | Reason |
|---|---|---|
| `count` | **Yes** | Global variable, conditionally incremented (`count++`) by multiple iterations. |
| `vel` | **Yes** | Local accumulator, concurrent `+=`. |
| `vh[i]` | No | Read-only access. |

### `domove`

If the loop (line 59) is parallelized:

| Variable | Race? | Reason |
|---|---|---|
| `x[i]`, `vh[i]`, `f[i]` | No | Each iteration accesses a unique index `i`. Fully independent. |

**No race conditions** — `domove` is embarrassingly parallel.

### `dscal` / `dfill`

**No race conditions** — each iteration writes to a distinct array element.

### `fcc`

The `ijk` counter is incremented sequentially (`ijk += 3`) and used to index into `x[]`. If the loop nest were parallelized, `ijk` would be a **race condition** since it is a shared, serially-incremented write index. However, `ijk` could be computed as a closed-form function of the loop indices, eliminating the dependency.

### Summary of all race-prone variables

| Function | Variable | Kind |
|---|---|---|
| `forces` | `epot` | Global scalar reduction |
| `forces` | `vir` | Global scalar reduction |
| `forces` | `f[j], f[j+1], f[j+2]` | Array write overlap (Newton's 3rd law update) |
| `mkekin` | `sum` | Local scalar reduction |
| `velavg` | `count` | Global scalar reduction |
| `velavg` | `vel` | Local scalar reduction |
| `fcc` | `ijk` | Shared induction variable |

The `forces` function presents the most complex parallelization challenge: `epot` and `vir` are straightforward reductions, but the **overlapping writes to `f[j]`** (arising from the triangular Newton's-third-law optimization) represent a true structural data conflict that cannot be resolved by a simple reduction clause — it requires either atomic updates, private force arrays per thread with a merge step, or restructuring the iteration space to eliminate the asymmetric update pattern.

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

### Which Loops Can Be Parallelized

**1. `domove` — the loop at line 59: Fully parallelizable**

Each iteration `i` reads and writes exclusively to `x[i]`, `vh[i]`, and `f[i]`. There are no loop-carried dependencies, no shared accumulators, and no cross-iteration data flow. The unit of parallel work is a single coordinate component (or, more naturally, a particle — grouping three consecutive iterations). This is **embarrassingly parallel** with perfect data partitioning: if you assign each thread a contiguous block of indices, each thread operates on a disjoint memory region with no synchronization required. The unit-stride sequential access pattern means each thread's working set is contiguous in memory, yielding good cache line utilization with no false sharing (provided partition boundaries are aligned to particle boundaries, i.e., multiples of 3).

**2. `forces` — the outer loop at line 144: Parallelizable with significant caveats**

This is the computational hotspot (O(n²) pair interactions) and the primary target for parallelization. The outer loop iterates over particles, and the inner loop computes interactions with all higher-indexed particles. It *can* be parallelized over the outer loop, but three categories of conflicts must be resolved:

- **Scalar reductions** on `epot` and `vir`: straightforward — each thread accumulates into a private copy, then a final reduction merges them.
- **The `f[j]` write conflict**: This is the hard problem. The inner loop writes to `f[j]` for all `j > i` (the Newton's third law optimization). When two threads handle outer indices `i₁` and `i₂` where `i₁ < i₂`, the thread handling `i₁` will write to `f[i₂]` in its inner loop at the same time the thread handling `i₂` writes to `f[i₂]` after its own inner loop. The overlapping write region is the entire `f[]` array above each thread's own `i`. Resolution strategies include:
  - **Private force arrays per thread** with a post-loop reduction (merge) pass. This trades memory (O(threads × n)) for eliminating synchronization, and adds a O(threads × n) merge cost.
  - **Atomic updates** on `f[j]` elements. This preserves memory efficiency but introduces serialization at high contention.
  - **Abandoning Newton's third law** — each thread computes the full force on its own particles by iterating `j` over *all other* particles, not just `j > i`. This doubles the floating-point work but makes `f[]` writes perfectly partitioned (each thread only writes to its own particle's force entries). The inner loop becomes `j = 0..npart*3` (skipping `j == i`), and no thread ever writes to another thread's `f[]` region.

**3. `mkekin` — the loop at line 328: Parallelizable with a reduction**

The updates `f[i] *= hsq2` and `vh[i] += f[i]` are per-element with no cross-iteration dependency. The only conflict is the accumulation into `sum`. Each thread can maintain a private `sum` and a final reduction produces the global value. The unit of work is a single coordinate component. Unit-stride access means excellent spatial locality per thread.

**4. `velavg` — the loop at line 426: Parallelizable with reductions**

Each iteration reads `vh[i], vh[i+1], vh[i+2]` (read-only), and the only shared state is `vel` (accumulated) and `count` (conditionally incremented). Both are scalar reductions. The unit of work is one particle (one iteration of the stride-3 loop). Stride-3 access reads three consecutive elements per particle, so spatial locality is good.

**5. `dscal` and `dfill`: Trivially parallelizable**

Each iteration writes to a distinct element. No dependencies, no shared state. These are called on arrays of size `3*npart` with unit stride, so they partition cleanly.

---

### Which Loops Cannot Be Parallelized

**1. The main timestep loop (`move = 1..movemx`, line 268): Cannot be parallelized**

Each timestep consumes the positions, velocities, and forces produced by the previous timestep. The dependency chain is:

```
step t: domove(x,vh,f) → forces(x,f) → mkekin(f,vh) → velavg(vh)
                                                          │
step t+1: domove(x,vh,f) ←──────────────────────────────┘
```

The output state `(x, vh, f)` of step `t` is the input to step `t+1`. This is a fundamental loop-carried dependency inherent to the physics: the trajectory is a sequential Markov chain. No amount of restructuring can pipeline or overlap timesteps because `domove` at step `t+1` needs the fully updated `vh` and `f` from step `t`.

**2. The `mxwell` Box-Muller loop (line 355): Cannot be straightforwardly parallelized**

The `while (s >= 1.0)` rejection loop calls `drand48()`, which maintains internal PRNG state. Parallelizing the outer `for` loop would cause a race on the PRNG state, and even with a thread-safe PRNG, results would be non-reproducible (the mapping of random numbers to array positions would depend on scheduling). Since this is a one-time initialization cost, it is not worth parallelizing.

**3. The `mxwell` center-of-mass removal loops (lines 367-388): Partially constrained**

Within each axis, the first loop (compute `sp`) must complete before the second loop (subtract `sp` and accumulate `ekin`) can begin — there is a data dependency on the mean value. The three axes (x, y, z) are *independent of each other* in the `sp` computation, but they share `ekin` in the subtraction loops. So the three axis groups could in principle execute concurrently (task parallelism over 3 units), but each individual loop is small and the synchronization overhead would dominate.

**4. `fcc` lattice generation (lines 104-122): Not parallelizable as written**

The `ijk` variable is a sequentially-incremented induction variable shared across all iterations of the 4-deep loop nest. However, this is an *artificial* dependency: `ijk` could be computed as a closed-form expression of `(lg, i, j, k)`, making each iteration independent. Even so, this is a one-time initialization with O(n) work, so parallelizing it offers negligible benefit.

---

### Units of Parallel Work

| Function | Natural unit | Granularity |
|---|---|---|
| `domove` | One particle (3 consecutive array elements) | Fine-grained, `npart` independent units |
| `forces` outer loop | One particle's interactions with all higher-indexed particles | Coarse-grained but **imbalanced** — particle 0 does ~`npart` pair evaluations, particle `npart-1` does 0 |
| `mkekin` | One coordinate component (or one particle for groups of 3) | Fine-grained, `3*npart` independent units (aside from the reduction) |
| `velavg` | One particle | Fine-grained, `npart` independent units |

For `forces`, the triangular iteration creates a **load imbalance** problem. If the outer loop is statically partitioned into contiguous blocks, the first block does the most work and the last block does the least. A dynamic or guided scheduling policy, or a cyclic distribution, would better balance the load.

---

### Kind of Parallelism

**The dominant form is data parallelism.** Every parallelizable loop partitions the particle array (or coordinate array) across threads, and each thread applies the same operations to its subset. There is no functional decomposition — all threads execute the same code on different data.

More specifically:

- **`domove`, `dscal`, `dfill`**: Pure data parallelism with no communication. SIMD-like — the same operation is applied to every element independently.
- **`forces`**: Data parallelism with a **reduction** pattern (for `epot`, `vir`) and a **scatter-update** conflict (for `f[j]`). The Newton's third law optimization introduces a communication pattern that breaks the clean data-parallel model.
- **`mkekin`, `velavg`**: Data parallelism with scalar reductions.

**There is no viable task parallelism** within a timestep. The functions in the pipeline (`domove → forces → mkekin → velavg`) have strict producer-consumer dependencies — `forces` cannot begin until `domove` completes, `mkekin` cannot begin until `forces` completes, and so on. They cannot be overlapped.

**There is no viable pipeline parallelism** across timesteps, due to the loop-carried dependency on the full state vector.

---

### How Memory Access Patterns Affect the Strategy

**Cache locality and partitioning.** The position array `x[]`, velocity array `vh[]`, and force array `f[]` are flat arrays of `3*npart` doubles laid out contiguously. For `domove`, `mkekin`, and `velavg`, a contiguous block partition (thread `t` owns indices `[t*chunk, (t+1)*chunk)`) gives each thread a contiguous memory region. This maximizes spatial locality: each thread's working set occupies consecutive cache lines, and there is no cache line sharing between threads (assuming chunk boundaries align to cache line boundaries, or at minimum to particle boundaries — multiples of `3 * sizeof(double)` = 24 bytes).

**False sharing risk.** If partition boundaries do not align to cache line boundaries (typically 64 bytes = 8 doubles), two threads may write to different elements within the same cache line, causing false sharing. Aligning partition boundaries to multiples of 8 elements (or at least to particle boundaries of 3 elements) mitigates this. For `domove` and `mkekin` this is trivially handled.

**The `forces` function has adversarial memory access for parallelism.** The inner loop's `f[j] -= force` writes sweep across the entire `f[]` array above index `i`. If the outer loop is partitioned across threads, every thread writes into every *other* thread's region of `f[]`. This means:

- **No spatial partitioning of writes is possible** under the current algorithm. Thread 0 (handling low `i` values) writes to `f[j]` elements that "belong to" every other thread.
- The **private force array** approach trades this conflict for O(`threads × 3 × npart`) additional memory and a post-computation merge pass. The merge pass itself is data-parallel (each thread sums its slice of the private arrays), but the working set of the force computation is multiplied by the thread count, potentially exceeding cache capacity and causing capacity misses.
- The **redundant computation** approach (dropping Newton's third law) trades 2× arithmetic for clean memory partitioning: each thread only *reads* the shared `x[]` array and only *writes* to its own slice of `f[]`. The read sharing of `x[]` is cache-friendly — all threads sweep the same data, which likely stays in a shared L2/L3 cache. This approach has the best memory access pattern for parallelism despite the doubled FLOP count, and it can be preferable on architectures where memory access costs dominate (e.g., NUMA systems, GPUs).

**Reduction variables (`epot`, `vir`, `sum`, `vel`, `count`) are scalars.** They occupy a single cache line and would be the site of extreme contention if updated atomically. Thread-private copies with a final merge eliminate this entirely at negligible memory cost (one scalar per thread).

**The triangular iteration space in `forces` also creates non-uniform memory access.** Early outer iterations sweep nearly the full `x[]` and `f[]` arrays, while late outer iterations touch only a small tail. Under a block partition, the first thread traverses the most memory (and does the most work), while the last thread touches the least. A cyclic or dynamic distribution distributes the memory traffic more evenly across threads, but at the cost of reduced spatial locality (each thread touches non-contiguous regions of `f[]` and `x[]`).

---

### Summary

The program's parallelism is fundamentally **intra-timestep data parallelism** over the particle array. The timestep loop is irreducibly sequential. Within each timestep, all four major kernels (`domove`, `forces`, `mkekin`, `velavg`) can be parallelized over particles, but they must execute in sequence. The primary challenge is the `forces` kernel, where Newton's third law creates overlapping writes to the force array — the central design decision in any parallelization strategy is how to resolve that conflict while maintaining acceptable memory overhead and load balance.

---

**User**

Implement the parallelization strategy you proposed using OpenMP. Modify the given code. Explain every OpenMP directive you add.

---

**Cursor**

Let me re-read the file to get the exact content for precise edits.

Starting with the OpenMP header, then applying all the parallelization changes across the file.

Now applying the parallelization to `dfill`, `domove`, and `dscal` simultaneously.

Now the critical change — restructuring the `forces` function. This replaces the triangular N*(N-1)/2 iteration (which writes to both `f[i]` and `f[j]`) with a full N*(N-1) symmetric iteration where each thread only writes to its own `f[i]`.

Now `mkekin` and `velavg`.

Now let me verify the final file looks correct, then test compilation.

The file looks correct. Let me verify it compiles.

