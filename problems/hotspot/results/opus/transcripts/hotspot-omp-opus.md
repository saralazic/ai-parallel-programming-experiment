# Code structural analysis
_Exported on 3/9/2026 at 19:51:43 GMT+1 from Cursor (2.4.31)_

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

This is the **HotSpot** thermal simulation from the Rodinia benchmark suite. It models transient heat conduction on a 2D chip floorplan by discretizing the heat equation into finite differences and iterating forward in time. Here is the full structural analysis.

---

## 1. Control Flow Structure

### `main` (lines 253–302)

Execution follows a strictly linear sequence:

1. **Argument validation** — If `argc != 8`, calls `usage()` which terminates via `exit(1)`. A second guard checks that grid dimensions, sim time, and thread count are positive; if not, also terminates.
2. **Memory allocation** — Three `calloc` calls for `temp`, `power`, `result`. A null check on `temp` or `power` calls `fatal()` (note: `result` is not checked).
3. **File I/O** — Two sequential calls to `read_input` for temperature and power data.
4. **Computation** — A single call to `compute_tran_temp`.
5. **Conditional output** — The environment variable `OUTPUT` is checked via `getenv()`; if set, results are written to `output.txt`.
6. **Cleanup** — `free(temp)` and `free(power)` are called. `result` is never freed (a memory leak).

### `compute_tran_temp` (lines 156–185)

1. Computes physical constants from chip geometry and material properties.
2. Derives reciprocals (`Rx_1`, `Ry_1`, `Rz_1`, `Cap_1`) to convert divisions into multiplications.
3. Enters a time-stepping loop that alternates two buffers via pointer swapping.

### `single_iteration` (lines 40–149)

The core computational kernel. The control flow here has two distinct paths per chunk:

1. **Boundary chunk path** (line 56): If the chunk touches any grid boundary (`r_start == 0`, `c_start == 0`, `r_end == row`, `c_end == col`), a doubly-nested loop iterates over the chunk with an 8-way `if/else if` chain distinguishing:
   - 4 corner cases (lines 60, 66, 72, 81)
   - 4 edge cases (lines 88, 96, 107, 118)
   - Interior cells within boundary chunks fall through all conditions and use an **uninitialized `delta`** (see Section 5).
   - After processing, `continue` skips the interior kernel.

2. **Interior chunk path** (lines 133–148): For non-boundary chunks, a simpler doubly-nested loop applies the standard 5-point stencil without boundary checks.

---

## 2. All Loops

### Loop A — Time-stepping loop

```177:183:/Users/saradordevic/Desktop/rodinia-master/openmp/hotspot/hotspot.cpp
        for (int i = 0; i < num_iterations; i++) {
            single_iteration(r, t, power, row, col, Cap_1, Rx_1, Ry_1, Rz_1,
                             step);
            FLOAT *tmp = t;
            t = r;
            r = tmp;
        }
```

- **Purpose**: Advances the simulation through `num_iterations` discrete time steps.
- **Iteration space**: `i ∈ [0, num_iterations)`.
- **Key behavior**: After each iteration, the `t` (input temperature) and `r` (output result) pointers are swapped. This double-buffering avoids copying the result array back to the temperature array.

### Loop B — Chunk loop

```50:50:/Users/saradordevic/Desktop/rodinia-master/openmp/hotspot/hotspot.cpp
    for (chunk = 0; chunk < num_chunk; ++chunk) {
```

- **Purpose**: Partitions the 2D grid into rectangular tiles (blocks) of size `BLOCK_SIZE_R × BLOCK_SIZE_C` (16×16) and iterates over them.
- **Iteration space**: `chunk ∈ [0, num_chunk)` where `num_chunk = (row × col) / (16 × 16)`.
- **Mapping**: Each chunk index is mapped to a 2D tile origin via:
  - `r_start = BLOCK_SIZE_R * (chunk / chunks_in_col)`
  - `c_start = BLOCK_SIZE_C * (chunk % chunks_in_row)`

### Loop C — Boundary cell loops (lines 57–58)

```57:58:/Users/saradordevic/Desktop/rodinia-master/openmp/hotspot/hotspot.cpp
            for (r = r_start; r < r_start + BLOCK_SIZE_R; ++r) {
                for (c = c_start; c < c_start + BLOCK_SIZE_C; ++c) {
```

- **Purpose**: Iterates over every cell within a boundary chunk and applies the appropriate boundary-condition formula based on the cell's position.
- **Iteration space**: `r ∈ [r_start, r_start + 16)`, `c ∈ [c_start, c_start + 16)`. Each loop body executes 256 times per chunk.

### Loop D — Interior cell loops (lines 133–134)

```133:134:/Users/saradordevic/Desktop/rodinia-master/openmp/hotspot/hotspot.cpp
        for (r = r_start; r < r_start + BLOCK_SIZE_R; ++r) {
            for (c = c_start; c < c_start + BLOCK_SIZE_C; ++c) {
```

- **Purpose**: Applies the standard interior 5-point stencil update to every cell within non-boundary chunks.
- **Iteration space**: Identical to Loop C: 16×16 = 256 cells per chunk.

### Loop E — `read_input` loop (line 222)

```222:222:/Users/saradordevic/Desktop/rodinia-master/openmp/hotspot/hotspot.cpp
    for (i = 0; i < grid_rows * grid_cols; i++) {
```

- **Purpose**: Reads `grid_rows × grid_cols` floating-point values from a file into a flat array.
- **Iteration space**: `i ∈ [0, grid_rows × grid_cols)`.

### Loop F — `writeoutput` loops (lines 201–207)

```201:207:/Users/saradordevic/Desktop/rodinia-master/openmp/hotspot/hotspot.cpp
    for (i = 0; i < grid_rows; i++)
        for (j = 0; j < grid_cols; j++) {

            sprintf(str, "%d\t%g\n", index, vect[i * grid_cols + j]);
            fputs(str, fp);
            index++;
        }
```

- **Purpose**: Writes the entire grid to a file in tab-separated format.
- **Iteration space**: `i ∈ [0, grid_rows)`, `j ∈ [0, grid_cols)`.

### Loop G — Output loop in `main` (line 292)

```292:293:/Users/saradordevic/Desktop/rodinia-master/openmp/hotspot/hotspot.cpp
        for (i = 0; i < grid_rows * grid_cols; i++)
            fprintf(file, "%d\t%g\n", i, temp[i]);
```

- **Purpose**: Writes the temperature array to `output.txt` (conditionally).
- **Iteration space**: `i ∈ [0, grid_rows × grid_cols)`.

---

## 3. Data Dependencies

### Inter-iteration (temporal) dependencies

The time-stepping loop (Loop A) has a **strict sequential dependency**: iteration `i+1` reads from the output of iteration `i`. The pointer swap makes the previous output the new input. **The iterations cannot be parallelized.**

### Intra-iteration (spatial) dependencies

Within a single call to `single_iteration`:

- **Read set**: `temp[r*col+c]`, and its 4 neighbors: `temp[(r±1)*col+c]`, `temp[r*col+(c±1)]`. Also `power[r*col+c]`.
- **Write set**: `result[r*col+c]`.

Since `result` and `temp` point to **different buffers** (ensured by the double-buffering scheme), writes to `result` do not affect reads from `temp`. This means **within a single time step, there are no read-after-write or write-after-read hazards between different cells**. Each cell's computation is independent of other cells' results within the same iteration.

The `power` array is **read-only** throughout the entire simulation — it is never written after initial loading.

### Scalar dependencies

- `delta` is a local scalar recomputed on every cell visit. It carries no dependency across iterations of Loops C/D.
- `r`, `c`, `chunk` are loop induction variables.
- `r_start`, `c_start`, `r_end`, `c_end` are derived from `chunk` and recomputed each chunk iteration.

---

## 4. Memory Access Patterns

### Array layout

All three arrays (`temp`, `power`, `result`) are allocated as **flat 1D arrays** of size `row × col`, using **row-major** indexing: element `(r, c)` is at offset `r * col + c`.

### Stencil access pattern

For each cell `(r, c)`, the interior kernel reads 5 elements from `temp`:

| Neighbor | Offset from `(r, c)` |
|---|---|
| `temp[r*col + c]` | center: `+0` |
| `temp[r*col + c - 1]` | left: `-1` |
| `temp[r*col + c + 1]` | right: `+1` |
| `temp[(r-1)*col + c]` | up: `-col` |
| `temp[(r+1)*col + c]` | down: `+col` |

This is a classic **5-point stencil** (Von Neumann neighborhood, radius 1).

### Spatial locality analysis

- **Within a row** (`c` is the inner loop): accesses to `temp[r*col + c-1]`, `temp[r*col + c]`, `temp[r*col + c+1]` are contiguous — **good spatial locality**, benefiting from cache lines.
- **Across rows**: accesses to `temp[(r±1)*col + c]` are `col` elements apart. For large `col`, these strides may span multiple cache lines, but the blocking scheme (16×16 tiles) is designed to keep the working set within cache.
- **Write pattern**: `result[r*col + c]` is written exactly once per cell per iteration — pure sequential writes within each tile row, providing **good write locality**.

### Blocking / tiling

The chunk loop (Loop B) tiles the grid into `16 × 16` blocks. The tiling maps a **1D chunk index to a 2D tile coordinate** using division and modulo:

```51:52:/Users/saradordevic/Desktop/rodinia-master/openmp/hotspot/hotspot.cpp
        int r_start = BLOCK_SIZE_R * (chunk / chunks_in_col);
        int c_start = BLOCK_SIZE_C * (chunk % chunks_in_row);
```

Note a subtle issue: the tile traversal order depends on the values of `chunks_in_col` and `chunks_in_row`. The division is by `chunks_in_col` and the modulo is by `chunks_in_row`. If `row != col`, the traversal order may not follow a simple row-major or column-major pattern of tiles, and for non-square grids, the mapping may produce incorrect tile origins (see variable naming — `chunks_in_col` is computed as `row / BLOCK_SIZE_R` but used as a divisor where `chunks_in_row` would be expected for row-major tile ordering, and vice versa).

### Boundary vs. interior access

Boundary chunks access the same neighbor pattern but with asymmetric formulas: edges use a 3-point stencil along the boundary dimension (one neighbor missing), and corners use a 2-point stencil (two neighbors missing). The `amb_temp` global constant substitutes for the missing neighbor in the vertical (z) direction.

---

## 5. Variables That May Cause Race Conditions Under Shared-Memory Parallelism

Although the code as shown has **no OpenMP pragmas** (despite residing in an `openmp` directory), it is structured for potential parallelization of the chunk loop. Here is the analysis of what would happen if chunks were distributed across threads:

### Safe (no race condition)

- **`result[r*col+c]`**: Each cell `(r, c)` belongs to exactly one chunk, so each element of `result` is written by exactly one chunk. No two chunks write to the same location. **Parallelizing the chunk loop would not create write-write conflicts on `result`.**
- **`temp[]`** and **`power[]`**: Read-only within `single_iteration`. No write-read hazard.
- **`Cap_1`, `Rx_1`, `Ry_1`, `Rz_1`, `step`**: Passed by value. Read-only.
- **`amb_temp`**: Global `const`. Read-only.
- **`r_start`, `c_start`, `r_end`, `c_end`**: Local to each chunk iteration, recomputed from `chunk`. Safe.

### Potential race conditions

- **`delta` (line 43)**: Declared as a **function-scoped** variable, not local to the chunk loop body. If the chunk loop (Loop B) were parallelized with `#pragma omp parallel for`, `delta` would be **shared by default** across threads. Multiple threads would read and write `delta` concurrently. This is a **race condition**. It would need to be declared `private` in the OpenMP clause, or moved inside the loop body.

- **`r` and `c` (line 44)**: Similarly declared at function scope. If the chunk loop were parallelized, `r` and `c` would be shared by default. Multiple threads executing different chunks would concurrently modify these loop induction variables. This is a **race condition**. They would need `private` clauses or to be redeclared inside the loop.

- **`chunk` (line 45)**: As the parallelized loop's induction variable, OpenMP would automatically make it private in a `#pragma omp parallel for`. This is safe under standard OpenMP semantics.

### Uninitialized value issue (latent bug)

For boundary chunks, if a cell within the chunk is **not** on any corner or edge of the global grid (i.e., it is an interior cell that happens to fall inside a boundary chunk), none of the 8 `if/else if` branches match. The variable `delta` retains whatever value it held from a previous iteration — or is uninitialized on first encounter. The statement at line 127 then uses this stale/uninitialized `delta`:

```127:127:/Users/saradordevic/Desktop/rodinia-master/openmp/hotspot/hotspot.cpp
                    result[r * col + c] = temp[r * col + c] + delta;
```

This is **undefined behavior** in C/C++. Under parallelism, the effect would be compounded: the stale value of `delta` could come from a different thread's computation.

### Summary table

| Variable | Scope | Access | Race under `parallel for` on chunk loop? |
|---|---|---|---|
| `result[]` | parameter (array) | write (unique per cell) | No |
| `temp[]` | parameter (array) | read-only | No |
| `power[]` | parameter (array) | read-only | No |
| `delta` | function scope | read/write per cell | **Yes** |
| `r` | function scope | loop induction | **Yes** |
| `c` | function scope | loop induction | **Yes** |
| `chunk` | function scope | outer loop induction | No (auto-private by OpenMP) |
| `r_start`, `c_start`, `r_end`, `c_end` | block scope (inside loop) | local per chunk | No |
| `num_chunk`, `chunks_in_row`, `chunks_in_col` | function scope | read-only after init | No |

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

**The chunk loop (Loop B)** — the `for (chunk = 0; chunk < num_chunk; ++chunk)` loop inside `single_iteration` — is the primary candidate for parallelization.

The data dependency analysis from the previous step establishes the precondition: within a single time step, every cell reads from `temp` and writes to `result`, and these are guaranteed to be distinct buffers by the double-buffering pointer swap in `compute_tran_temp`. Since each chunk covers a disjoint tile of the grid, no two chunks ever write to the same element of `result`. Every chunk's computation depends only on `temp` values (read-only) and produces `result` values (write-only, unique per chunk). There are no loop-carried dependencies across chunk iterations, making this loop **fully parallel**.

**The inner cell loops (Loops C and D)** — the `for (r = ...) for (c = ...)` doubly-nested loops within each chunk — are also technically parallelizable for the same reason: each cell writes to a unique location in `result` and reads only from `temp`. However, parallelizing at this level is not a good strategic choice (discussed below under "unit of parallel work").

### Which loops cannot be parallelized

**The time-stepping loop (Loop A)** — `for (int i = 0; i < num_iterations; i++)` — **cannot be parallelized**. There is a true data dependency between consecutive iterations: iteration `i` writes the `result` buffer, the pointer swap makes that buffer become the `temp` input for iteration `i+1`. Every cell's value at time step `i+1` depends on the completed stencil output from time step `i`. This is a **flow dependency (read-after-write)** across the entire grid that cannot be eliminated without changing the algorithm. Time steps must execute sequentially.

### Unit of parallel work

The natural unit of parallel work is **one chunk (a 16×16 tile of 256 cells)**. This is the right granularity for several reasons:

- **Sufficient work per unit**: Each chunk performs 256 stencil computations, each involving 5 reads, several multiply-adds, and 1 write. This provides enough arithmetic to amortize the overhead of thread scheduling/synchronization.
- **Sufficient parallelism**: For a grid of size `N × N`, there are `(N/16)²` chunks. A 1024×1024 grid yields 4,096 chunks — far more than the thread count on any typical shared-memory system, ensuring good load balance.
- **Not too fine-grained**: Parallelizing at the individual cell level (256× more parallel tasks) would introduce excessive scheduling overhead relative to the work per cell (~15 floating-point operations). The chunk granularity naturally batches cells into work units that match cache-line boundaries.
- **No synchronization needed between chunks**: Because chunks write to disjoint regions of `result` and only read from `temp`, no locks, atomics, or barriers are needed between chunks within a time step. The only synchronization point is the implicit barrier at the end of the parallel region, before the pointer swap and the next time step begins.

### Type of parallelism

This represents **data parallelism**, specifically the **geometric decomposition** (also called domain decomposition) pattern.

The same computation — a 5-point stencil update with boundary-condition handling — is applied uniformly to every cell in a 2D grid. The grid is spatially partitioned into tiles, and each tile undergoes the identical operation on its local subset of the data. There is no functional differentiation between work units (it is not task parallelism), no pipeline of stages (it is not pipeline parallelism), and no speculative or conditional parallel execution. It is the same operation replicated across spatial partitions of a data domain.

The outer sequential time-stepping loop wrapping the parallel spatial computation constitutes a **bulk-synchronous** execution model: parallel phase (all chunks compute), barrier, sequential bookkeeping (pointer swap), repeat. This is sometimes called the **iterative data-parallel** pattern, common in stencil codes and explicit time-stepping PDE solvers.

### How memory access patterns affect the decision

**Spatial locality within chunks favors chunk-level parallelism.** Within a chunk, the inner loop sweeps `c` (the column index) in the innermost position. Because the arrays use row-major layout, consecutive `c` values map to consecutive memory addresses. A cache line (typically 64 bytes = 16 floats) loaded for `temp[r*col + c]` also satisfies the reads for `temp[r*col + c-1]` and `temp[r*col + c+1]`. The 16×16 block size is specifically chosen so that one tile row (16 floats = 64 bytes) fits exactly in one cache line, and the full 16-row tile has a small enough footprint to remain in L1 cache.

**Stencil halos create read sharing at tile boundaries.** A chunk reading cell `(r, c)` at the left edge of its tile also reads `temp[r*col + c-1]`, which belongs to the adjacent chunk's tile. This means neighboring chunks share **overlapping read regions** (a 1-cell-wide halo). In a shared-memory model, this is not a correctness issue — `temp` is read-only, and the shared cache hierarchy handles concurrent reads naturally. However, it does mean that threads working on adjacent tiles will load partially overlapping cache lines, creating **cache-line sharing**. On NUMA systems or with poor thread-to-core mapping, this could lead to false sharing on cache-line boundaries between tiles. The 16-wide tile alignment to cache-line boundaries mitigates this: the halo reads (±1 in the column direction) only touch the first/last element of an adjacent tile's cache line, and since those lines are read-only, the coherence protocol only needs to keep them in Shared state — no invalidation traffic occurs.

**The vertical stencil arms (`±col` stride) affect prefetch behavior.** The reads to `temp[(r±1)*col + c]` access addresses that are `col` elements (i.e., `col × 4` bytes) away. For large grids, this stride exceeds the L1 cache size, making these accesses potential cache misses. The tiling strategy helps: when a chunk processes rows `r` through `r+15`, the values `temp[(r+1)*col + c]` read for row `r` are the same as `temp[r*col + c]` read when processing row `r+1`, so they remain in cache from one row to the next within the same chunk. The tile height (16 rows) keeps the vertical working set — 3 rows of 16 floats each (previous, current, next) plus the corresponding `power` and `result` lines — within L1 capacity. This **temporal reuse** within a tile is precisely why the code is blocked, and it reinforces the decision to parallelize at the chunk level rather than, say, distributing individual rows across threads (which would destroy this intra-tile reuse).

**Write exclusivity makes synchronization-free parallelism possible.** Because each chunk writes to a disjoint 16×16 region of `result` and these regions are tile-aligned, there is no write contention. No atomics, critical sections, or reductions are needed. This is the most favorable memory access pattern for shared-memory parallelism — the combination of read-only input sharing and write-exclusive output partitioning eliminates all coherence-related serialization.

---

**User**

Implement the parallelization strategy you proposed using OpenMP. Modify the given code. Explain every OpenMP directive you add.

---

**Cursor**

Now I'll add the OpenMP directive to the chunk loop and fix the variable scoping issues. The variables `delta`, `r`, and `c` are declared at function scope and must become private to each thread.

Now let me also add `omp_set_num_threads` in `main` so the thread count argument is actually used instead of being ignored.

Let me verify the final state of the file to confirm everything is correct.

No linter errors. Here is a full explanation of every change and every OpenMP directive.

---

## Changes Made and Explanation

### 1. `#include <omp.h>` (line 4)

Provides the OpenMP runtime API declarations, needed for `omp_set_num_threads()` and for the compiler to recognize `#pragma omp` directives.

### 2. `omp_set_num_threads(atoi(argv[4]))` (line 275)

The original code accepted a thread-count argument but marked it as "ignored." This call configures the OpenMP runtime to use the user-specified number of threads for all subsequent parallel regions. It is placed after argument validation so the value is guaranteed to be a positive integer.

### 3. `#pragma omp parallel for schedule(static)` (line 48)

This is the single parallelization directive and the core of the change. It is placed immediately before the chunk loop. Here is what each clause means:

- **`parallel`** — Creates a team of threads. Each thread gets its own stack and execution context. The team is created once when execution reaches this point and destroyed (joined) at the closing brace of the `for` loop. An implicit barrier at the end ensures all chunks are completed before execution continues.

- **`for`** — Distributes the iterations of the immediately following `for` loop across the thread team. Each thread gets a subset of the `chunk` index range `[0, num_chunk)`. This is a worksharing construct — it does not create threads itself, it partitions work among threads already created by `parallel`.

- **`schedule(static)`** — Divides the iteration space into contiguous blocks of approximately equal size (`num_chunk / num_threads` iterations each) and assigns them to threads at compile time in round-robin order. Static scheduling is chosen because every chunk performs nearly identical work (256 stencil computations), so no dynamic load-balancing is needed. Static scheduling also produces the most predictable and cache-friendly memory access pattern: each thread works on a contiguous range of tiles, which maps to a contiguous region of the grid.

### 4. Variable scoping — eliminating race conditions

The original code declared `delta`, `r`, `c`, and `chunk` at function scope (lines 43–45 of the original). Under `#pragma omp parallel for`, the loop induction variable `chunk` would be automatically privatized, but `delta`, `r`, and `c` would default to **shared** — causing race conditions.

The fix was to **move all variable declarations into the narrowest possible scope**:

- **`chunk`** (line 49): Declared as `int chunk` in the `for` header itself (`for (int chunk = ...)`). OpenMP automatically makes the loop induction variable of a workshared `for` private to each thread.

- **`r_start`, `c_start`, `r_end`, `c_end`** (lines 50–53): Already declared inside the loop body in the original code. Since they are declared within the parallel region's dynamic extent, each thread has its own copy on its stack. No change needed.

- **`r` and `c`** (lines 56, 57, 144, 145): Changed from function-scope `int r, c;` to declarations inside each `for` header (`for (int r = ...)`, `for (int c = ...)`). This makes them local variables on each thread's stack, eliminating the race where multiple threads would have shared a single `r` and `c`.

- **`delta`** (line 58): Moved from function scope into the innermost loop body of the boundary path, declared at line 58 as `FLOAT delta;` within the `for (int c = ...)` block. Each thread (and each iteration) now has its own independent `delta`. This eliminates the race condition **and** fixes the pre-existing latent bug described below.

### 5. Added `else` clause — fixing the uninitialized `delta` bug (lines 126–136)

This is a correctness fix, not strictly an OpenMP change, but it was exposed and made worse by parallelization. In the original code, boundary chunks contain a mix of cells that actually lie on the grid boundary and cells that are interior but happen to be in a boundary-adjacent tile. The 8-way `if/else if` chain (4 corners + 4 edges) had **no `else` branch**. Interior cells within boundary chunks fell through all conditions, leaving `delta` unset. The original code then used this uninitialized value at line 127:

```
result[r * col + c] = temp[r * col + c] + delta;
```

This was undefined behavior in the original serial code. Under parallelism, the stale `delta` could come from a different thread's computation.

The fix adds an `else` branch that applies the standard 5-point interior stencil — the same formula used in the non-boundary chunk path (lines 146–156). This ensures `delta` is always assigned before use:

```126:136:/Users/saradordevic/Desktop/rodinia-master/openmp/hotspot/hotspot.cpp
                    } else {
                        delta =
                            (Cap_1 *
                             (power[r * col + c] +
                              (temp[(r + 1) * col + c] + temp[(r - 1) * col + c] -
                               2.f * temp[r * col + c]) *
                                  Ry_1 +
                              (temp[r * col + c + 1] + temp[r * col + c - 1] -
                               2.f * temp[r * col + c]) *
                                  Rx_1 +
                              (amb_temp - temp[r * col + c]) * Rz_1));
                    }
```

### What was **not** parallelized and why

The **time-stepping loop** in `compute_tran_temp` (line 187) remains sequential. Iteration `i+1` reads from the output of iteration `i` (after the pointer swap). This is a true flow dependency across the entire grid — no cell at step `i+1` can be computed until all cells at step `i` are complete. The implicit barrier at the end of the `#pragma omp parallel for` region provides exactly this guarantee: all chunks finish before `single_iteration` returns, the pointer swap happens on the single master thread, and then the next call to `single_iteration` re-enters the parallel region.

