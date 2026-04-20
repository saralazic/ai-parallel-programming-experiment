# GPU parallelization analysis of algorithm
_Exported on 3/21/2026 at 13:58:08 GMT+1 from Cursor (2.4.31)_

---

**User**

This code is a sequential implementation of an algorithm.
Analyze and identify which parts of the algorithm can be parallelized on GPU and provide detailed explanation:
1. Which computations are parallelizable and why
2. Data dependencies that may limit parallelization
3. Parallelization strategy for GPU execution
4. Expected performance challenges when mapping this algorithm to GPU architecture

---

**Cursor**

Let me first explore the project structure and understand the codebase.

This is the **HotSpot** thermal simulation — a classic 2D stencil benchmark from the Rodinia suite. It solves discretized heat transfer equations iteratively over a chip grid. Here is a detailed GPU parallelization analysis.

---

## 1. Which Computations Are Parallelizable and Why

### The Inner Stencil Computation (Highly Parallelizable)

The core of this program is the `single_iteration` function. Within a **single iteration**, every cell `(r, c)` computes its new temperature based solely on **read-only** access to the *current* temperature array `temp[]` and the power array `power[]`. The result is written to a separate output array `result[]`.

```133:148:hotspot-seq.cpp
        for (r = r_start; r < r_start + BLOCK_SIZE_R; ++r) {
            for (c = c_start; c < c_start + BLOCK_SIZE_C; ++c) {
                /* Update Temperatures */
                result[r * col + c] =
                    temp[r * col + c] +
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
        }
```

This is a **5-point stencil**: each cell reads itself, its north/south/east/west neighbors, plus its corresponding power value. Because every cell reads from `temp[]` and writes to a *distinct* location in `result[]`, **all `row × col` cells within one iteration are embarrassingly parallel** — there are no write conflicts and no read-after-write dependencies between cells in the same iteration.

### The Boundary/Edge Cases (Also Parallelizable)

The boundary handling (lines 56–130) computes the same stencil but with modified neighbor accesses for corners and edges (reflecting boundary conditions). These are also parallelizable — they just require conditional logic per thread based on position. Each boundary cell still reads only from `temp[]` and writes only to its own slot in `result[]`.

### The Parameter Computation (Trivially Parallelizable / Host-side)

```158:172:hotspot-seq.cpp
    FLOAT grid_height = chip_height / row;
    FLOAT grid_width = chip_width / col;

    FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    FLOAT Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    FLOAT Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    FLOAT Rz = t_chip / (K_SI * grid_height * grid_width);

    FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    FLOAT step = PRECISION / max_slope / 1000.0;

    FLOAT Rx_1 = 1.f / Rx;
    FLOAT Ry_1 = 1.f / Ry;
    FLOAT Rz_1 = 1.f / Rz;
    FLOAT Cap_1 = step / Cap;
```

This is a small constant-time computation. It should remain on the CPU and the results passed to the GPU as kernel arguments or constant memory.

### I/O (`read_input`, `writeoutput`) — Not Parallelizable on GPU

These are file I/O operations that are inherently sequential and host-bound.

---

## 2. Data Dependencies That Limit Parallelization

### Inter-Iteration (Temporal) Dependency — The Critical Bottleneck

```177:183:hotspot-seq.cpp
        for (int i = 0; i < num_iterations; i++) {
            single_iteration(r, t, power, row, col, Cap_1, Rx_1, Ry_1, Rz_1,
                             step);
            FLOAT *tmp = t;
            t = r;
            r = tmp;
        }
```

Each iteration `i` requires the **complete result** of iteration `i-1` as its input. The pointer swap (`t = r; r = tmp;`) means iteration `i+1` reads from what iteration `i` just wrote. This is a **strict temporal dependency** — you cannot start iteration `i+1` until iteration `i` is fully complete. This means:

- **Within** one iteration: fully parallel (all cells independent).
- **Across** iterations: strictly sequential — each iteration is a global synchronization barrier.

This limits the total parallelism to `row × col` threads per iteration, with `num_iterations` serial kernel launches.

### Intra-Iteration Spatial (Stencil) Dependency

Within a single iteration there is no true write dependency (each cell writes only to its own output location). However, there is a **read pattern dependency**: each cell reads from its 4 neighbors. This means:

- Threads in adjacent positions will read overlapping data (e.g., thread at `(r,c)` and thread at `(r,c+1)` both read `temp[r*col+c]`).
- This doesn't prevent parallelism, but it heavily influences **memory access patterns** and the **shared memory tiling strategy** (see below).

### Power Array — Read-Only, No Dependency

`power[]` is never modified. It is purely read-only across all iterations, making it an ideal candidate for GPU **texture memory** or **constant caching**.

---

## 3. Parallelization Strategy for GPU Execution

### Thread Mapping

Map one GPU thread to one grid cell `(r, c)`:

```
threadIdx.x → column within a tile
threadIdx.y → row within a tile
blockIdx.x  → tile column index
blockIdx.y  → tile row index
```

For a grid of size `row × col`, launch a 2D grid of `(col/BLOCK_SIZE_C) × (row/BLOCK_SIZE_R)` blocks, each of size `BLOCK_SIZE_C × BLOCK_SIZE_R` threads. The existing block decomposition in the sequential code (using `BLOCK_SIZE = 16`) maps directly to a 16×16 CUDA thread block — which is a natural fit since 16×16 = 256 threads per block, a sensible occupancy target.

### Shared Memory Tiling (Critical Optimization)

Because this is a stencil, each thread reads neighbors that are also read by adjacent threads. The strategy:

1. **Allocate shared memory** of size `(BLOCK_SIZE_R + 2) × (BLOCK_SIZE_C + 2)` per block — the "+2" accounts for the 1-cell-wide halo/ghost zone around the tile.
2. **Cooperative loading**: all threads in a block cooperatively load the tile plus its halo from global memory into shared memory.
3. **Synchronize** (`__syncthreads()`).
4. **Compute** the stencil from shared memory instead of global memory.

This reduces global memory bandwidth from ~5 reads per cell to ~1 read per cell (amortized), since neighbors are reused from shared memory.

### Boundary Handling

The branching for corners/edges (lines 56–130) should be folded into the kernel using conditional logic on `(r, c)`. On GPU, these boundary cells will cause some **warp divergence**, but since boundary cells are a tiny fraction of the total grid (perimeter vs. area), the impact is negligible for large grids.

### Iterative Kernel Launch

The outer loop over `num_iterations` becomes a loop of kernel launches on the host:

```c
for (int i = 0; i < num_iterations; i++) {
    hotspot_kernel<<<gridDim, blockDim>>>(d_result, d_temp, d_power, row, col, ...);
    // swap d_temp and d_result pointers (on host side)
    float *tmp = d_temp; d_temp = d_result; d_result = tmp;
}
```

No data needs to be transferred back to the host between iterations — only the device pointers are swapped on the host side.

### Memory Placement Summary

| Data | GPU Memory Type | Rationale |
|------|----------------|-----------|
| `temp[]` | Global memory (+ shared memory tile with halo) | Read by stencil, too large for constant memory |
| `result[]` | Global memory | Write-only output per iteration |
| `power[]` | Global memory or texture memory | Read-only across all iterations, benefits from spatial locality caching |
| `Cap_1, Rx_1, Ry_1, Rz_1, amb_temp` | Constant memory or kernel arguments | Small, uniform, read by all threads identically |

---

## 4. Expected Performance Challenges

### A. Kernel Launch Overhead for Many Iterations

Each iteration requires a separate kernel launch because of the global synchronization (all cells must finish before the next iteration reads). For very high `num_iterations` with a small grid, kernel launch overhead can dominate. Mitigation: **persistent kernel** techniques or **cooperative groups** (`cudaLaunchCooperativeKernel`) that allow grid-wide synchronization within a single kernel launch.

### B. Memory Bandwidth Bound

The stencil computation has a low **arithmetic intensity** (only ~7 FLOPs per cell but 5 floats read + 1 float written = 24 bytes). This yields an arithmetic intensity of roughly 7/24 ≈ 0.29 FLOPs/byte, which is well below the compute-to-bandwidth ratio of any modern GPU. The kernel will be **memory-bandwidth bound**, not compute-bound. Mitigations:
- Shared memory tiling (reduces redundant global loads).
- **Temporal blocking** (fuse multiple iterations into one kernel pass so intermediate results live in shared memory / registers).

### C. Warp Divergence at Boundaries

The boundary logic (corners, edges) introduces branching:

```56:56:hotspot-seq.cpp
        if (r_start == 0 || c_start == 0 || r_end == row || c_end == col) {
```

Threads in boundary blocks will follow different code paths. Since boundary cells are concentrated in specific blocks (perimeter of the grid), only those blocks suffer divergence — interior blocks take a uniform code path. For large grids (e.g., 1024×1024), this affects only ~0.4% of blocks, so the impact is minimal.

### D. Shared Memory Bank Conflicts

With a tile width of 16 + 2 = 18 floats, accesses to shared memory when threads in a warp read consecutive columns will hit bank conflicts (18 is not a multiple of 32 banks, but column-wise access patterns in a row-major layout can still cause 2-way conflicts). Padding the shared memory array to `(BLOCK_SIZE_C + 2 + 1)` width can eliminate this.

### E. Host-Device Transfer Latency

The `temp[]`, `power[]`, and `result[]` arrays must be transferred to the GPU before computation and back after all iterations complete. For a 1024×1024 grid of floats, this is only ~4 MB per array — trivially fast compared to computation time for many iterations. But for very few iterations on small grids, the transfer cost can exceed the computation time, making GPU execution counterproductive.

### F. Occupancy and Block Size Tuning

The 16×16 = 256 threads per block is a reasonable starting point but may not maximize occupancy depending on the specific GPU's register file and shared memory capacity. The halo-augmented shared memory tile of `(18 × 18 × 4 bytes) = 1,296 bytes` per block is very small, so shared memory won't be the occupancy limiter — register pressure from the stencil computation might be, though the computation is simple enough that this is unlikely.

---

## Summary

| Aspect | Assessment |
|--------|-----------|
| **Parallelism within one iteration** | Fully parallel — all `row × col` cells are independent |
| **Parallelism across iterations** | None — strict sequential dependency |
| **Dominant bottleneck** | Memory bandwidth (low arithmetic intensity stencil) |
| **Best GPU mapping** | 2D thread blocks with shared memory tiling + halo |
| **Key optimization** | Temporal blocking to fuse iterations and reuse data in shared memory |
| **Boundary handling** | Minor warp divergence, negligible for large grids |
| **Launch strategy** | One kernel launch per iteration, pointer swap on host |

The algorithm is a textbook candidate for GPU parallelization. The 2D stencil pattern maps naturally to CUDA's grid/block model, and the existing `BLOCK_SIZE = 16` decomposition in the sequential code already mirrors the intended GPU tiling. The main challenges are bandwidth optimization (shared memory tiling, temporal blocking) and managing the serial iteration dependency.

---

**User**

Based on your analysis of code and proposed strategy GPU paralelization:
1. Implement CUDA kernel(s)
2. Organize threads into blocks and grids appropriately
3. Manage memory transfers between host (CPU) and device (GPU)
4. Use suitable CUDA memory types (global, shared, local)

---

**Cursor**

Now I have a thorough understanding of the sequential code. Let me implement the CUDA version.

Let me verify the file was written correctly and review the complete implementation.

The file is clean. Here is a walkthrough of every design decision in `hotspot-cuda.cu`.

---

## 1. CUDA Kernel (`hotspot_kernel`) — lines 47–101

### Thread-to-cell mapping

Each thread computes exactly one grid cell. The 2D thread index maps directly to a `(row, col)` position in the chip grid:

```89:100:hotspot-cuda.cu
    if (r < row && c < col) {
        FLOAT center = temp_s[sy][sx];
        result[r * col + c] =
            center +
            Cap_1 * (power[r * col + c] +
                     (temp_s[sy + 1][sx] + temp_s[sy - 1][sx] -
                      2.f * center) * Ry_1 +
                     (temp_s[sy][sx + 1] + temp_s[sy][sx - 1] -
                      2.f * center) * Rx_1 +
                     (amb_temp_val - center) * Rz_1);
    }
```

### Shared memory with halo

A `(BLOCK_SIZE+2) x (BLOCK_SIZE+2)` = `18 x 18` shared memory tile stores the block's cells plus a 1-cell-wide halo ring. The stencil reads all five values (center + 4 neighbors) from fast shared memory instead of slow global memory:

```52:52:hotspot-cuda.cu
    __shared__ FLOAT temp_s[BLOCK_SIZE_R + 2][BLOCK_SIZE_C + 2];
```

### Halo loading with boundary clamping

Instead of the sequential code's 8-way if/else chain for corners and edges, the kernel uses `min`/`max` index clamping. A missing neighbor (e.g., north of row 0) is replaced by the cell's own value, which yields the same zero-gradient Neumann boundary condition:

```74:85:hotspot-cuda.cu
    /* --- Load the 1-cell halo ring from global memory ------------------- */
    if (ty == 0)
        temp_s[0][sx] = temp[max(cr - 1, 0) * col + cc];

    if (ty == BLOCK_SIZE_R - 1)
        temp_s[BLOCK_SIZE_R + 1][sx] = temp[min(cr + 1, row - 1) * col + cc];

    if (tx == 0)
        temp_s[sy][0] = temp[cr * col + max(cc - 1, 0)];

    if (tx == BLOCK_SIZE_C - 1)
        temp_s[sy][BLOCK_SIZE_C + 1] = temp[cr * col + min(cc + 1, col - 1)];
```

Only the border threads of each block (first/last row, first/last column within the tile) load the halo. All threads then synchronize before the stencil computation reads shared memory.

### Out-of-bounds thread handling

When `row` or `col` isn't a multiple of 16, the last blocks have extra threads. Instead of an early `return` (which would break `__syncthreads()`), those threads clamp their coordinates and harmlessly load duplicate data into shared memory. Only the final stencil write is guarded by a bounds check:

```68:69:hotspot-cuda.cu
    int cr = min(r, row - 1);
    int cc = min(c, col - 1);
```

---

## 2. Grid and Block Organization — lines 138–141

The block size is `16 x 16 = 256` threads, matching the sequential code's `BLOCK_SIZE`. The grid is sized to cover the entire chip grid with ceiling division:

```138:141:hotspot-cuda.cu
    /* ---- Kernel launch configuration ----------------------------------- */
    dim3 blockDim(BLOCK_SIZE_C, BLOCK_SIZE_R);
    dim3 gridDim((col + BLOCK_SIZE_C - 1) / BLOCK_SIZE_C,
                 (row + BLOCK_SIZE_R - 1) / BLOCK_SIZE_R);
```

For a 1024x1024 chip grid this produces a `64 x 64 = 4096` block grid, each with 256 threads — totaling 1,048,576 threads per kernel launch.

---

## 3. Memory Transfers (Host <-> Device) — lines 126–167

The flow is:

1. **Allocate** three device buffers (`d_temp`, `d_result`, `d_power`) via `cudaMalloc`
2. **Upload** the initial temperature and power arrays once before the iteration loop
3. **Iterate** entirely on the GPU — no mid-loop transfers, only device pointer swaps on the host side
4. **Download** the final result once after all iterations complete
5. **Free** all device memory

```126:167:hotspot-cuda.cu
    /* ---- Device memory allocation -------------------------------------- */
    size_t size = (size_t)row * col * sizeof(FLOAT);
    FLOAT *d_temp, *d_result, *d_power;

    CUDA_CHECK(cudaMalloc(&d_temp,   size));
    CUDA_CHECK(cudaMalloc(&d_result, size));
    CUDA_CHECK(cudaMalloc(&d_power,  size));

    /* ---- Host → Device transfers --------------------------------------- */
    CUDA_CHECK(cudaMemcpy(d_temp,  temp,  size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_power, power, size, cudaMemcpyHostToDevice));
    // ... kernel loop ...
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ---- Device → Host transfer ---------------------------------------- */
    CUDA_CHECK(cudaMemcpy(result, d_src, size, cudaMemcpyDeviceToHost));
    memcpy(temp, result, size);

    /* ---- Free device memory -------------------------------------------- */
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_power));
```

The `d_power` array is uploaded once and never modified — it acts as read-only global memory across all iterations.

---

## 4. CUDA Memory Types Used

| Memory type | What's stored | Why |
|---|---|---|
| **Global memory** | `d_temp`, `d_result`, `d_power` — the full chip-grid arrays | Required for data that doesn't fit in on-chip memory. `d_power` is read-only and benefits from L1/L2 caching. |
| **Shared memory** | `temp_s[18][18]` — the tile + halo for temperatures | Each stencil point reads 5 values, and adjacent threads share 4 of them. Loading through shared memory reduces redundant global memory traffic from ~5x to ~1x per cell. |
| **Registers (local)** | `center`, `delta`, loop indices, clamped coords | Scalar temporaries private to each thread — fastest access. |
| **Kernel arguments** (constant bank) | `Cap_1`, `Rx_1`, `Ry_1`, `Rz_1`, `amb_temp_val`, `row`, `col` | Uniform scalars broadcast to all threads. Passed as kernel arguments, which nvcc places in constant memory / registers. |

---

## Compilation

To build the CUDA version:

```bash
nvcc -O3 -o hotspot-cuda hotspot-cuda.cu
```

The command-line interface is identical to the sequential version:

```bash
./hotspot-cuda <grid_rows> <grid_cols> <sim_time> <threads> <temp_file> <power_file> <output_file>
```

---

**User**

Explain 
1. how this CUDA solution maps onto GPU hardware architecture.
2. How threads and thread blocks are organized
3. How memory hierarchy is used
4. What performance limitations may exist

---

**Cursor**

Here is a detailed explanation of all four aspects.

---

## 1. How This CUDA Solution Maps onto GPU Hardware Architecture

A GPU is built from an array of **Streaming Multiprocessors (SMs)**. Each SM can run multiple thread blocks concurrently, and contains its own shared memory, register file, warp schedulers, and execution units. This kernel maps onto that architecture as follows:

**Problem domain → Hardware mapping:**

| Concept in code | GPU hardware unit |
|---|---|
| One thread (one cell `(r,c)`) | One CUDA core (SP) execution lane |
| One thread block (16x16 = 256 threads) | Scheduled on one SM |
| The full grid of blocks | Distributed across all SMs on the chip |
| `__shared__ temp_s[18][18]` | Allocated in the SM's on-chip shared memory |
| `Cap_1, Rx_1, Ry_1, Rz_1`, loop variables | Stored in each thread's private registers within the SM's register file |
| `d_temp, d_result, d_power` arrays | Reside in off-chip DRAM (global memory), accessed through the L2 cache and per-SM L1 cache |

**Execution model:** When the kernel is launched on line 148, the CUDA runtime distributes the grid of blocks across all available SMs. Each SM picks up a block, allocates shared memory and registers for it, splits its 256 threads into **warps** of 32 threads, and begins executing those warps. When a warp stalls (e.g., waiting on a global memory load), the SM's warp scheduler instantly switches to another ready warp — this **latency hiding** is the fundamental performance mechanism the kernel relies on.

**One kernel launch = one time step.** The iteration loop on line 147 launches the kernel `num_iterations` times. Between launches, the entire GPU acts as an implicit global barrier — iteration `i+1` cannot begin until every thread of iteration `i` has completed. The pointer swap (lines 152–154) happens purely on the host and costs nothing.

---

## 2. How Threads and Thread Blocks Are Organized

### Block dimensions

```139:139:hotspot-cuda.cu
    dim3 blockDim(BLOCK_SIZE_C, BLOCK_SIZE_R);
```

Each block is a **2D 16x16 tile** = 256 threads. This means:
- `threadIdx.x` ranges 0–15 (column direction)
- `threadIdx.y` ranges 0–15 (row direction)

256 threads per block gives **8 warps** per block (256 / 32). A warp is formed by linearizing the 2D index: `warpId = (ty * 16 + tx) / 32`. So warp 0 covers rows 0–1 (all 16 columns each), warp 1 covers rows 2–3, etc. This means threads in the same warp access **adjacent columns in the same two rows** — which is important for memory coalescing (see below).

### Grid dimensions

```140:141:hotspot-cuda.cu
    dim3 gridDim((col + BLOCK_SIZE_C - 1) / BLOCK_SIZE_C,
                 (row + BLOCK_SIZE_R - 1) / BLOCK_SIZE_R);
```

Ceiling division ensures every cell is covered even when dimensions aren't multiples of 16. For a **1024x1024** chip grid:
- `gridDim = (64, 64)` = **4,096 blocks**
- Total threads: 4,096 x 256 = **1,048,576**

For a typical GPU with ~80 SMs, each SM would handle ~51 blocks (not concurrently — they are time-multiplexed). The number of blocks that can run **concurrently** per SM depends on shared memory and register usage (see section 4).

### Thread-to-cell mapping

```56:57:hotspot-cuda.cu
    int c  = blockIdx.x * BLOCK_SIZE_C + tx;
    int r  = blockIdx.y * BLOCK_SIZE_R + ty;
```

Each thread maps to exactly one cell `(r, c)` in the chip grid. `blockIdx.x` selects the tile column, `blockIdx.y` selects the tile row. Within the tile, `threadIdx.x/y` selects the local position. This 2D mapping preserves spatial locality — threads in the same block process a contiguous 16x16 patch of the chip, which is exactly the region loaded into shared memory.

### Out-of-bounds threads

```68:69:hotspot-cuda.cu
    int cr = min(r, row - 1);
    int cc = min(c, col - 1);
```

Rather than returning early (which would be illegal before `__syncthreads()` — it would cause a deadlock since `__syncthreads()` requires **all** threads in the block to reach the barrier), out-of-bounds threads clamp their coordinates and load duplicated, harmless data. Only the final write is guarded:

```90:90:hotspot-cuda.cu
    if (r < row && c < col) {
```

---

## 3. How the Memory Hierarchy Is Used

The GPU memory hierarchy has five levels, from slowest/largest to fastest/smallest. Here's how each is used:

### Level 1: Global Memory (DRAM) — off-chip, ~800 GB/s, ~400 cycle latency

Three arrays live here:

```130:132:hotspot-cuda.cu
    CUDA_CHECK(cudaMalloc(&d_temp,   size));
    CUDA_CHECK(cudaMalloc(&d_result, size));
    CUDA_CHECK(cudaMalloc(&d_power,  size));
```

- **`d_temp`** (read source): Every thread reads its own cell + 4 neighbors through shared memory, which batches the global loads.
- **`d_result`** (write destination): Each thread writes exactly one float. Since threads in a warp process adjacent columns (tx = 0..15 with same ty), their writes are to consecutive addresses `result[r*col + c]` for adjacent `c`. This gives **fully coalesced writes** — the hardware merges them into a single wide transaction.
- **`d_power`** (read-only): Uploaded once on line 136 and never modified. Each thread reads one float. Again, adjacent threads read adjacent addresses, giving coalesced reads. Since it's never written, the L1/L2 caches are highly effective.

### Level 2: L2 Cache — on-chip, shared across all SMs, ~3–6 MB

The L2 cache automatically caches global memory accesses. Since `d_power` is read-only and the same `d_temp` data is read by overlapping halos of neighboring blocks, the L2 provides significant implicit reuse. No code changes needed — this is transparent hardware caching.

### Level 3: L1 Cache / Shared Memory — on-chip per-SM, ~100 KB, ~30 cycle latency

This is where the explicit optimization lives:

```52:52:hotspot-cuda.cu
    __shared__ FLOAT temp_s[BLOCK_SIZE_R + 2][BLOCK_SIZE_C + 2];
```

This allocates `18 x 18 x 4 bytes = 1,296 bytes` per block on the SM's shared memory. The loading proceeds in two phases:

**Phase 1 — Cooperative tile load** (all 256 threads participate):

```72:85:hotspot-cuda.cu
    temp_s[sy][sx] = temp[cr * col + cc];
    // ... halo loads by border threads ...
```

Each thread loads its own cell (256 loads). The 4 edges of threads additionally load the halo ring: top row (`ty==0` loads 16 cells), bottom row (`ty==15` loads 16 cells), left column (`tx==0` loads 16 cells), right column (`tx==15` loads 16 cells). Total: 256 + 64 = **320 global loads** to fill the 324-element shared array (the 4 corners of the halo are unused by the stencil since there are no diagonal accesses).

**Phase 2 — Stencil computation** (after `__syncthreads()`):

```91:99:hotspot-cuda.cu
        FLOAT center = temp_s[sy][sx];
        result[r * col + c] =
            center +
            Cap_1 * (power[r * col + c] +
                     (temp_s[sy + 1][sx] + temp_s[sy - 1][sx] -
                      2.f * center) * Ry_1 +
                     (temp_s[sy][sx + 1] + temp_s[sy][sx - 1] -
                      2.f * center) * Rx_1 +
                     (amb_temp_val - center) * Rz_1);
```

Each thread reads 5 values from shared memory (~5 cycle latency each vs ~400 from global). Without shared memory, each of those 5 reads would be a global memory access. With shared memory, **4 of 5 neighbor reads are reused** from data loaded by adjacent threads — reducing global bandwidth demand by roughly 4x for the temperature array.

### Level 4: Registers — per-thread, ~64K per SM, ~1 cycle latency

The fastest storage. The compiler automatically places scalar locals here:

- `center` (cached from shared memory for reuse in the stencil)
- `tx`, `ty`, `sx`, `sy`, `r`, `c`, `cr`, `cc`
- Intermediate arithmetic results
- Kernel arguments `Cap_1`, `Rx_1`, `Ry_1`, `Rz_1`, `amb_temp_val`

The kernel is simple enough that register pressure is very low — likely ~16–20 registers per thread. With 256 threads/block, that's ~5,120 registers per block, well within the 65,536 registers available per SM on modern GPUs.

### Level 5: Constant Memory (kernel arguments)

```148:150:hotspot-cuda.cu
        hotspot_kernel<<<gridDim, blockDim>>>(
            d_dst, d_src, d_power,
            row, col, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp);
```

The 10 kernel arguments are placed in constant memory by the CUDA runtime. Since every thread reads the same values (they are **uniform** across all threads), the hardware broadcasts them in a single cycle to all threads in a warp. This is effectively free.

### Memory transfer pattern (host <-> device)

```
Host RAM ──cudaMemcpy──→ Device Global Memory  (once, before iteration loop)
           d_temp ←→ d_result                   (pointer swap only, no data movement)
Device Global Memory ──cudaMemcpy──→ Host RAM   (once, after iteration loop)
```

The key optimization: **zero host-device transfers during the iteration loop**. Only device pointers are swapped on lines 152–154, which is a host-side register operation with no data movement.

---

## 4. Performance Limitations

### A. Memory Bandwidth Bound (the dominant bottleneck)

The stencil computation does very little arithmetic relative to the data it loads:

- **Reads per cell**: 5 floats from `temp_s` (shared) + 1 float from `power` (global) = 6 reads (but only ~2 effective global reads per cell after shared memory tiling: the cell itself + its share of halo loads)
- **Writes per cell**: 1 float to `result` (global)
- **FLOPs per cell**: ~13 (3 subtractions, 3 multiplications for the three terms, 3 additions, some scaling)

**Arithmetic intensity**: ~13 FLOPs / (3 floats x 4 bytes) = ~1.1 FLOPs/byte. Modern GPUs (e.g., A100) have a compute-to-bandwidth ratio of ~150 TFLOPS / 2 TB/s = ~75 FLOPs/byte. The kernel needs 75 but provides 1.1 — it is **~68x below the compute/bandwidth equilibrium**. The SMs' compute units will be mostly idle, waiting for data from memory. This is inherent to stencil algorithms.

### B. Kernel Launch Overhead

```147:155:hotspot-cuda.cu
    for (int i = 0; i < num_iterations; i++) {
        hotspot_kernel<<<gridDim, blockDim>>>(
            d_dst, d_src, d_power,
            row, col, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp);

        FLOAT *tmp = d_src;
        d_src = d_dst;
        d_dst = tmp;
    }
```

Each iteration requires a separate kernel launch (~5–10 microseconds overhead each). For `num_iterations = 1000`, that's 5–10ms of pure overhead. On small grids where each kernel execution takes only microseconds itself, launch overhead can be a significant fraction of total time. Mitigation would require **temporal blocking** (fusing multiple iterations into one kernel launch so intermediate data stays in shared memory/registers).

### C. Global Synchronization Between Iterations

The inter-iteration dependency — iteration `i+1` reads what iteration `i` wrote — requires a full **device-wide barrier** between kernel launches. There is no CUDA primitive for a grid-wide barrier within a single kernel (except cooperative groups on limited grid sizes). This means the GPU must fully drain all blocks of iteration `i` before any block of iteration `i+1` can start. This creates a "sawtooth" pattern in SM utilization as the last blocks finish.

### D. Occupancy Considerations

**Shared memory per block**: 1,296 bytes. Modern SMs have ~100 KB of configurable shared memory, so shared memory allows up to ~79 blocks per SM — far more than the hardware can actually schedule.

**Registers per block**: ~16 registers x 256 threads = ~4,096 registers. With 65,536 registers per SM, this allows ~16 concurrent blocks.

**Warp limit**: Most SMs support up to 32–64 warps. Each block has 8 warps, so 4–8 concurrent blocks per SM from this limit.

The effective occupancy is likely **4–8 resident blocks per SM**, dictated by the warp limit. This is healthy — it gives the warp scheduler 32–64 warps to choose from for latency hiding. But since the kernel is memory-bound, even perfect occupancy won't help if global memory bandwidth is saturated.

### E. Warp Divergence (minor)

The boundary handling uses `min`/`max` clamping (lines 68–69, 75–85) instead of branches, so it doesn't cause warp divergence. The only conditional branch is the bounds check on line 90:

```90:90:hotspot-cuda.cu
    if (r < row && c < col) {
```

This only affects the **last block column and last block row** when grid dimensions aren't multiples of 16. In those blocks, some warps diverge (some threads compute, others skip). For a 1024x1024 grid (perfectly aligned to 16), this branch is never taken — all threads are in-bounds. Even for misaligned sizes, at most 1 row and 1 column of blocks are affected, which is a negligible fraction of total work.

### F. Bank Conflicts in Shared Memory (minor)

The shared memory array has a row stride of 18 floats. A warp of 32 threads spans two rows (16 threads per row in the 16x16 block). When both half-warps access the same column across different rows, their bank indices differ by `18 mod 32 = 18`. Since both half-warps access 16 consecutive banks, there is a 2-element overlap in bank indices, causing **2-way bank conflicts** on 2 out of 32 accesses. This doubles the shared memory access time for those 2 transactions — a ~6% shared memory overhead. In practice, since the kernel is global-memory-bound, this is negligible.

