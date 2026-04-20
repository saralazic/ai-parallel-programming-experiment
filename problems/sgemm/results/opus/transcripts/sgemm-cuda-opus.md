# GPU parallelization analysis of algorithm
_Exported on 3/15/2026 at 12:33:32 GMT+1 from Cursor (2.4.31)_

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

Let me explore the project to understand the algorithm first.

Here is a detailed analysis of your SGEMM (Single-precision General Matrix Multiply) codebase.

---

## Algorithm Overview

The core computation lives in `sgemm_kernel.cc` and implements the standard BLAS operation:

**C = β·C + α·A·B^T**

where A is `m×k`, B^T is `n×k` (so B is `k×n`), and C is `m×n`, all stored in **column-major** layout. The kernel is a classic triple-nested loop:

```27:37:sgemm_kernel.cc
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      float c = 0.0f;
      for (int i = 0; i < k; ++i) {
        float a = A[mm + i * lda]; 
        float b = B[nn + i * ldb];
        c += a * b;
      }
      C[mm+nn*ldc] = C[mm+nn*ldc] * beta + alpha * c;
    }
  }
```

---

## 1. Which Computations Are Parallelizable and Why

### Fully parallelizable: the outer two loops (mm and nn)

Each iteration of the `mm` loop (rows of C) and `nn` loop (columns of C) computes **one independent element** `C[mm, nn]`. There are `m × n` such elements, and each can be computed without any knowledge of or interaction with any other element. This is the textbook definition of **embarrassingly parallel** work.

- **Why:** Element `C[i, j]` is the dot product of row `i` of A and row `j` of B^T (i.e., column `j` of B). This dot product reads only from A and B (which are read-only inputs) and writes to a unique location `C[i, j]`. No output element depends on any other output element.

- **Degree of parallelism:** `m × n` independent tasks. For the medium test case, if the matrices are, say, 512×512, that's 262,144 fully independent computations — more than enough to saturate a modern GPU with thousands of cores.

### Partially parallelizable: the inner loop (i over k — the reduction)

The innermost loop accumulates `c += a * b` across the shared dimension `k`. This is a **reduction** (summation). It *can* be parallelized using parallel reduction techniques (tree-based sum, warp shuffles, etc.), but it has a sequential data dependency: each addition depends on the running sum. In practice:

- For small-to-moderate `k`, it is more efficient to let each thread compute the full dot product sequentially.
- For very large `k`, the reduction can be split across threads using **split-k** parallelism, where partial sums are computed in parallel and then atomically or hierarchically merged.

### Not parallelizable on GPU: I/O

The file reading (`readColMajorMatrixFile`) and writing (`writeColMajorMatrixFile`) in `io.cc` are inherently sequential host-side operations. These stay on the CPU.

---

## 2. Data Dependencies That May Limit Parallelization

### No write-write conflicts (the good news)

Each `(mm, nn)` pair writes to a **unique** location `C[mm + nn * ldc]`. There are zero write conflicts between parallel tasks, so no synchronization is needed for the output.

### Read-only shared inputs (also good)

Matrices A and B^T are only read, never written. Multiple threads can read the same elements concurrently without any hazard.

### The reduction dependency (the main constraint)

Inside the inner loop, the accumulation `c += a * b` forms a **loop-carried dependency chain**:

```
c_0 = 0
c_1 = c_0 + a[0]*b[0]
c_2 = c_1 + a[1]*b[1]
...
c_k = c_{k-1} + a[k-1]*b[k-1]
```

Each step depends on the previous. This limits instruction-level parallelism within a single thread and constrains split-k strategies to requiring a final merge step.

### The final read-modify-write

Line 35 — `C[mm+nn*ldc] = C[mm+nn*ldc] * beta + alpha * c` — reads the old value of C, scales it, and writes back. If `beta == 0` (as it is in your `main.cc` call), the old value doesn't matter and this simplifies to a pure write. But in the general case, this read-modify-write must happen atomically if multiple threads were to write to the same C element (which they don't here, so it's fine).

### Memory access patterns (implicit dependency on layout)

Column-major storage means:
- **A access** `A[mm + i * lda]`: iterating `mm` in the inner dimension gives stride-1 (coalesced) access across threads that vary in `mm`. Iterating `i` strides by `lda` — large jumps within a single thread.
- **B^T access** `B[nn + i * ldb]`: same pattern — coalesced across threads varying in `nn`, strided within a thread across `i`.

This is not a data dependency per se, but the memory layout **constrains** how you map threads to work to achieve efficient memory coalescing.

---

## 3. Parallelization Strategy for GPU Execution

### Thread-to-element mapping

The canonical CUDA approach for SGEMM:

| GPU Concept | Maps To |
|---|---|
| **Grid** | The entire C matrix (`m × n` elements) |
| **Thread block** | A tile of C, e.g. 16×16 or 32×32 elements |
| **Thread** | One (or more, via register tiling) element(s) of C |

Each thread computes one `C[mm, nn]` by performing the full dot-product reduction over `k`.

### Tiled algorithm with shared memory

The naive approach (each thread independently loads from global memory) is bandwidth-bound. The standard optimization:

1. **Partition** the `k` dimension into tiles of size `TILE_K` (e.g., 16 or 32).
2. **Cooperatively load** a `BLOCK_M × TILE_K` tile of A and a `BLOCK_N × TILE_K` tile of B^T into **shared memory** using all threads in the block.
3. **Synchronize** (`__syncthreads()`).
4. **Compute** partial dot products using the shared memory tiles.
5. **Repeat** for the next tile along `k`.
6. **Write** the final accumulated result to C in global memory.

This reduces global memory accesses by a factor of ~`TILE_K`, turning the algorithm from memory-bound to compute-bound.

### Register tiling (thread coarsening)

Each thread computes not just one but a small `r_m × r_n` sub-tile of C (e.g., 4×4 or 8×8). This:
- Increases arithmetic intensity per thread.
- Amortizes shared memory loads across more FMA operations.
- Reduces total thread count, which can improve occupancy characteristics.

### Sketch of the CUDA kernel structure

```c
__global__ void sgemm_kernel(int m, int n, int k, float alpha,
                              const float *A, int lda,
                              const float *B, int ldb,
                              float beta, float *C, int ldc) {
    __shared__ float As[TILE_K][BLOCK_M];
    __shared__ float Bs[TILE_K][BLOCK_N];

    // Each thread accumulates its register tile
    float acc[REG_M][REG_N] = {0.0f};

    for (int tile = 0; tile < k; tile += TILE_K) {
        // Cooperative load of A and B tiles into shared memory
        // ...
        __syncthreads();

        // Compute partial products from shared memory
        for (int kk = 0; kk < TILE_K; ++kk) {
            // Load from shared mem into registers, accumulate
        }
        __syncthreads();
    }

    // Write results: C = beta*C + alpha*acc
}
```

### Grid launch configuration

```c
dim3 block(BLOCK_M / REG_M, BLOCK_N / REG_N);  // e.g., 16×16 = 256 threads
dim3 grid((m + BLOCK_M - 1) / BLOCK_M, (n + BLOCK_N - 1) / BLOCK_N);
sgemm_kernel<<<grid, block>>>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
```

---

## 4. Expected Performance Challenges

### (a) Memory bandwidth bottleneck

SGEMM has an arithmetic intensity of `O(k)` FLOPs per element loaded (when tiled). Without shared-memory tiling, each thread independently fetches `2k` floats from global memory for just `2k` FLOPs — an arithmetic intensity of ~1 FLOP/byte, which will be **severely bandwidth-bound** on any GPU. Tiling is not optional; it is mandatory for reasonable performance.

### (b) Global memory coalescing

Your column-major layout means adjacent threads should map to adjacent `mm` values (rows) to achieve coalesced 128-byte transactions. If threads in a warp map to adjacent `nn` values instead, each warp would load from `n` different cache lines — a massive bandwidth waste. Getting the thread-to-element mapping right is critical.

### (c) Shared memory bank conflicts

Shared memory has 32 banks. If your tile layout causes multiple threads in a warp to access the same bank (but different addresses), accesses are serialized. Padding shared memory arrays (e.g., `As[TILE_K][BLOCK_M + 1]`) is a common workaround.

### (d) Occupancy vs. register pressure trade-off

Register tiling (e.g., each thread computing an 8×8 sub-tile) uses ~64 registers just for accumulators, plus temporaries. High register usage reduces the number of concurrent warps per SM (occupancy). Low occupancy means less ability to hide memory latency. Tuning the tile sizes involves balancing:
- More register tiling → higher arithmetic intensity, fewer threads
- Less register tiling → more threads for latency hiding, lower intensity

### (e) Boundary handling

When `m` or `n` is not a multiple of the block/tile size, boundary threads must be masked to avoid out-of-bounds memory accesses. This adds branch divergence within warps at the matrix edges, reducing efficiency for non-power-of-two sizes.

### (f) Host-device data transfer overhead

The matrices must be copied from host (CPU) memory to device (GPU) memory before the kernel launches, and the result C must be copied back. For small matrices, this transfer time can dominate the actual computation time, negating GPU acceleration. For your medium test case, whether GPU is net-faster depends on the matrix dimensions. Overlapping transfers with computation (using CUDA streams) can help for larger problems.

### (g) Floating-point associativity

The sequential CPU code accumulates the dot product in a specific order. Parallel reduction (or even just tiled computation) changes the summation order. Due to floating-point non-associativity, results may differ in the last few bits. This is expected and generally acceptable, but it means the GPU output may not be **bit-identical** to the CPU reference output in `medium/output/matrix3.txt`.

---

## Summary

| Aspect | Detail |
|---|---|
| **Parallelism** | `m × n` independent dot products — embarrassingly parallel |
| **Key dependency** | Reduction over `k` within each dot product |
| **Primary strategy** | 2D thread blocks, each block computes a tile of C using shared memory |
| **Critical optimization** | Shared memory tiling to exploit data reuse along `k` |
| **Main challenges** | Memory coalescing, bank conflicts, register pressure, transfer overhead |
| **Expected speedup** | Orders of magnitude for large matrices (e.g., 1024×1024+) vs. single-threaded CPU |

This is one of the most well-studied GPU parallelization problems. The algorithm maps almost perfectly to GPU architecture — the main engineering effort is in the memory hierarchy optimizations (tiling, coalescing, bank-conflict avoidance), not in exposing parallelism.

---

**User**

Based on your analysis of code and proposed strategy GPU paralelization:
1. Implement CUDA kernel(s)
2. Organize threads into blocks and grids appropriately
3. Manage memory transfers between host (CPU) and device (GPU)
4. Use suitable CUDA memory types (global, shared, local)

---

**Cursor**

I'll re-read the existing files once more, then create the CUDA implementation with the tiled shared-memory kernel, proper host-device memory management, and an updated build system.

All three files are in place. Here is a walkthrough of the implementation.

---

## File Structure

| File | Role |
|---|---|
| `sgemm_kernel.cu` | CUDA kernel + host wrapper (`basicSgemm`) |
| `main.cu` | Entry point — reads matrices, calls wrapper, reports timing |
| `Makefile` | Builds both CPU (`make`) and CUDA (`make cuda`) targets |
| *(originals untouched)* | `main.cc`, `sgemm_kernel.cc`, `io.cc` remain as-is |

---

## 1. CUDA Kernel — Tiled Shared-Memory SGEMM

The kernel in `sgemm_kernel.cu` implements a **16×16 tiled** matrix multiply. Key design choices:

**Thread-to-element mapping** — each thread computes one element of C:

```72:75:sgemm_kernel.cu
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCK_SIZE + ty;
    const int col = blockIdx.x * BLOCK_SIZE + tx;
```

**Shared memory tiles** — two `BLOCK_SIZE × BLOCK_SIZE` tiles are declared in fast on-chip shared memory, eliminating redundant global memory reads:

```67:68:sgemm_kernel.cu
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
```

**Coalesced global loads** — the tile load uses `threadIdx.x` for the **row** (stride-1) index of the column-major matrices, so consecutive threads in a warp touch consecutive memory addresses:

```84:95:sgemm_kernel.cu
        int aRow = blockIdx.y * BLOCK_SIZE + tx;
        int aCol = tileStart + ty;
        As[ty][tx] = (aRow < m && aCol < k) ? A[aRow + (size_t)aCol * lda]
                                             : 0.0f;

        /*
         * Load B^T tile — same coalescing strategy.
         */
        int bRow = blockIdx.x * BLOCK_SIZE + tx;
        int bCol = tileStart + ty;
        Bs[ty][tx] = (bRow < n && bCol < k) ? B[bRow + (size_t)bCol * ldb]
                                             : 0.0f;
```

This transposed storage means the compute step reads `As[kk][ty]` and `Bs[kk][tx]` — a broadcast and a stride-1 access respectively — which is **bank-conflict-free**.

**Compute loop** — walks down the k-dimension within the shared tile, fully unrolled:

```99:101:sgemm_kernel.cu
        #pragma unroll
        for (int kk = 0; kk < BLOCK_SIZE; ++kk)
            acc += As[kk][ty] * Bs[kk][tx];
```

**Boundary handling** — out-of-bounds loads return 0.0f, and the final writeback is guarded:

```105:107:sgemm_kernel.cu
    if (row < m && col < n)
        C[row + (size_t)col * ldc] = beta * C[row + (size_t)col * ldc]
                                   + alpha * acc;
```

---

## 2. Grid and Block Organization

```135:137:sgemm_kernel.cu
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);               // 16×16 = 256 threads
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,      // covers columns of C
              (m + BLOCK_SIZE - 1) / BLOCK_SIZE);      // covers rows of C
```

- **Block**: 16×16 = **256 threads** — a good balance between occupancy and shared memory usage (2 × 16 × 16 × 4 bytes = **2 KB** shared memory per block, well within the 48–100 KB available per SM).
- **Grid**: ceil(n/16) × ceil(m/16) blocks — one block per tile of C. The ceiling division handles matrices whose dimensions are not multiples of 16.

---

## 3. Host-Device Memory Management

The `basicSgemm` host wrapper maintains the exact same function signature as the original CPU version, making it a drop-in replacement. Memory flow:

```119:130:sgemm_kernel.cu
    float *d_A, *d_B, *d_C;
    size_t sizeA = (size_t)lda * k * sizeof(float);
    size_t sizeB = (size_t)ldb * k * sizeof(float);
    size_t sizeC = (size_t)ldc * n * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    /* ----- Host → Device transfers -------------------------------- */
    CUDA_CHECK(cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice));
```

- `cudaMalloc` allocates **device global memory** for A, B^T, and C.
- `cudaMemcpy` with `HostToDevice` copies the input matrices from the CPU `std::vector` data to the GPU.
- C is only copied to the device when `beta != 0` (since `C_old` matters in that case). For the `beta = 0` call in `main.cu`, this transfer is skipped.
- After the kernel, `DeviceToHost` copies the result back.
- `cudaFree` releases all device allocations.

All CUDA calls are wrapped in a `CUDA_CHECK` macro that reports the error location and terminates if anything fails.

---

## 4. Memory Types Used

| CUDA Memory | Used For | Why |
|---|---|---|
| **Global memory** | `d_A`, `d_B`, `d_C` — full matrices on the device | Required for input/output; accessible by all threads |
| **Shared memory** | `As[16][16]`, `Bs[16][16]` — per-block tiles | ~100× lower latency than global; enables data reuse across the BLOCK_SIZE×BLOCK_SIZE threads in a block |
| **Registers** | `acc` — the running dot-product accumulator; `tx`, `ty`, `row`, `col` | Fastest storage; private to each thread |
| **Constant values** | `m`, `n`, `k`, `alpha`, `beta`, `lda`, `ldb`, `ldc` (kernel arguments) | Passed via constant memory/registers by the CUDA runtime |

---

## Building and Running

```bash
# Build the CUDA version (requires nvcc / CUDA toolkit)
make cuda

# Run on the medium test case
make run_cuda

# The original CPU build still works unchanged
make
make run
```

The CUDA binary `sgemm_cuda` is a drop-in replacement — same CLI arguments, same I/O format, same output file.

---

**User**

Explain 
1. how this CUDA solution maps onto GPU hardware architecture.
2. How threads and thread blocks are organized
3. How memory hierarchy is used
4. What performance limitations may exist

---

**Cursor**

Let me re-read the kernel to give precise, line-referenced answers.

Here is a detailed explanation of all four aspects.

---

## 1. How the CUDA Solution Maps onto GPU Hardware Architecture

A GPU is organized as an array of **Streaming Multiprocessors (SMs)**, each containing many ALUs (CUDA cores), a register file, shared memory (SRAM), and warp schedulers. The CUDA programming model maps onto this in three layers:

### Kernel → the entire GPU

The single call on line 144 dispatches work to the entire device. The GPU's **global scheduler** distributes thread blocks from the grid across the available SMs. Each SM can execute one or more thread blocks concurrently (limited by shared memory, register count, and max-thread-per-SM caps). Blocks that don't fit run as soon as an earlier block retires — the hardware handles all scheduling.

### Thread block → one SM

Each 16×16 block (256 threads) is assigned to exactly one SM and never migrates. This is what makes `__shared__` memory (line 59–60) work — it lives in the SM's on-chip SRAM and is visible to all 256 threads in the block, but invisible to threads in other blocks. The two `__syncthreads()` barriers (lines 91 and 97) map to the SM's hardware barrier unit, which stalls warps within a block until all of them have reached the barrier.

### Warp → the SIMT execution unit

The 256 threads in a block are split into **8 warps** of 32 threads each. The SM's warp scheduler issues the same instruction to all 32 threads in a warp simultaneously (SIMT). In this kernel, with `blockDim.x = 16`, a warp consists of 2 consecutive rows of the 16×16 thread grid (e.g., threads with `ty=0, tx=0..15` and `ty=1, tx=0..15`). This matters directly for memory coalescing — see Section 3.

### Compute mapping

Each output element `C[row, col]` maps to exactly one thread. The thread performs a dot product of length `k`, broken into `ceil(k/16)` tile iterations. Each tile iteration does 16 multiply-add operations (line 94–95). The total FLOPs per thread are `2k` (one multiply + one add per `k` element), and with `m × n` threads executing in parallel, the total is `2·m·n·k` — matching the GFLOPs formula in `main.cu`.

---

## 2. How Threads and Thread Blocks Are Organized

### Block dimensions

```140:140:sgemm_kernel.cu
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);               // 16×16 = 256 threads
```

A 2D block of 16×16 = **256 threads**. This size was chosen because:
- 256 is a multiple of the warp size (32), so no partial warps are wasted.
- 16×16 tiles fit comfortably in shared memory (2 tiles × 16×16 × 4 bytes = 2 KB).
- 256 threads/block allows 4–8 blocks per SM on most architectures (good for occupancy).

### Grid dimensions

```141:142:sgemm_kernel.cu
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,      // covers columns of C
              (m + BLOCK_SIZE - 1) / BLOCK_SIZE);      // covers rows of C
```

The grid is 2D: `ceil(n/16)` blocks in x (covering columns of C) and `ceil(m/16)` blocks in y (covering rows of C). For a 512×512 matrix, that's 32×32 = **1,024 blocks**, each with 256 threads = **262,144 threads** total.

### Thread-to-element mapping

```62:65:sgemm_kernel.cu
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCK_SIZE + ty;
    const int col = blockIdx.x * BLOCK_SIZE + tx;
```

Thread `(tx, ty)` in block `(blockIdx.x, blockIdx.y)` owns output element `C[row, col]`. The `x` dimension maps to **columns** and `y` to **rows** of C. This means threads within a warp (consecutive `tx` values at the same `ty`) map to the same row but consecutive columns. However — and this is subtle — the *tile-loading* step deliberately swaps this (see memory hierarchy below).

### Warp structure within a block

Since the linearized thread ID is `ty * 16 + tx`, warp 0 contains `{ty=0, tx=0..15}` ∪ `{ty=1, tx=0..15}`. All threads in a half-warp (16 threads) share the same `ty` and have consecutive `tx`. This is the key fact exploited by the coalescing strategy.

---

## 3. How the Memory Hierarchy Is Used

The GPU memory hierarchy has four levels, ordered from slowest/largest to fastest/smallest. This kernel uses all of them:

### Level 1: Global Memory (device DRAM, ~900 GB/s, ~400 cycle latency)

The three matrices `d_A`, `d_B`, `d_C` live here after being transferred from the host:

```129:131:sgemm_kernel.cu
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));
```

Global memory is the **bottleneck** — every byte loaded from it must be maximally reused. Without tiling, each element of A would be read `n` times and each element of B^T would be read `m` times. Tiling reduces this by a factor of `BLOCK_SIZE` (16×).

**Coalesced access pattern:** The tile loads use `tx` (the warp-contiguous index) for the stride-1 dimension of the column-major layout:

```78:81:sgemm_kernel.cu
        int aRow = blockIdx.y * BLOCK_SIZE + tx;
        int aCol = tileStart + ty;
        As[ty][tx] = (aRow < m && aCol < k) ? A[aRow + (size_t)aCol * lda]
                                             : 0.0f;
```

For threads in a half-warp (same `ty`, consecutive `tx`), `aRow` increments by 1 → consecutive 4-byte addresses → merged into a single 64-byte or 128-byte transaction. Without this coalescing, 16 separate transactions would be needed instead of 1, wasting 15/16 of the memory bandwidth.

### Level 2: Shared Memory (on-chip SRAM, ~19 TB/s effective, ~30 cycle latency)

```59:60:sgemm_kernel.cu
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
```

Each block allocates 2 × 16 × 16 × 4 = **2,048 bytes** of shared memory. This is allocated from the SM's ~48–100 KB SRAM pool. The data flow is:

1. **Load phase** (lines 78–89): All 256 threads cooperatively load a 16×16 tile from global → shared.
2. **Barrier** (line 91): `__syncthreads()` ensures the tile is fully populated before any thread reads it.
3. **Compute phase** (lines 93–95): Each thread reads 16 values from `As` and 16 from `Bs` — data that was loaded by *other threads* in the block. This is the reuse: each float loaded from global memory is consumed by 16 threads.
4. **Barrier** (line 97): Ensures no thread starts overwriting the tile for the next iteration while others are still reading.

**Bank-conflict-free access:** Shared memory is divided into 32 banks, with consecutive 4-byte words in consecutive banks. During the compute loop:
- `As[kk][ty]`: all threads in a half-warp (same `ty`, different `tx`) read the **same address** → hardware **broadcast**, zero conflict.
- `Bs[kk][tx]`: threads with consecutive `tx` read consecutive elements → consecutive banks, zero conflict.

### Level 3: Registers (per-thread, ~10 TB/s, 0 latency)

```67:67:sgemm_kernel.cu
    float acc = 0.0f;
```

The accumulator `acc` lives in a register for the entire kernel lifetime — `ceil(k/16)` iterations of 16 FMAs each. It is only written to global memory once, at line 101. Similarly, `tx`, `ty`, `row`, `col` are all register-resident. The compiler also allocates registers for the temporaries in the FMA (the values loaded from `As` and `Bs` each iteration).

Each thread uses roughly **10–12 registers** (acc, tx, ty, row, col, loop counters, temporaries). With 256 threads/block, that's ~3,072 registers per block, out of the 65,536 typically available per SM — leaving plenty of room for multiple concurrent blocks.

### Level 4: Constant/Kernel-Argument Memory

The scalar parameters `m`, `n`, `k`, `alpha`, `beta`, `lda`, `ldb`, `ldc` are passed as kernel arguments. The CUDA runtime places these in constant memory, which is cached and broadcast-efficient — all threads reading the same constant in the same cycle incur only a single memory transaction.

### Host ↔ Device Transfers (PCIe / NVLink)

```134:137:sgemm_kernel.cu
    CUDA_CHECK(cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice));
    if (beta != 0.0f)
        CUDA_CHECK(cudaMemcpy(d_C, C, sizeC, cudaMemcpyHostToDevice));
```

These cross the PCIe bus (~32 GB/s for PCIe 4.0 x16). The `beta != 0` check on line 136 avoids an unnecessary transfer when C doesn't need its old values — as is the case in the `main.cu` call where `beta = 0.0f`.

---

## 4. Performance Limitations

### (a) Arithmetic intensity is capped by tile size

Each tile iteration loads `2 × 16 × 16 = 512` floats (2,048 bytes) from global memory and performs `16 × 16 × 16 = 4,096` FLOPs. That gives an **arithmetic intensity of 2 FLOPs/byte**. A GPU with 900 GB/s bandwidth and 10 TFLOP/s peak needs at least ~11 FLOPs/byte to be compute-bound. At 2 FLOPs/byte, this kernel is **memory-bandwidth-bound** — it achieves only ~18% of peak FLOPs on such a GPU.

To fix this, production SGEMM kernels use larger tiles (64×64 or 128×128) with **register tiling** — each thread computes an `r×r` sub-tile (e.g., 8×8), raising the intensity to ~8–16 FLOPs/byte. This kernel uses a 1×1 per-thread mapping, the simplest correct approach but not optimal.

### (b) Low per-thread work → limited instruction-level parallelism

Each thread performs only 16 FMAs per tile iteration. Modern GPU pipelines can issue independent instructions back-to-back, but with a single accumulator and a tight 16-iteration loop, there is very little instruction-level parallelism (ILP) to exploit. Register tiling (multiple accumulators) would let the hardware pipeline multiple independent FMAs.

### (c) Occupancy is not the bottleneck, but it's not fully exploited

256 threads/block × 2,048 bytes shared memory/block allows many concurrent blocks per SM (typically 4–8). High occupancy means more warps available to hide memory latency. However, since the kernel is bandwidth-bound (not latency-bound), simply adding more warps does not help — the bottleneck is total bandwidth, not latency hiding.

### (d) Host-device transfer overhead

For an `n×n` matrix multiply, the transfers move `3n²` floats = `12n²` bytes over PCIe. The compute does `2n³` FLOPs. The ratio (compute/transfer) scales as `n` — meaning:
- **Small matrices (n < ~256):** Transfer time dominates. The GPU may actually be slower than the CPU.
- **Large matrices (n > ~1024):** Transfer is negligible; GPU wins by orders of magnitude.

The current implementation uses **synchronous** transfers (one `cudaMemcpy` blocks until complete before the next starts). Using **CUDA streams** with `cudaMemcpyAsync` + pinned memory could overlap transfers with computation.

### (e) Boundary divergence

```100:100:sgemm_kernel.cu
    if (row < m && col < n)
```

When `m` or `n` is not a multiple of 16, some threads in edge blocks are masked off. This causes **warp divergence** — some threads idle while others execute the writeback. The impact is proportional to the perimeter of the matrix relative to its area: negligible for large matrices, but measurable for small ones.

### (f) No double-buffering of shared memory tiles

The kernel loads a tile, syncs, computes, syncs, then loads the *next* tile. During the load phase, the compute units are idle; during compute, the memory pipeline is idle. **Double-buffering** — loading tile `t+1` while computing tile `t` — would overlap these two phases. This requires twice the shared memory (4 KB instead of 2 KB, still easily affordable) and software pipelining of the loop.

### Summary: where this kernel sits on the optimization ladder

| Optimization level | Technique | This kernel |
|---|---|---|
| 0 — Naive | One thread per C element, global memory only | No |
| 1 — Tiled | Shared memory tiling, coalesced loads | **Yes** |
| 2 — Register-tiled | Each thread computes r×r sub-tile | No |
| 3 — Double-buffered | Overlap load and compute phases | No |
| 4 — Vectorized loads | `float4` loads, 128-bit transactions | No |
| 5 — Warp-level (tensor cores) | `wmma` / `mma` intrinsics | No |

This implementation is at **level 1** — correct, significantly faster than CPU, and demonstrating the key GPU concepts (thread hierarchy, shared memory, coalescing, barriers). Reaching levels 2–5 is what separates a teaching kernel from a cuBLAS-class implementation.

