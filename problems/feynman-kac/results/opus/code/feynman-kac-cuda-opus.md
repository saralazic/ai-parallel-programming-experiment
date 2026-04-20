# GPU parallelization analysis of algorithm
_Exported on 4/20/2026 at 17:31:12 GMT+2 from Cursor (3.0.16)_

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

Let me start by exploring the project structure and understanding the code.

This is a **Feynman-Kac 1D Monte Carlo solver** — it estimates the solution to a PDE at grid points by simulating random walks from each point to the boundary and accumulating a weighted functional along each path. Here's a detailed parallelization analysis.

---

## Algorithm Structure Overview

The code has three nested levels:

1. **Outer loop** (line 88): iterates over `ni + 2 = 23` spatial grid points
2. **Middle loop** (line 119): runs `n = 10,000` Monte Carlo trajectories per grid point
3. **Inner loop** (line 132): a `while` loop that walks a single trajectory until it exits the interval `[-a, a]`

The total work is approximately `23 × 10,000 × (variable steps)` random walk iterations.

---

## 1. Which Computations Are Parallelizable and Why

### A. The Monte Carlo trajectories (middle loop) — **Massively parallelizable**

```119:163:feynman-kac-seq.c
    for ( it = 1; it <= n; it++ )
    {

      x1 = x;
      // ... random walk to boundary ...
      wt = wt + w;
    }
```

Each of the 10,000 trajectories for a given grid point is **statistically independent** by design. A trajectory starts at `x`, walks randomly until it exits the domain, and produces a single weight `w`. The final estimate is just the average of all `w` values. This is a textbook **embarrassingly parallel** Monte Carlo workload — each trajectory can be assigned to a separate GPU thread with zero inter-thread communication during the walk.

### B. The grid point loop (outer loop) — **Fully parallelizable**

```88:176:feynman-kac-seq.c
  for ( i = 0; i <= ni + 1; i++ )
  {
    // ... compute x, exact solution, run Monte Carlo ...
  }
```

Each grid point's computation is independent of every other grid point. The `err` accumulation (line 172) and `n_int` counter (line 108) are simple reductions that can be handled after all points finish. With 23 grid points × 10,000 trajectories, you have **230,000 independent trajectories** that can execute simultaneously on GPU.

### C. The `potential()` function — **Trivially parallelizable**

```197:204:feynman-kac-seq.c
double potential ( double a, double x )
{
  double value;

  value = 2.0 * pow ( x / a / a, 2 ) + 1.0 / a / a;

  return value;
}
```

This is a pure arithmetic function with no side effects — it maps directly to per-thread computation with no synchronization.

### D. The exact solution computation — **Parallelizable**

```112:112:feynman-kac-seq.c
    w_exact = exp ( pow ( x / a, 2 ) - 1.0 );
```

One `exp` per grid point, trivially computed in parallel.

---

## 2. Data Dependencies That May Limit Parallelization

### A. The PRNG state — **The critical serial bottleneck**

```207:227:feynman-kac-seq.c
double r8_uniform_01 ( int *seed )
{
  int k;
  double r;

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + 2147483647;
  }
  // ...
  r = ( double ) ( *seed ) * 4.656612875E-10;

  return r;
}
```

This is a **linear congruential generator (LCG)** with a single shared `seed`. Every call mutates `seed`, and the next value depends on the previous — a **strict sequential dependency**. This is the single most important blocker to naive parallelization. You **cannot** share this PRNG across threads.

**Solution:** Replace with a **parallel-friendly PRNG** strategy:
- **Per-thread independent seeds:** Give each GPU thread its own seed (e.g., derived from `thread_id + base_seed`). This breaks reproducibility with the sequential version but is standard practice.
- **Use cuRAND:** NVIDIA's cuRAND library provides GPU-optimized generators (XORWOW, Philox, MRG32k3a) specifically designed for massively parallel random number generation with guaranteed statistical independence across threads.

### B. The inner `while` loop — **Variable-length, data-dependent**

```132:161:feynman-kac-seq.c
      while ( chk < 1.0 )
      {
        us = r8_uniform_01 ( &seed ) - 0.5;
        // ... step and update ...
        chk = pow ( x1 / a, 2 );
      }
```

The number of steps per trajectory is **not known in advance** — it depends on the random walk itself. Trajectories starting near the center of the interval take many more steps than those starting near the boundary. This creates **warp divergence** on GPU (threads in the same warp finish at different times).

### C. Accumulation of `wt` — **Reduction dependency**

```162:162:feynman-kac-seq.c
      wt = wt + w;
```

All trajectories for a grid point contribute to a single sum. This is a classic **parallel reduction** — well-supported on GPU via atomic adds or hierarchical reduction.

### D. Global error accumulation — **Reduction dependency**

```172:172:feynman-kac-seq.c
    err = err + pow ( w_exact - wt, 2 );
```

Depends on the fully computed `wt` for each grid point. This is a secondary reduction over only 23 values, trivially handled after the Monte Carlo phase.

---

## 3. Parallelization Strategy for GPU Execution

### Thread/Block Mapping

| Dimension | Maps to | Count |
|-----------|---------|-------|
| Grid point `i` | Block Y (or flattened) | 23 |
| Trajectory `it` | Thread X + Block X | 10,000 |
| **Total threads** | | **~230,000** |

A natural CUDA grid would be:

```c
dim3 blocks(ceil(n / THREADS_PER_BLOCK), ni + 2);
dim3 threads(THREADS_PER_BLOCK);  // e.g., 256
```

### Kernel Structure (pseudocode)

```c
__global__ void feynman_kac_kernel(double a, double h, int n, int ni,
                                    double *wt_out, int *steps_out) {
    int grid_idx = blockIdx.y;
    int traj_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (traj_idx >= n) return;

    // Compute x for this grid point
    double x = compute_grid_point(grid_idx, ni, a);
    if (a*a - x*x < 0.0) { /* boundary point, write 1.0 */ return; }

    // Initialize per-thread PRNG (cuRAND)
    curandState state;
    curand_init(base_seed, grid_idx * n + traj_idx, 0, &state);

    // Random walk
    double x1 = x, w = 1.0, rth = sqrt(h);
    int local_steps = 0;
    while ((x1/a)*(x1/a) < 1.0) {
        double us = curand_uniform_double(&state) - 0.5;
        double dx = (us < 0.0) ? -rth : rth;
        double vs = potential(a, x1);
        x1 += dx;
        local_steps++;
        double vh = potential(a, x1);
        double we = (1.0 - h * vs) * w;
        w = w - 0.5 * h * (vh * we + vs * w);
    }

    // Parallel reduction via atomicAdd
    atomicAdd(&wt_out[grid_idx], w);
    atomicAdd(&steps_out[grid_idx], local_steps);
}
```

### Post-Kernel (Host Side)

After the kernel completes, the host divides each `wt_out[i]` by `n`, computes `w_exact` values, and accumulates the RMS error. This is negligible work (23 points).

---

## 4. Expected Performance Challenges

### A. Warp Divergence from Variable-Length Walks

The `while` loop has a **data-dependent termination condition**. Within a warp of 32 threads, some trajectories (starting near `x = 0`) will take ~40,000 steps while others (starting near `x = ±a`) will take very few. The warp cannot retire until its *longest* trajectory finishes — all shorter-lived threads idle. This is the **dominant performance concern**.

**Mitigations:**
- **Sort or group trajectories** by starting grid point so that threads within a warp share similar expected walk lengths.
- **Persistent-thread approach:** have threads pick up new trajectories from a work queue when they finish, avoiding idle time.

### B. Random Number Generation Overhead

cuRAND state initialization (`curand_init`) is expensive. For 230,000 threads, this setup cost is non-trivial. The Philox generator has lower init cost than XORWOW and is often preferred for Monte Carlo workloads.

### C. Double-Precision Arithmetic

The entire computation uses `double`. GPU double-precision throughput is typically **1/2 to 1/32** of single-precision throughput depending on the GPU architecture (consumer GPUs are particularly penalized). This algorithm requires `double` for numerical stability of the exponential weighting, so this is largely unavoidable.

### D. Low Arithmetic Intensity

Each inner-loop iteration does: one PRNG call, a branch, two `potential()` evaluations (each a multiply/divide/add), and a few multiply-adds. This is **memory-light but also compute-light** per step — the bottleneck is likely **instruction throughput and latency hiding** rather than memory bandwidth. You need enough concurrent threads (occupancy) to hide instruction pipeline latency.

### E. Atomic Contention on Reduction

Using `atomicAdd(&wt_out[grid_idx], w)` means all 10,000 threads for a single grid point contend on the same memory address. For `double` atomics, this serializes at the L2 cache level.

**Mitigation:** Use a **shared-memory block-level reduction** first, then a single atomic per block:

```c
__shared__ double partial[THREADS_PER_BLOCK];
partial[threadIdx.x] = w;
__syncthreads();
// tree reduction in shared memory ...
if (threadIdx.x == 0) atomicAdd(&wt_out[grid_idx], block_sum);
```

### F. Small Problem Size (as written)

With only 230,000 trajectories, the GPU may not be fully saturated — modern GPUs can run millions of threads. Increasing `n` (e.g., to 100,000 or 1,000,000 trajectories) or `ni` (finer spatial grid) would improve GPU utilization and amortize kernel launch overhead. The algorithm scales linearly in `n`, making it ideal for "turning up the dial" on GPU.

---

## Summary

| Aspect | Assessment |
|--------|-----------|
| **Parallelism type** | Embarrassingly parallel (Monte Carlo) |
| **Primary parallel dimension** | 230,000 independent trajectories |
| **Critical serial dependency** | PRNG state → replace with cuRAND |
| **Main GPU challenge** | Warp divergence from variable-length walks |
| **Secondary challenges** | Double precision penalty, atomic reduction contention |
| **Expected speedup potential** | Very high (100-1000×) with proper PRNG and sufficient `n` |

The Feynman-Kac Monte Carlo solver is one of the most GPU-friendly algorithm classes. The primary engineering work is replacing the sequential LCG with a parallel PRNG (cuRAND) and managing warp divergence from the variable-length inner loop. The reductions are standard and well-solved on GPU.

---

**User**

Based on your analysis of code and proposed strategy GPU paralelization:
1. Implement CUDA kernel(s)
2. Organize threads into blocks and grids appropriately
3. Manage memory transfers between host (CPU) and device (GPU)
4. Use suitable CUDA memory types (global, shared, local)

---

**Cursor**

Let me do a final review of the implementation to make sure everything is consistent.

The CUDA implementation is complete. Here's a walkthrough of every design decision, mapped to your four requirements.

---

## 1. CUDA Kernel

A single kernel `feynman_kac_kernel` handles the entire Monte Carlo workload. Each thread executes one complete random walk trajectory:

```41:101:feynman-kac-cuda.cu
__global__ void feynman_kac_kernel(
    double              a,
    double              h,
    double              rth,
    int                 n,
    int                 ni,
    unsigned long long  base_seed,
    double             *d_wt_sums,
    int                *d_step_sums)
{
    // ... each thread: init PRNG, walk until boundary, accumulate weight
}
```

The sequential PRNG (`r8_uniform_01` with shared seed) is replaced by **cuRAND Philox4_32_10**, a counter-based generator that gives each thread its own independent random stream with zero inter-thread dependency:

```73:76:feynman-kac-cuda.cu
        curandStatePhilox4_32_10_t rng;
        curand_init(base_seed,
                    (unsigned long long)point_idx * n + traj_idx,
                    0, &rng);
```

Philox was chosen over XORWOW because it has much faster initialization (critical when every thread inits its own state) and excellent statistical quality for Monte Carlo.

The `potential` function is marked `__device__ __forceinline__` so it compiles into pure register arithmetic with zero function-call overhead:

```23:27:feynman-kac-cuda.cu
__device__ __forceinline__ double potential_d(double a, double x)
{
    double xa2 = x / (a * a);
    return 2.0 * xa2 * xa2 + 1.0 / (a * a);
}
```

---

## 2. Thread/Block/Grid Organization

The kernel uses a **2D grid** that directly maps the two parallel dimensions of the problem:

| Dimension | Maps to | Size |
|-----------|---------|------|
| `blockIdx.y` | Grid point index | `num_points = 23` |
| `blockIdx.x * blockDim.x + threadIdx.x` | Trajectory index | `n = 10,000` |
| Block size | — | `256 threads` |

```214:217:feynman-kac-cuda.cu
    int blocks_x = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    dim3 grid(blocks_x, num_points);
    dim3 block(THREADS_PER_BLOCK);
```

This yields `ceil(10000/256) × 23 = 40 × 23 = 920 blocks` and `235,520 total threads`. The 2D layout ensures threads in the same warp (consecutive `threadIdx.x` values) are trajectories for the **same grid point**, which means they share similar expected walk lengths — reducing warp divergence.

---

## 3. Host ↔ Device Memory Transfers

**Host → Device:** The algorithm parameters (`a`, `h`, `rth`, `n`, `ni`, `base_seed`) are passed as kernel arguments — they fit in the kernel parameter space (constant memory under the hood), so no explicit `cudaMemcpy` is needed for inputs.

**Device allocation and zeroing:**

```196:199:feynman-kac-cuda.cu
    CUDA_CHECK(cudaMalloc(&d_wt_sums,   num_points * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_step_sums, num_points * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_wt_sums,   0, num_points * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_step_sums, 0, num_points * sizeof(int)));
```

**Device → Host:** Only the two small result arrays (23 doubles + 23 ints = 276 bytes) are copied back after the kernel finishes:

```255:258:feynman-kac-cuda.cu
    CUDA_CHECK(cudaMemcpy(h_wt_sums,   d_wt_sums,
                           num_points * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_step_sums, d_step_sums,
                           num_points * sizeof(int),    cudaMemcpyDeviceToHost));
```

Post-processing (exact solutions, averaging, error computation, formatted printing) stays on the host because it's only 23 iterations and requires `printf`.

---

## 4. CUDA Memory Types Used

| Memory type | What's stored | Why |
|-------------|---------------|-----|
| **Registers / Local** | Per-thread walk state (`x1`, `w`, `chk`, `dx`, `vs`, `vh`, `we`), PRNG state (`curandStatePhilox4_32_10_t`) | Fastest access; each thread's walk is independent, so all working variables are private |
| **Shared** (`__shared__`) | `s_wt[256]` and `s_steps[256]` — partial sums for block-level reduction | Enables fast intra-block communication for the tree reduction; 3072 bytes per block, well within limits |
| **Global** | `d_wt_sums[23]`, `d_step_sums[23]` — per-grid-point accumulators | Only location visible across blocks; written via `atomicAdd` by block leaders only (not every thread) |
| **Constant** (implicit) | Kernel parameters (`a`, `h`, `rth`, etc.) | Passed as kernel args, cached in constant memory by the CUDA runtime |

The shared-memory tree reduction is the key optimization — it reduces 256 partial results down to one in 8 steps (`log2(256)`), so only one `atomicAdd` per block hits global memory instead of one per thread:

```109:127:feynman-kac-cuda.cu
    s_wt[threadIdx.x]    = my_w;
    s_steps[threadIdx.x] = my_steps;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            s_wt[threadIdx.x]    += s_wt[threadIdx.x + stride];
            s_steps[threadIdx.x] += s_steps[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&d_wt_sums[point_idx],   s_wt[0]);
        atomicAdd(&d_step_sums[point_idx], s_steps[0]);
    }
```

---

## Building

The `Makefile` builds both versions:

```bash
make seq    # builds feynman-kac-seq (gcc)
make cuda   # builds feynman-kac-cuda (nvcc, requires sm_60+ for double atomicAdd)
make all    # builds both
```

Run with `./feynman-kac-cuda`. The output format matches the sequential version, with added kernel timing and launch configuration info.

---

**User**

Explain 
1. how this CUDA solution maps onto GPU hardware architecture.
2. How threads and thread blocks are organized
3. How memory hierarchy is used
4. What performance limitations may exist

---

**Cursor**

Let me re-read the implementation to give precise references throughout the explanation.

## 1. How the CUDA Solution Maps onto GPU Hardware Architecture

A modern NVIDIA GPU is organized as an array of **Streaming Multiprocessors (SMs)**, each containing many CUDA cores, a register file, shared memory, warp schedulers, and L1 cache. The runtime's job is to distribute work across these SMs. Here is how each piece of this kernel maps to that hardware.

### Kernel → SM Assignment

When the kernel launches with `920 blocks` (40 x 23), the CUDA runtime's **block scheduler** distributes those blocks across all available SMs. Each SM can host multiple blocks concurrently (the exact number depends on register and shared memory usage per block). Since each block uses only 3,072 bytes of shared memory and a moderate number of registers, most GPUs can fit several blocks per SM simultaneously.

The key insight: the GPU hardware does not see "grid points" or "trajectories." It sees **blocks of 256 threads** to schedule. The 2D grid layout at line 216 is purely a logical convenience — the hardware flattens them into a queue of 920 blocks to distribute.

### Warp Execution on CUDA Cores

Inside each SM, the 256 threads of a block are divided into **8 warps** of 32 threads each. Each warp is the fundamental execution unit — all 32 threads in a warp execute the **same instruction** at the **same time** (SIMT model). The warp schedulers on each SM pick ready warps and issue their instructions to the CUDA cores.

In this kernel, the inner `while` loop (line 82-98) is where threads spend almost all of their time. Within a warp, all 32 threads are executing this same loop structure together, step by step. As long as all 32 threads are still inside the domain (`chk < 1.0`), they proceed in lockstep with full hardware utilization.

### Scalar Parameters → Constant Cache

The kernel parameters (`a`, `h`, `rth`, `n`, `ni`, `base_seed`) are passed through the kernel argument space, which the GPU caches in a dedicated **constant memory cache** — all threads across all SMs read these values with single-cycle broadcast latency after the first access.

---

## 2. How Threads and Thread Blocks Are Organized

### The 2D Grid

```214:217:feynman-kac-cuda.cu
    int blocks_x = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    dim3 grid(blocks_x, num_points);
    dim3 block(THREADS_PER_BLOCK);
```

This creates a grid that looks like:

```
            blockIdx.x →  0    1    2    3   ...   39
                        ┌────┬────┬────┬────┬───┬────┐
blockIdx.y = 0  (x=-2.2)│256 │256 │256 │256 │...│256 │  ← boundary point
                        ├────┼────┼────┼────┼───┼────┤
blockIdx.y = 1  (x=-2.0)│256 │256 │256 │256 │...│256 │  ← interior point
                        ├────┼────┼────┼────┼───┼────┤
blockIdx.y = 2  (x=-1.8)│256 │256 │256 │256 │...│256 │
                        ├────┼────┼────┼────┼───┼────┤
           ...          │    │    │    │    │   │    │
                        ├────┼────┼────┼────┼───┼────┤
blockIdx.y = 22 (x=+2.2)│256 │256 │256 │256 │...│256 │  ← boundary point
                        └────┴────┴────┴────┴───┴────┘
```

- **Y axis (23 rows):** each row corresponds to one spatial grid point
- **X axis (40 columns):** tiles the 10,000 trajectories into chunks of 256

Each thread computes its identity from two indices:

```51:52:feynman-kac-cuda.cu
    int point_idx = blockIdx.y;
    int traj_idx  = blockIdx.x * blockDim.x + threadIdx.x;
```

So thread `(blockIdx.x=3, blockIdx.y=5, threadIdx.x=17)` handles trajectory `3 * 256 + 17 = 785` at grid point `5`.

### Why 256 Threads per Block

The choice of 256 (line 8) is a deliberate balance:

- **Warp alignment:** 256 / 32 = 8 warps exactly. No partial warps waste CUDA cores.
- **Occupancy:** Modern SMs support up to 2048 resident threads. With 256-thread blocks, up to 8 blocks can coexist on one SM, providing ample warps for the scheduler to hide latency.
- **Shared memory:** 256 * (8 + 4) = 3,072 bytes per block. At the typical 48 KB shared memory limit, this allows up to 16 blocks by shared memory alone (though register pressure will be the tighter constraint here).
- **Reduction efficiency:** 256 = 2^8, so the tree reduction takes exactly 8 steps with no remainder handling.

### Boundary Guard

The last block in each row may launch more threads than trajectories remain (40 * 256 = 10,240 > 10,000). The guard at line 66 ensures surplus threads skip the walk and contribute zero to the reduction:

```66:66:feynman-kac-cuda.cu
    if (traj_idx < n && test >= 0.0)
```

This also skips threads for boundary grid points (`test < 0.0`), where the answer is known analytically to be 1.0. Those threads write `my_w = 0.0` into shared memory and participate in the reduction harmlessly (the host ignores those rows).

---

## 3. How the Memory Hierarchy Is Used

Here's a layer-by-layer breakdown, from fastest to slowest:

### Layer 1: Registers (per-thread, ~1 cycle latency)

The bulk of the computation lives in registers. Each thread's private variables — `x1`, `w`, `chk`, `dx`, `vs`, `vh`, `we`, `us`, `my_w`, `my_steps` — are all scalar values that the compiler allocates to the thread's slice of the SM's register file.

```78:100:feynman-kac-cuda.cu
        double x1  = x;
        double w   = 1.0;
        double chk = 0.0;

        while (chk < 1.0)
        {
            double us = curand_uniform_double(&rng) - 0.5;
            double dx = (us < 0.0) ? -rth : rth;
            // ... all register arithmetic ...
        }

        my_w = w;
```

The `curandStatePhilox4_32_10_t rng` structure (line 73) is also in registers/local memory — it holds the counter-based PRNG state (~40 bytes) that the compiler tries to keep in registers. If the total register count per thread exceeds the SM's per-thread limit (typically 255 registers), some spill into **local memory** (which is physically in L1/L2 cache-backed global memory, much slower).

The `potential_d` function is `__forceinline__` (line 23), so it generates no function call overhead — its arithmetic is emitted directly into the caller's register operations.

### Layer 2: Shared Memory (per-block, ~5 cycle latency)

Shared memory is an on-chip SRAM bank within each SM, shared by all threads in a block. It is used here exclusively for the reduction phase:

```54:55:feynman-kac-cuda.cu
    __shared__ double s_wt[THREADS_PER_BLOCK];
    __shared__ int    s_steps[THREADS_PER_BLOCK];
```

This allocates 256 * 8 + 256 * 4 = **3,072 bytes** per block. The tree reduction (lines 113-121) reads and writes shared memory in a pattern that halves the active threads each step:

```
Step 0: threads 0-127 read from [128-255], write to [0-127]   (128 active)
Step 1: threads 0-63  read from [64-127],  write to [0-63]    (64 active)
Step 2: threads 0-31  read from [32-63],   write to [0-31]    (32 active)
...
Step 7: thread 0      reads from [1],       writes to [0]     (1 active)
```

After 8 steps, `s_wt[0]` holds the sum of all 256 partial `w` values. This is critical — without shared memory, every thread would need its own `atomicAdd` to global memory (256 contended atomics per block instead of 1).

**Bank conflict analysis:** Shared memory is organized in 32 banks. In step 2 and beyond (stride <= 32), threads access consecutive addresses, mapping to consecutive banks — no conflicts. In earlier steps (stride > 32), threads 0..stride-1 access `s_wt[threadIdx.x]` and `s_wt[threadIdx.x + stride]`, which also map to distinct banks because `double` (8 bytes) on 32 banks means index `i` maps to bank `(i * 8 / 4) % 32 = (2*i) % 32`. Adjacent threads hit different banks, so the reduction is **bank-conflict-free**.

### Layer 3: Global Memory (device DRAM, ~400-600 cycle latency)

Only two tiny arrays reside in global memory:

```196:199:feynman-kac-cuda.cu
    CUDA_CHECK(cudaMalloc(&d_wt_sums,   num_points * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_step_sums, num_points * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_wt_sums,   0, num_points * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_step_sums, 0, num_points * sizeof(int)));
```

That's 23 * 8 + 23 * 4 = **276 bytes** total. These arrays are touched only once per block (at the end, by thread 0 via `atomicAdd`). Because there are 40 blocks per grid point, each `d_wt_sums[point_idx]` receives 40 atomic additions over the kernel's lifetime — very light traffic.

The `atomicAdd` for `double` (line 125) is a hardware-level read-modify-write at the L2 cache. On compute capability 6.0+, this is a native operation. The L2 cache (typically 4-6 MB) easily holds the 276-byte array, so most atomics resolve in L2 without going to DRAM.

### Layer 4: Host Memory (CPU DRAM, across PCIe/NVLink)

Two transfers happen:

1. **Host → Device** (implicit): `cudaMemset` zeros the output arrays. Minimal — 276 bytes.
2. **Device → Host** (explicit): After the kernel finishes, results are copied back:

```255:258:feynman-kac-cuda.cu
    CUDA_CHECK(cudaMemcpy(h_wt_sums,   d_wt_sums,
                           num_points * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_step_sums, d_step_sums,
                           num_points * sizeof(int),    cudaMemcpyDeviceToHost));
```

276 bytes over PCIe — effectively zero cost. The kernel launch overhead (~5-10 microseconds) dwarfs the transfer time. This is an ideal ratio: massive GPU computation, negligible data movement.

### Memory Hierarchy Summary Diagram

```
┌─────────────────────────────────────────────────────────┐
│  Per-Thread Registers (~64 per thread)                  │
│  x1, w, chk, dx, vs, vh, we, us, rng state             │
│  Latency: ~1 cycle  │  Bandwidth: highest               │
├─────────────────────────────────────────────────────────┤
│  Per-Block Shared Memory (3,072 bytes)                  │
│  s_wt[256], s_steps[256] — used only during reduction   │
│  Latency: ~5 cycles │  Bandwidth: ~1.5 TB/s per SM      │
├─────────────────────────────────────────────────────────┤
│  L1 Cache (per-SM, ~128 KB)                             │
│  Caches register spills (cuRAND state if it overflows)  │
│  Latency: ~30 cycles                                    │
├─────────────────────────────────────────────────────────┤
│  L2 Cache (shared across SMs, ~4-6 MB)                  │
│  Caches d_wt_sums[], d_step_sums[] atomics              │
│  Latency: ~200 cycles                                   │
├─────────────────────────────────────────────────────────┤
│  Device Global Memory (DRAM, several GB)                │
│  d_wt_sums[23], d_step_sums[23] — backing store         │
│  Latency: ~400-600 cycles │  Bandwidth: ~900 GB/s        │
├─────────────────────────────────────────────────────────┤
│  Host Memory (CPU DRAM, via PCIe)                       │
│  h_wt_sums[23], h_step_sums[23] — final results         │
│  Bandwidth: ~32 GB/s (PCIe 4.0 x16)                     │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Performance Limitations

### A. Warp Divergence from Variable-Length Walks (Primary Bottleneck)

This is the most significant limitation. The `while (chk < 1.0)` loop (line 82) runs a different number of iterations per thread because each random walk takes a stochastic path. Within a single warp of 32 threads:

- A walk starting near `x = 0` (center of the interval) must wander ~40,000 steps to reach the boundary at `|x| = a = 2.0`.
- A walk starting near `x = 1.9` needs only ~100 steps.

The warp cannot retire until its **slowest** thread exits the loop. The other 31 threads sit idle (their CUDA cores are masked but still allocated), wasting hardware cycles. Since threads within a block all share the same `point_idx`, threads in the same warp have the same starting `x` — this design choice mitigates cross-point divergence. But the **stochastic variation** between walks from the same starting point still causes significant intra-warp divergence: some walks happen to wander away from the boundary while others reach it quickly.

The impact scales with starting position. The blocks for `point_idx` values near the center (around `x = 0`) have the worst divergence and dominate total kernel runtime.

### B. Double-Precision Throughput Penalty

Every floating-point operation in the inner loop — `curand_uniform_double`, the potential evaluations, the weight update — uses `double` (FP64). GPU hardware ratios between FP32 and FP64 throughput are stark:

| GPU Class | FP64:FP32 ratio |
|-----------|----------------|
| Data center (A100, H100) | 1:2 |
| Professional (RTX A6000) | 1:2 |
| Consumer (RTX 3090, 4090) | 1:64 |

On a consumer GPU, this kernel would run **up to 32x slower** than if it could use single precision. The algorithm's numerical stability (exponential weighting in `w`) genuinely requires `double`, so this is largely unavoidable.

### C. Register Pressure and Occupancy

Each thread uses a substantial number of registers: the walk variables (`x1`, `w`, `chk`, etc.) are doubles (2 registers each), plus the Philox PRNG state, plus loop temporaries. A rough estimate is **40-50 registers per thread**.

With 256 threads/block and 48 registers/thread, that's 12,288 registers per block. An SM with a 65,536-register file can host at most 5 such blocks = 1,280 threads. The theoretical max is 2,048, so occupancy would be ~62%. Lower occupancy means fewer warps available for the scheduler to switch between, reducing its ability to hide instruction and memory latency.

### D. PRNG Initialization Cost

The `curand_init` call (line 74) for Philox involves a moderate amount of arithmetic to set up the counter state. With 230,000 threads all initializing simultaneously, this represents a one-time cost at the start of each thread's execution. For short walks (boundary-adjacent grid points), init time can be a significant fraction of total thread runtime. In the overall kernel, though, the long walks at interior points dominate.

### E. Tail Effect (Load Imbalance Across SMs)

The 920 blocks do not all take the same amount of time. Blocks for `point_idx` near the center of the interval run far longer than those near the boundary. The kernel cannot finish until the **last block** completes. Near the end of execution, most SMs are idle while a few SMs are still grinding through the longest-running blocks. This is a form of global load imbalance at the SM scheduling level.

With only 920 blocks and (say) 80 SMs on a large GPU, each SM gets ~11 blocks. The variance in block runtime is high, so the tail latency can be significant.

### F. Problem Size vs. GPU Capacity

At 920 blocks with 256 threads each = 235,520 threads. A modern GPU can sustain millions of resident threads. This kernel does not fully saturate the hardware. Increasing `n` from 10,000 to, say, 100,000 would create 9,200 blocks (2.3M threads), dramatically improving SM utilization and allowing much better latency hiding via warp switching. The algorithm's statistical accuracy also improves with larger `n`, so scaling up is both computationally and mathematically beneficial.

### Limitation Summary

| Limitation | Severity | Root Cause | Possible Mitigation |
|-----------|----------|------------|-------------------|
| Warp divergence | **High** | Variable walk lengths | Persistent threads / work queues |
| FP64 throughput | **High** (consumer GPUs) | Algorithm needs `double` | Use data-center GPUs; or mixed precision for non-critical ops |
| Register pressure / occupancy | **Medium** | Many doubles + PRNG state | Reduce register count via compiler flags (`--maxrregcount`) |
| PRNG init overhead | **Low-Medium** | Per-thread Philox setup | Amortized over long walks; negligible at large `n` |
| SM-level tail effect | **Medium** | Grid points with vastly different walk lengths | Interleave grid points across X-dimension to mix work |
| GPU under-saturation | **Medium** | Only 230K threads at default `n` | Increase `n` for better accuracy and utilization |

