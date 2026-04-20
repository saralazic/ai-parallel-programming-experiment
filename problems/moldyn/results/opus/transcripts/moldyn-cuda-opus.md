# GPU parallelization analysis of algorithm
_Exported on 3/21/2026 at 16:15:07 GMT+1 from Cursor (2.4.31)_

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

Let me start by exploring the project structure and understanding the codebase.

This is a classic **Lennard-Jones Molecular Dynamics (MD) simulation** — an N-body problem that is one of the canonical targets for GPU acceleration. Let me provide a detailed analysis.

---

## Overview of the Algorithm

The main simulation loop (line 280) iterates over timesteps, and each step executes:

```280:315:moldyn.c
  for (move = 1; move <= movemx; move++)
  {
    domove(3 * npart, x, vh, f, side);
    forces(npart, x, f, side, rcoff);
    ekin = mkekin(npart, f, vh, hsq2, hsq);
    vel = velavg(npart, vh, vaver, h);
    if (move < istop && fmod(move, irep) == 0)
    {
      sc = sqrt(tref / (tscale * ekin));
      dscal(3 * npart, sc, vh, 1);
      ekin = tref / tscale;
    }
    if (fmod(move, iprint) == 0)
      prnout(move, ekin, epot, tscale, vir, vel, count, npart, den);
  }
```

---

## 1. Which Computations Are Parallelizable and Why

### `forces()` — **Massively parallelizable** (highest priority)

```134:210:moldyn.c
void forces(int npart, double x[], double f[], double side, double rcoff)
{
  int i, j;
  // ...
  for (i = 0; i < npart * 3; i += 3)
  {
    // ...
    for (j = 0; j < npart * 3; j += 3)
    {
      if (i == j) continue;
      // ... pairwise distance, Lennard-Jones potential, force accumulation
    }
    f[i] += fxi;
    f[i + 1] += fyi;
    f[i + 2] += fzi;
  }
  // ...
}
```

This is the **computational bottleneck** — it is O(N²) where N = `npart`. For each particle `i`, the inner loop over all other particles `j` computes pairwise distances, applies minimum-image convention (periodic boundary conditions), evaluates the Lennard-Jones potential, and accumulates forces. Each outer-loop iteration `i` computes forces independently on particle `i` (the force on `i` only writes to `f[i], f[i+1], f[i+2]`), making the outer loop **embarrassingly parallel over particles**. The inner loop for each particle is also a reduction (accumulating `fxi, fyi, fzi, epot, vir`). On a GPU, you can assign **one thread (or one warp/block) per particle** for the outer loop, with each thread iterating over all `j` partners.

**Why it's parallelizable:** No write conflicts between outer-loop iterations — each `i` writes to its own force slot. `epot` and `vir` are reductions that GPUs handle efficiently via shared-memory or warp-level reduction.

### `domove()` — **Embarrassingly parallel**

```57:78:moldyn.c
  void
  domove(int n3, double x[], double vh[], double f[], double side){
    int i;
    for (i=0; i<n3; i++) {
      x[i] += vh[i]+f[i];
      if (x[i] < 0.0)  x[i] += side;
      if (x[i] > side) x[i] -= side;
      vh[i] += f[i];
      f[i] = 0.0;
    }
  }
```

Each iteration operates on index `i` independently. Position update, periodic boundary wrapping, velocity update, and force zeroing are all per-element with no cross-element dependencies. This maps to a simple **one-thread-per-element** GPU kernel. The branching for PBC is minimal and uniform enough to avoid significant warp divergence.

### `mkekin()` — **Parallel map + reduction**

```335:349:moldyn.c
  double
  mkekin(int npart, double f[], double vh[], double hsq2, double hsq){
    int i;
    double sum=0.0, ekin;
    for (i=0; i<3*npart; i++) {
      f[i]*=hsq2;
      vh[i]+=f[i];
      sum+=vh[i]*vh[i];
    }
    ekin=sum/hsq;
    return(ekin);
  }
```

This is a **fused map + reduction**: the per-element updates to `f[i]` and `vh[i]` are independent, and `sum` is a standard sum-reduction. On GPU, each thread handles one element's map operations, followed by a parallel reduction for `sum` using shared memory or `cub::DeviceReduce`.

### `velavg()` — **Parallel map + dual reduction**

```430:448:moldyn.c
  double
  velavg(int npart, double vh[], double vaver, double h){
    int i;
    double vaverh=vaver*h;
    double vel=0.0;
    double sq;
    extern double count;
    count=0.0;
    for (i=0; i<npart*3; i+=3){
      sq=sqrt(vh[i]*vh[i]+vh[i+1]*vh[i+1]+vh[i+2]*vh[i+2]);
      if (sq>vaverh) count++;
      vel+=sq;
    }
    vel/=h;
    return(vel);
  }
```

Each particle's speed computation is independent. `vel` and `count` are both reductions (sum and conditional count). Stride-3 access can be restructured for coalescing. Straightforward GPU kernel with **one thread per particle** and a two-variable reduction.

### `dscal()` — **Embarrassingly parallel**

```83:97:moldyn.c
  void
  dscal(int n,double sa,double sx[], int incx){
    int i,j;
    if (incx == 1) {
      for (i=0; i<n; i++)
        sx[i] *= sa;
    } else {
      for (i=0; i<n; i++) {
        sx[i * incx] *= sa;
      }
    }
  }
```

Pure element-wise scaling — trivially parallel. In practice on GPU you'd just call a BLAS-like `cublasDscal`.

### `dfill()` — **Embarrassingly parallel**

Trivial array fill. On GPU, equivalent to `cudaMemset` or a simple fill kernel.

---

## 2. Data Dependencies That May Limit Parallelization

### Timestep-level sequential dependency (the critical constraint)

The main loop is **inherently sequential across timesteps**:

```
domove(step N) → forces(step N) → mkekin(step N) → velavg(step N) → [dscal] → next step
```

- `forces()` reads positions `x[]` produced by `domove()`
- `mkekin()` reads forces `f[]` produced by `forces()` and updates velocities `vh[]`
- `velavg()` reads the updated `vh[]`
- `domove()` of the next step reads the updated `x[]`, `vh[]`, and `f[]`

**You cannot overlap timesteps.** Each function within a timestep must complete before the next begins. Parallelism is only *within* each function.

### Newton's Third Law is **not** exploited (simplifies GPU mapping)

The current `forces()` computes force on particle `i` from **all** `j` independently (full N² loop, not N²/2). This means:

- **No write conflicts** between threads computing different particles `i`
- But it **doubles the work** compared to a Newton's-3rd-law-optimized version

If you were to exploit Newton's Third Law (computing `f[i] += F_ij` and `f[j] -= F_ij` simultaneously), you'd introduce **write conflicts** on `f[j]` requiring atomics, which on GPU would be a significant bottleneck. The current full-N² approach is actually **GPU-friendly** because it avoids atomics entirely.

### Reduction variables: `epot` and `vir` in `forces()`

These are global accumulators across all particle pairs. On GPU, these require either:
- Per-thread local accumulators → block-level shared-memory reduction → global atomic add
- Or a two-pass approach (compute per-particle contributions, then reduce)

This is well-understood and not a significant limitation, but it does require careful implementation.

### `mxwell()` — **Not parallelizable** (but irrelevant)

```359:405:moldyn.c
  void
  mxwell(double vh[], int n3, double h, double tref){
    // ...
    srand48(4711);
    for (i=0; i<n3; i+=2) {
      s=2.0;
      while (s>=1.0) {
        v1=2.0*drand48()-1.0;
        v2=2.0*drand48()-1.0;
        s=v1*v1+v2*v2;
      }
      // ...
    }
    // ...
  }
```

This uses Box-Muller with sequential `drand48()` calls — inherently sequential due to the PRNG state dependency and the rejection-sampling `while` loop. However, this is only called **once during initialization**, so it has zero impact on simulation performance. On GPU, you'd replace this with `cuRAND`.

### `fcc()` — **Sequential but irrelevant**

The FCC lattice generator uses a running counter `ijk` which creates a loop-carried dependency. Called once at initialization. On GPU, you could compute `ijk` analytically from `(lg, i, j, k)` to parallelize, but it's not worth the effort.

---

## 3. Parallelization Strategy for GPU Execution

### Data layout: Array of Structures (AoS) → Structure of Arrays (SoA)

The current code stores particle coordinates interleaved as `[x0, y0, z0, x1, y1, z1, ...]` (AoS layout). This causes **non-coalesced memory access** on GPU because adjacent threads access elements 3 apart. The first transformation should be:

```
// Current (AoS): x[i*3], x[i*3+1], x[i*3+2]
// Target  (SoA): pos_x[i], pos_y[i], pos_z[i]
```

This enables coalesced reads where thread `i` reads `pos_x[i]`, and adjacent threads read adjacent memory locations.

### Kernel design for `forces()` — The critical kernel

**Strategy: One thread-block per particle `i`, threads within the block cooperatively iterate over `j`.**

```
__global__ void forces_kernel(int npart, double *px, double *py, double *pz,
                               double *fx, double *fy, double *fz,
                               double sideh, double side, double rcoffs,
                               double *d_epot, double *d_vir) {
    int i = blockIdx.x;              // one block per particle
    double xi = px[i], yi = py[i], zi = pz[i];
    double fxi = 0, fyi = 0, fzi = 0;
    double local_epot = 0, local_vir = 0;

    // Threads within block collaboratively loop over j
    for (int j = threadIdx.x; j < npart; j += blockDim.x) {
        if (i == j) continue;
        // ... minimum image, distance, LJ force computation ...
        // accumulate into fxi, fyi, fzi, local_epot, local_vir
    }

    // Warp/block reduction for fxi, fyi, fzi, local_epot, local_vir
    // Thread 0 writes f[i] and atomicAdd to global epot, vir
}
```

**Alternative (tiled with shared memory):** Load tiles of particle positions into shared memory to reduce global memory bandwidth:

```
__shared__ double tile_x[TILE_SIZE], tile_y[TILE_SIZE], tile_z[TILE_SIZE];
for (int tile = 0; tile < npart; tile += TILE_SIZE) {
    // Cooperatively load tile into shared memory
    tile_x[threadIdx.x] = px[tile + threadIdx.x];  // coalesced
    // ...
    __syncthreads();
    // Each thread computes interaction of particle i with all in tile
    for (int j = 0; j < TILE_SIZE; j++) { ... }
    __syncthreads();
}
```

This is the classic **tiled N-body** approach and dramatically reduces global memory traffic.

### Kernel design for `domove()`, `mkekin()`, `dscal()`, `dfill()`

These are simple element-wise or map-reduce kernels:

```
// domove: one thread per coordinate component
thread i: x[i] += vh[i] + f[i]; apply PBC; vh[i] += f[i]; f[i] = 0;

// mkekin: one thread per component, then parallel reduction for sum
// velavg: one thread per particle, then parallel reduction for vel and count
```

These are straightforward and can use libraries like Thrust or CUB for the reduction portion.

### Memory management strategy

- Allocate `x, vh, f` (in SoA form as `px, py, pz, vhx, vhy, vhz, fx, fy, fz`) on GPU **once** at the start
- Keep all data **resident on GPU** for the entire simulation loop
- Only transfer back to host for the occasional `prnout()` call (every `iprint` steps)
- `epot`, `vir`, `ekin`, `vel`, `count` are small scalars — transfer via `cudaMemcpy` or pinned memory

### Kernel fusion opportunities

`mkekin()` updates `f[]` and `vh[]` and computes a reduction — this could potentially be **fused with** `domove()` of the next timestep or at least with the force scaling. However, the reduction for `ekin` is needed before `velavg` and the conditional `dscal`, so fusion options are limited by the control flow at lines 303-308.

---

## 4. Expected Performance Challenges

### (a) Memory bandwidth bottleneck in `forces()`

The force computation is **compute-bound for large N** but can become **memory-bandwidth-bound** if the particle data doesn't fit in L1/shared memory. Each thread needs to read positions of all other particles. For `npart` = 4 * mm³:

| mm  | npart  | Position data (SoA, double) |
|-----|--------|-----------------------------|
| 10  | 4,000  | ~94 KB                      |
| 15  | 13,500 | ~316 KB                     |
| 20  | 32,000 | ~750 KB                     |

For moderate sizes, tiled shared-memory approach is essential. For large N, the data won't fit in shared memory at all and you'll rely on L2 cache.

### (b) Double-precision performance

The code uses `double` throughout. Consumer GPUs (GeForce) have **1/32 to 1/2 the FP64 throughput** compared to FP32. On a typical gaming GPU, you'd see 1/32 FP64 rate. Data-center GPUs (A100, H100) have 1/2 FP64 rate. This is a **major performance consideration** — if the physics permits, switching to single precision (`float`) could yield 2-32x speedup depending on the GPU.

### (c) Warp divergence

The `if (rd <= rcoffs)` branch in `forces()` at line 184 means some threads within a warp will take the expensive LJ computation path and others won't. With a cutoff radius, especially at low density, a significant fraction of pairs will be outside cutoff. This causes **warp divergence** where idle threads waste cycles waiting. Mitigations:
- Sort particles spatially (cell lists) so neighboring threads process spatially-close particles with similar interaction patterns
- Use compaction to remove non-interacting pairs before the expensive computation

### (d) Kernel launch overhead vs. useful work

For small systems, the overhead of launching 5+ kernels per timestep (domove, forces, mkekin, velavg, possibly dscal) and synchronizing between them can dominate. Mitigation:
- **CUDA Graphs** to record and replay the kernel sequence, reducing launch overhead
- Kernel fusion where possible
- Ensure enough parallelism (large N) to amortize launch costs

### (e) Reduction overhead

`forces()` needs global reductions for `epot` and `vir`. `mkekin()` needs a reduction for `sum`. `velavg()` needs reductions for `vel` and `count`. Each reduction requires synchronization across the GPU. For small N, this synchronization cost is proportionally large.

### (f) The O(N²) scaling itself

The current algorithm is **all-pairs** — no neighbor list, no cell list. This scales as O(N²) and becomes prohibitive for large systems regardless of GPU acceleration. For production-scale MD:
- Implement **cell lists** or **Verlet neighbor lists** to reduce to O(N) per timestep
- On GPU, cell lists translate to spatial hashing / grid-based binning which also improves memory locality and reduces warp divergence

However, the cutoff `rcoff = mm/4.0` suggests a short-range potential where cell lists would be highly effective.

### (g) Host-device synchronization for control flow

Lines 303-308 have conditional logic that depends on `ekin` (computed on GPU):

```303:308:moldyn.c
    if (move < istop && fmod(move, irep) == 0)
    {
      sc = sqrt(tref / (tscale * ekin));
      dscal(3 * npart, sc, vh, 1);
      ekin = tref / tscale;
    }
```

This requires reading `ekin` back to the host to evaluate the condition, which forces a **GPU-to-host synchronization point** and pipeline stall. Mitigations:
- Since the condition only depends on `move`, `istop`, and `irep` (all known ahead of time), you can **predict on the host** whether scaling is needed without waiting for `ekin` — you only need the *value* of `ekin` for computing `sc`, which can be done in a device-side kernel that conditionally scales

---

## Summary Table

| Function | Complexity | Parallelizable? | GPU Strategy | Priority |
|----------|-----------|-----------------|-------------|----------|
| `forces()` | O(N²) | Yes — over particles | Tiled N-body, 1 block/particle | **Critical** |
| `domove()` | O(N) | Yes — embarrassingly | 1 thread/element | High |
| `mkekin()` | O(N) | Yes — map + reduce | 1 thread/element + CUB reduce | High |
| `velavg()` | O(N) | Yes — map + reduce | 1 thread/particle + reduce | Medium |
| `dscal()` | O(N) | Yes — embarrassingly | cuBLAS or 1 thread/element | Low |
| `dfill()` | O(N) | Yes — embarrassingly | cudaMemset | Low |
| `fcc()` | O(N) | Possible but unnecessary | Init only — run on host | None |
| `mxwell()` | O(N) | No (sequential PRNG) | Init only — cuRAND or host | None |
| `prnout()` | O(1) | N/A | Host-side I/O | None |

The single most impactful optimization is GPU-parallelizing `forces()`, which dominates runtime at O(N²). The O(N) kernels should also live on the GPU to **avoid repeated host-device data transfers**, even though individually they add modest speedup. Keeping all data GPU-resident and only transferring scalars (`ekin`, `vel`, `count`, `epot`, `vir`) to the host for control flow and output is the key architectural decision.

---

**User**

Based on your analysis of code and proposed strategy GPU paralelization:
1. Implement CUDA kernel(s)
2. Organize threads into blocks and grids appropriately
3. Manage memory transfers between host (CPU) and device (GPU)
4. Use suitable CUDA memory types (global, shared, local)

---

**Cursor**

The CUDA implementation is complete in `moldyn.cu`. Here's a walkthrough of the design decisions across all four requirements:

---

## 1. CUDA Kernels (6 total)

| Kernel | Purpose | Original function |
|---|---|---|
| `dfill_kernel` | Fill array with constant | `dfill()` |
| `domove_kernel` | Position integration + PBC + partial velocity update + zero forces | `domove()` |
| `dscal_kernel` | Scale array by scalar | `dscal()` |
| `forces_kernel` | Pairwise Lennard-Jones force computation with **tiled shared-memory** N-body | `forces()` |
| `mkekin_kernel` | Scale forces, update velocities, compute kinetic energy (map + reduction) | `mkekin()` |
| `velavg_kernel` | Average velocity + threshold count (map + dual reduction) | `velavg()` |

The host-only functions (`fcc`, `mxwell`, `prnout`, `secnds`) remain unchanged on the CPU since they run only at initialization or for I/O.

---

## 2. Thread/Block Organization

**Simple element-wise kernels** (`domove`, `mkekin`, `dscal`, `dfill`):
- Block size: **256 threads** (`BLOCK_SIZE`)
- Grid: `ceil(n3 / 256)` blocks
- Each thread handles one array element — adjacent threads access adjacent memory (coalesced)

**Forces kernel** (the O(N^2) bottleneck):
- Block size: **128 threads** (`TILE_SIZE`)
- Grid: `ceil(npart / 128)` blocks
- **One thread per particle `i`** — each thread iterates through all `j` particles via shared-memory tiles

**Velavg kernel**:
- Block size: **256 threads**
- Grid: `ceil(npart / 256)` blocks
- One thread per particle (reads 3 consecutive velocity components)

---

## 3. Memory Transfer Strategy

The key principle is **keeping data GPU-resident**:

```
Host → Device:  ONCE at initialization (x, vh, f arrays)
Device → Host:  Only small scalars, only when needed:
  - ekin (every step — needed for temperature control logic)
  - vel, count (every step — needed for diagnostics)
  - epot, vir (only on print steps — lines 338-345)
```

Reduction accumulators (`d_epot`, `d_vir`, `d_sum`, `d_vel`, `d_count`) are zeroed with `cudaMemset` before each kernel that writes to them. The particle arrays (`d_x`, `d_vh`, `d_f`) are **never transferred back** — all physics runs entirely on the GPU.

---

## 4. CUDA Memory Types Used

**Global memory** — all particle arrays and reduction scalars:

```33:37:moldyn.cu
/* Block sizes — tunable for different GPU architectures */
#define BLOCK_SIZE 256   /* domove, mkekin, dscal, dfill, velavg        */
#define TILE_SIZE  128   /* forces kernel block size AND shared-mem tile */
```

```297:304:moldyn.cu
    double *d_x, *d_vh, *d_f;
    CUDA_CHECK(cudaMalloc(&d_x,  n3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vh, n3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_f,  n3 * sizeof(double)));

    double *d_epot, *d_vir, *d_sum, *d_vel, *d_count;
    CUDA_CHECK(cudaMalloc(&d_epot,  sizeof(double)));
```

**Shared memory** — used in three places:

1. **Forces kernel** — position tile (3 arrays x 128 doubles = 3 KB) so all threads in a block read the same j-positions from fast shared memory instead of slow global memory, reducing traffic by 128x:

```107:109:moldyn.cu
    __shared__ double tile_x[TILE_SIZE];
    __shared__ double tile_y[TILE_SIZE];
    __shared__ double tile_z[TILE_SIZE];
```

2. **Forces kernel** — reduction scratch (2 arrays x 128 doubles = 2 KB) for block-level tree reduction of `epot` and `vir`:

```161:163:moldyn.cu
    __shared__ double s_epot[TILE_SIZE];
    __shared__ double s_vir[TILE_SIZE];
```

3. **mkekin/velavg kernels** — reduction scratch for kinetic energy sum and velocity/count sums.

**Registers (local memory)** — per-thread accumulators like `xi, yi, zi, fxi, fyi, fzi, my_epot, my_vir` in the forces kernel. These live entirely in registers for maximum throughput during the inner loop.

---

## Compilation

```bash
nvcc -O3 -arch=sm_60 -o moldyn_cuda moldyn.cu -lm
./moldyn_cuda 10
```

The `-arch=sm_60` flag targets Pascal and newer (required for native `double` `atomicAdd`). Adjust to match your GPU.

---

**User**

Explain 
1. how this CUDA solution maps onto GPU hardware architecture.
2. How threads and thread blocks are organized
3. How memory hierarchy is used
4. What performance limitations may exist

---

**Cursor**

Here is a detailed explanation of all four points, grounded in the specific code.

---

## 1. How This CUDA Solution Maps Onto GPU Hardware Architecture

A GPU is organized as an array of **Streaming Multiprocessors (SMs)**, each containing many CUDA cores, a register file, shared memory, and an L1 cache. The CUDA runtime maps software abstractions (grids, blocks, warps) onto this hardware.

### Kernel-to-SM mapping

Each kernel launch creates a **grid of thread blocks**. The GPU's block scheduler distributes these blocks across SMs. Multiple blocks can reside on the same SM simultaneously (limited by register usage, shared memory, and thread count). This implementation launches 6 different kernels per timestep in sequence on the default stream:

```512:523:moldyn.cu
    for (int move = 1; move <= movemx; move++) {

        /* 1. Move particles, apply PBC, partial velocity update, zero f */
        domove_kernel<<<grid_n3, block_std>>>(n3, d_x, d_vh, d_f, side);

        /* 2. Pairwise LJ forces — the O(N²) computational bottleneck.
         *    Zero the reduction accumulators first (all-zero bytes ≡ 0.0
         *    in IEEE-754 double). */
        CUDA_CHECK(cudaMemset(d_epot, 0, sizeof(double)));
        CUDA_CHECK(cudaMemset(d_vir,  0, sizeof(double)));
        forces_kernel<<<grid_forces, block_forces>>>(
            npart, d_x, d_f, sideh, side, rcoffs, d_epot, d_vir);
```

All six kernels execute on the **default (null) stream**, meaning the GPU implicitly serializes them: `domove` finishes before `forces` begins, `forces` finishes before `mkekin` begins, and so on. This mirrors the data dependency chain from the original algorithm. Between kernel launches, the GPU hardware is idle for a brief period (the **kernel launch gap**, typically 5--10 microseconds).

### How the forces kernel saturates the hardware

The forces kernel is where the GPU spends virtually all its time. Consider `mm = 10`, so `npart = 4000`:

```503:503:moldyn.cu
    dim3 grid_forces   ((npart + TILE_SIZE  - 1) / TILE_SIZE);
```

- Grid: `ceil(4000 / 128)` = **32 blocks**
- Each block: **128 threads**
- Total: **4,096 threads** (32 trailing threads are idle, guarded by `if (i < npart)`)

On a modern GPU with, say, 80 SMs, only 32 blocks are available, so many SMs would sit idle. With `mm = 15` (`npart = 13,500`), you get **106 blocks** and better SM utilization. With `mm = 20` (`npart = 32,000`), you get **250 blocks** -- enough to fill an A100's 108 SMs with 2+ blocks each.

Each thread in `forces_kernel` performs **O(N) floating-point work** (the entire inner j-loop), so even a single thread does substantial computation. The GPU's advantage comes from running thousands of these threads simultaneously, each on a different CUDA core.

### Simple kernels and hardware underutilization

The O(N) kernels (`domove`, `mkekin`, `dscal`, `velavg`) produce far more threads but each thread does negligible work:

```500:501:moldyn.cu
    dim3 grid_n3       ((n3    + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_npart    ((npart + BLOCK_SIZE - 1) / BLOCK_SIZE);
```

For `mm = 10`: `n3 = 12,000`, so `grid_n3 = 47 blocks` of 256 threads = 12,032 threads. Each thread in `domove_kernel` performs ~5 arithmetic operations. The entire kernel finishes in microseconds, meaning the kernel launch overhead is a significant fraction of the total kernel time. These kernels are **memory-bandwidth-bound**, not compute-bound.

---

## 2. How Threads and Thread Blocks Are Organized

### Two distinct block-size strategies

The implementation uses two block sizes chosen for different workload characteristics:

```28:29:moldyn.cu
#define BLOCK_SIZE 256   /* domove, mkekin, dscal, dfill, velavg        */
#define TILE_SIZE  128   /* forces kernel block size AND shared-mem tile */
```

**BLOCK_SIZE = 256** for element-wise and reduction kernels: this is a standard choice that balances occupancy with register pressure. On most GPUs, 256 threads/block allows 4--8 blocks per SM (out of a typical max of 1,536--2,048 threads per SM).

**TILE_SIZE = 128** for the forces kernel: smaller because each thread uses many registers (the inner loop has ~15 live `double` variables = ~30 32-bit registers), and because the shared memory tile size must match the block size for the cooperative load pattern to work.

### Thread-to-data mapping

**Element-wise kernels** (domove, mkekin, dscal, dfill) use a **1:1 thread-to-element** mapping:

```68:69:moldyn.cu
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n3) return;
```

Thread 0 of block 0 handles element 0, thread 1 handles element 1, etc. This gives perfect coalescing since adjacent threads access adjacent memory locations.

**Particle-wise kernels** (forces, velavg) use a **1:1 thread-to-particle** mapping:

```127:127:moldyn.cu
    int i = blockIdx.x * blockDim.x + threadIdx.x;
```

In `forces_kernel`, thread `i` "owns" particle `i` — it reads `x[i*3 .. i*3+2]` and writes `f[i*3 .. i*3+2]`. Because each thread writes exclusively to its own particle's force slots, there are **no write conflicts** and no atomics needed for the force array.

### Warp execution model

Within each block, threads are physically grouped into **warps of 32 threads** that execute in lockstep on the SM's CUDA cores. The forces kernel's inner loop at line 154 iterates over the shared-memory tile:

```154:186:moldyn.cu
            for (int k = 0; k < tile_end; k++) {
                int j = t * TILE_SIZE + k;
                if (i == j) continue;
                // ... distance, PBC, LJ computation ...
                if (rd <= rcoffs) {
                    // ... expensive branch ...
                }
            }
```

All 32 threads in a warp execute the same `k` iteration simultaneously. They all read `tile_x[k]` from shared memory (broadcast, no bank conflict since same address). The `if (i == j) continue` causes one thread per warp to skip one iteration -- minimal divergence. The `if (rd <= rcoffs)` branch is where warp divergence can be more substantial (discussed in section 4).

### The cooperative tile load pattern

The tile load demonstrates how threads in a block cooperate:

```142:149:moldyn.cu
    for (int t = 0; t < num_tiles; t++) {
        /* ---- cooperative tile load into shared memory ---- */
        int j_idx = t * TILE_SIZE + threadIdx.x;
        if (j_idx < npart) {
            tile_x[threadIdx.x] = x[j_idx * 3];
            tile_y[threadIdx.x] = x[j_idx * 3 + 1];
            tile_z[threadIdx.x] = x[j_idx * 3 + 2];
        }
        __syncthreads();
```

Each of the 128 threads loads exactly one j-particle's coordinates. Thread 0 loads particle `t*128 + 0`, thread 1 loads particle `t*128 + 1`, etc. The `__syncthreads()` barrier ensures the entire tile is populated before any thread begins reading from it. Without this barrier, a thread could read a shared-memory slot that hasn't been written yet by another warp.

### The block-level reduction pattern

Reductions (summing `epot`, `vir`, kinetic energy, velocity, count) use the classic shared-memory tree reduction:

```206:212:moldyn.cu
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_epot[threadIdx.x] += s_epot[threadIdx.x + stride];
            s_vir[threadIdx.x]  += s_vir[threadIdx.x + stride];
        }
        __syncthreads();
    }
```

This takes `log2(128) = 7` steps. In each step, the active thread count halves. After 7 steps, `s_epot[0]` holds the block's total. Thread 0 then performs a single `atomicAdd` to the global accumulator:

```214:217:moldyn.cu
    if (threadIdx.x == 0) {
        atomicAdd(d_epot, s_epot[0]);
        atomicAdd(d_vir,  s_vir[0]);
    }
```

For 32 blocks, only 32 `atomicAdd` operations hit global memory -- far fewer than the 4,000 that a per-thread atomic would produce.

---

## 3. How the Memory Hierarchy Is Used

### Level 1: Registers (fastest, per-thread)

Each thread has a private register file. In the forces kernel, the particle's own position and its running force accumulators live in registers for the duration of the kernel:

```129:137:moldyn.cu
    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (i < npart) {
        xi = x[i * 3];
        yi = x[i * 3 + 1];
        zi = x[i * 3 + 2];
    }

    double fxi = 0.0, fyi = 0.0, fzi = 0.0;
    double my_epot = 0.0, my_vir = 0.0;
```

These 8 doubles (= 16 32-bit registers) are loaded once from global memory and then read/written at register speed (zero-latency for the core) throughout the entire j-loop across all tiles. The inner-loop temporaries (`xx, yy, zz, rd, rrd, rrd2, ...`) also live in registers but are transient per iteration. An approximate register budget per thread:

| Variables | Count | Registers (32-bit) |
|---|---|---|
| `xi, yi, zi` | 3 doubles | 6 |
| `fxi, fyi, fzi` | 3 doubles | 6 |
| `my_epot, my_vir` | 2 doubles | 4 |
| Inner-loop temps (`xx` through `r148`) | ~8 doubles | 16 |
| Index variables (`i, t, k, j, j_idx, ...`) | ~5 ints | 5 |
| **Total** | | **~37** |

A typical GPU has 65,536 registers per SM. At 37 registers/thread and 128 threads/block, one block uses ~4,736 registers, allowing potentially 13 blocks per SM from registers alone (though other limits apply).

### Level 2: Shared Memory (fast, per-block)

Shared memory is a programmer-managed scratchpad on each SM, accessible to all threads within a block at ~100x the bandwidth of global memory and ~5 ns latency (vs. ~400 ns for global).

**Forces kernel** uses shared memory for two purposes:

*Purpose 1 -- Position tile cache (3 KB):*

```123:125:moldyn.cu
    __shared__ double tile_x[TILE_SIZE];
    __shared__ double tile_y[TILE_SIZE];
    __shared__ double tile_z[TILE_SIZE];
```

128 particles x 3 doubles x 8 bytes = 3,072 bytes. This is the core of the tiling optimization. Without it, each thread would independently read all N j-positions from global memory, resulting in N x N total global reads. With tiling, each block reads N positions total (amortized across 128 threads), and each thread reads from fast shared memory -- an **128x reduction** in global memory traffic.

During the inner loop, reads like `tile_x[k]` are broadcast to all 32 threads in a warp (same address, single shared-memory transaction). Reads like `tile_x[threadIdx.x]` during the cooperative load access distinct banks with stride-1, giving full bandwidth with no bank conflicts.

*Purpose 2 -- Reduction scratch (2 KB):*

```199:200:moldyn.cu
    __shared__ double s_epot[TILE_SIZE];
    __shared__ double s_vir[TILE_SIZE];
```

**mkekin kernel** uses shared memory for a single reduction:

```233:233:moldyn.cu
    __shared__ double sdata[BLOCK_SIZE];
```

256 doubles x 8 bytes = 2,048 bytes.

**velavg kernel** uses shared memory for a dual reduction:

```268:269:moldyn.cu
    __shared__ double s_vel[BLOCK_SIZE];
    __shared__ double s_cnt[BLOCK_SIZE];
```

256 doubles x 2 x 8 bytes = 4,096 bytes.

Total shared memory per block:

| Kernel | Shared Memory | Typical SM Limit |
|---|---|---|
| forces | 5,120 bytes (5 KB) | 48--100 KB |
| mkekin | 2,048 bytes (2 KB) | 48--100 KB |
| velavg | 4,096 bytes (4 KB) | 48--100 KB |

All well under the limit, allowing high block-level occupancy.

### Level 3: Global Memory (large, slow, device-wide)

All particle data arrays reside in global memory (GDDR6/HBM on the GPU):

```473:476:moldyn.cu
    double *d_x, *d_vh, *d_f;
    CUDA_CHECK(cudaMalloc(&d_x,  n3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vh, n3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_f,  n3 * sizeof(double)));
```

For `mm = 15` (`npart = 13,500`): `3 arrays x 40,500 doubles x 8 bytes` = **~950 KB**. Trivially fits in GPU memory (typically 8--80 GB), and significant portions fit in the L2 cache (typically 4--40 MB on modern GPUs).

The five reduction scalars also live in global memory but are only 40 bytes total:

```478:483:moldyn.cu
    double *d_epot, *d_vir, *d_sum, *d_vel, *d_count;
    CUDA_CHECK(cudaMalloc(&d_epot,  sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vir,   sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum,   sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel,   sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(double)));
```

### Level 4: Host (CPU) Memory -- PCIe/NVLink transfers

The design minimizes host-device transfers. The bulk arrays are copied **once** at initialization:

```486:491:moldyn.cu
    CUDA_CHECK(cudaMemcpy(d_x,  h_x,  n3 * sizeof(double),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vh, h_vh, n3 * sizeof(double),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f,  h_f,  n3 * sizeof(double),
                           cudaMemcpyHostToDevice));
```

After that, only scalars cross the PCIe bus. Per timestep: `ekin` (always), `vel` and `count` (always), `epot` and `vir` (only on print steps). That is 3--5 `cudaMemcpy` calls of 8 bytes each. The particle arrays **never** return to the host.

---

## 4. Performance Limitations

### (a) Double-precision throughput

Every array and every computation uses `double` (64-bit floating point). Consumer GPUs (GeForce) have severely limited FP64 throughput:

| GPU Class | FP64 : FP32 ratio |
|---|---|
| GeForce (gaming) | 1:32 |
| Professional (Quadro/A-series) | 1:2 |
| Data center (A100, H100) | 1:2 |

On a GeForce RTX 4090 with 82 TFLOPS FP32, FP64 performance is only ~2.6 TFLOPS. The forces kernel's inner loop (lines 172--184) performs ~20 FP64 operations per pair interaction. For `npart = 13,500`, there are ~182 million pair evaluations per step, requiring ~3.6 GFLOP -- feasible even at reduced FP64 rates, but a 32x handicap relative to FP32 potential.

### (b) Warp divergence in the cutoff branch

The distance-cutoff check at line 171 causes the critical warp-divergence issue:

```169:185:moldyn.cu
                double rd = xx * xx + yy * yy + zz * zz;

                if (rd <= rcoffs) {
                    double rrd  = 1.0 / rd;
                    double rrd2 = rrd * rrd;
                    double rrd3 = rrd2 * rrd;
                    double rrd4 = rrd2 * rrd2;
                    double rrd6 = rrd2 * rrd4;
                    double rrd7 = rrd6 * rrd;
                    double r148 = rrd7 - 0.5 * rrd4;

                    my_epot += (rrd6 - rrd3);
                    my_vir  -= rd * r148;
                    fxi     += xx * r148;
                    fyi     += yy * r148;
                    fzi     += zz * r148;
                }
```

Within a warp of 32 threads, different particles `i` are at different spatial positions. For a given j-particle in the tile, some threads may find `rd <= rcoffs` (within cutoff) while others find `rd > rcoffs` (beyond cutoff). When this happens, the warp must execute **both paths** -- threads that took the `false` branch are masked off during the `true` branch's execution, wasting their CUDA core cycles.

The severity depends on the cutoff radius relative to the box size. With `rcoff = mm/4`, the cutoff-to-box ratio is 1/4, meaning roughly `(4/3 * pi * rcoff^3) / side^3` of the volume is within cutoff of any particle. At `den = 0.83134`, a significant fraction of pairs will be beyond cutoff, making divergence substantial.

### (c) Non-coalesced global memory access in forces

The position array uses Array-of-Structures (AoS) layout: `[x0, y0, z0, x1, y1, z1, ...]`. When the forces kernel loads the tile:

```144:147:moldyn.cu
        if (j_idx < npart) {
            tile_x[threadIdx.x] = x[j_idx * 3];
            tile_y[threadIdx.x] = x[j_idx * 3 + 1];
            tile_z[threadIdx.x] = x[j_idx * 3 + 2];
```

Adjacent threads (threadIdx.x = 0, 1, 2, ...) access `x[0], x[3], x[6], ...` -- a **stride-3 pattern**. A warp of 32 threads spans addresses 0 through 93 (in double units), which is 94 * 8 = 752 bytes across 12 cache lines, whereas perfectly coalesced access would touch only 4 cache lines (256 bytes). This is a **3x memory transaction overhead**.

The tiling amortizes this: each block loads `N / TILE_SIZE` tiles, each tile requiring 3 non-coalesced loads per thread. Without tiling, each thread would do `N * 3` non-coalesced loads. So tiling reduces the total non-coalesced load count by 128x, making this a secondary concern. A Structure-of-Arrays (SoA) layout (`px[], py[], pz[]` as separate arrays) would eliminate it entirely.

### (d) Kernel launch overhead and pipeline stalls

Each timestep involves this sequence of GPU operations:

```
domove_kernel → cudaMemset(x2) → forces_kernel → cudaMemset → mkekin_kernel → cudaMemcpy(D→H) → cudaMemset(x2) → velavg_kernel → cudaMemcpy(D→H, x2) → [dscal_kernel] → [cudaMemcpy(D→H, x2)]
```

That is **5--6 kernel launches** and **3--7 cudaMemcpy/cudaMemset** calls per step. Each kernel launch costs ~5--10 us. Each `cudaMemcpy` DeviceToHost is synchronous, meaning the CPU **blocks** until the GPU finishes all preceding work and delivers the 8-byte result.

The `cudaMemcpy` after `mkekin_kernel` at line 530 is the most impactful stall:

```529:532:moldyn.cu
        double h_sum;
        CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(double),
                               cudaMemcpyDeviceToHost));
        double ekin = h_sum / hsq;
```

This forces the CPU to wait for `domove_kernel` + `forces_kernel` + `mkekin_kernel` to all complete before it can launch `velavg_kernel`. The GPU pipeline is fully drained at this point. The same happens again at lines 541--544 after `velavg_kernel`. For small N where kernel execution is fast, these synchronization points dominate wall-clock time.

### (e) Insufficient parallelism for small N

For `mm = 5` (`npart = 500`):
- forces: `ceil(500/128)` = **4 blocks** of 128 threads = **512 threads**
- domove: `ceil(1500/256)` = **6 blocks**

A GPU with 80+ SMs has thousands of cores sitting idle at 4 blocks. The **occupancy** (active warps / maximum warps per SM) approaches zero. The GPU cannot hide memory latency because there aren't enough warps to switch between. This is the classic "not enough parallelism" problem -- the O(N^2) force computation would only be ~250,000 pair evaluations, which a modern CPU can do in milliseconds. The GPU launch overhead and synchronization costs would likely make the CUDA version *slower* than the CPU version for small N.

### (f) Atomic contention on reduction variables

After each block's shared-memory reduction, thread 0 performs an `atomicAdd` to global memory:

```214:217:moldyn.cu
    if (threadIdx.x == 0) {
        atomicAdd(d_epot, s_epot[0]);
        atomicAdd(d_vir,  s_vir[0]);
    }
```

For the forces kernel with 32 blocks, 32 `atomicAdd` operations hit the same two memory locations. On modern GPUs with L2-level atomics, this serialization is minor. But for the mkekin kernel with `grid_n3 = 47` blocks (at `mm = 10`) up to hundreds of blocks at larger sizes, the contention increases. Each `atomicAdd` on doubles takes ~50--100 ns in the worst case, so 200 blocks x 100 ns = 20 us of serialized atomic time -- small but non-zero.

### (g) O(N^2) algorithmic scaling

This is not a CUDA-specific limitation but a fundamental one: the all-pairs force computation scales as O(N^2). For `mm = 20` (`npart = 32,000`), there are ~1 billion pair evaluations per timestep. At `mm = 30` (`npart = 108,000`), this grows to ~11.7 billion. No amount of GPU parallelism changes the asymptotic complexity. Production MD codes use **neighbor lists** or **cell lists** to reduce this to O(N), which would also benefit from GPU parallelization but require a fundamentally different kernel structure.

