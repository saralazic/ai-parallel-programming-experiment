/*
 * moldyn.cu — CUDA GPU implementation of Lennard-Jones Molecular Dynamics
 *
 * Ported from the OpenMP C version (moldyn.c).  All particle data lives on
 * the GPU for the duration of the simulation; only small scalar results
 * (kinetic energy, potential energy, virial, velocity, count) are
 * transferred back to the host when needed for control flow or I/O.
 *
 * CUDA memory hierarchy usage:
 *   - Global memory : particle arrays (x, vh, f) and reduction scalars
 *   - Shared memory : position tiles in forces kernel; reduction scratch
 *                     buffers in forces, mkekin, and velavg kernels
 *   - Registers     : per-thread accumulators (forces, epot, vir, etc.)
 *
 * Compile:
 *   nvcc -O3 -arch=sm_60 -o moldyn_cuda moldyn.cu -lm
 * Run:
 *   ./moldyn_cuda <mm>
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

/* Block sizes — tunable for different GPU architectures */
#define BLOCK_SIZE 256   /* domove, mkekin, dscal, dfill, velavg        */
#define TILE_SIZE  128   /* forces kernel block size AND shared-mem tile */

#define CUDA_CHECK(call) do {                                              \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(err));              \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

/* ================================================================
 *  CUDA Kernels
 * ================================================================ */

/*
 * dfill_kernel — set every element of a device array to a constant.
 *
 * Grid : ceil(n / BLOCK_SIZE)   Block : BLOCK_SIZE
 * Memory : global (write-only, fully coalesced).
 */
__global__ void dfill_kernel(int n, double val, double *a)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        a[i] = val;
}

/*
 * domove_kernel — integrate positions, apply periodic boundary conditions,
 *                 partially update velocities, and zero the force array.
 *
 * Grid : ceil(n3 / BLOCK_SIZE)   Block : BLOCK_SIZE
 * Memory : global (x, vh, f — each element independent, coalesced access).
 *          Registers for per-element temporaries.
 */
__global__ void domove_kernel(int n3, double *x, double *vh,
                              double *f, double side)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n3) return;

    x[i] += vh[i] + f[i];

    if (x[i] < 0.0)  x[i] += side;
    if (x[i] > side)  x[i] -= side;

    vh[i] += f[i];
    f[i]   = 0.0;
}

/*
 * dscal_kernel — multiply every element of an array by a scalar.
 *
 * Grid : ceil(n / BLOCK_SIZE)   Block : BLOCK_SIZE
 * Memory : global (coalesced read-modify-write).
 */
__global__ void dscal_kernel(int n, double sa, double *sx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        sx[i] *= sa;
}

/*
 * forces_kernel — pairwise Lennard-Jones force computation with tiled
 *                 shared-memory optimisation (classic N-body tiling).
 *
 * Grid  : ceil(npart / TILE_SIZE) blocks
 * Block : TILE_SIZE threads  (one thread ↔ one particle i)
 *
 * Shared memory (per block):
 *   tile_x/y/z [TILE_SIZE]  — cached j-particle positions (3 KB)
 *   s_epot/vir [TILE_SIZE]  — block-level reduction scratch  (2 KB)
 *
 * Algorithm:
 *   Each thread owns particle i (position in registers).  The j-loop
 *   is split into tiles of TILE_SIZE particles.  At each tile step
 *   the entire block cooperatively loads the tile into shared memory,
 *   then every thread reads it to evaluate interactions.  This reduces
 *   global-memory traffic by a factor of TILE_SIZE compared with a
 *   naïve implementation.
 *
 *   After all tiles are processed, per-thread force results are written
 *   to global memory (no conflicts — one thread per particle).  The
 *   partial epot/vir values are reduced within the block using shared
 *   memory and then atomically accumulated into device-global totals.
 */
__global__ void forces_kernel(int npart,
                              const double * __restrict__ x,
                              double       * __restrict__ f,
                              double sideh, double side, double rcoffs,
                              double *d_epot, double *d_vir)
{
    __shared__ double tile_x[TILE_SIZE];
    __shared__ double tile_y[TILE_SIZE];
    __shared__ double tile_z[TILE_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (i < npart) {
        xi = x[i * 3];
        yi = x[i * 3 + 1];
        zi = x[i * 3 + 2];
    }

    double fxi = 0.0, fyi = 0.0, fzi = 0.0;
    double my_epot = 0.0, my_vir = 0.0;

    int num_tiles = (npart + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        /* ---- cooperative tile load into shared memory ---- */
        int j_idx = t * TILE_SIZE + threadIdx.x;
        if (j_idx < npart) {
            tile_x[threadIdx.x] = x[j_idx * 3];
            tile_y[threadIdx.x] = x[j_idx * 3 + 1];
            tile_z[threadIdx.x] = x[j_idx * 3 + 2];
        }
        __syncthreads();

        /* ---- compute interactions with all particles in tile ---- */
        if (i < npart) {
            int tile_end = min(TILE_SIZE, npart - t * TILE_SIZE);
            for (int k = 0; k < tile_end; k++) {
                int j = t * TILE_SIZE + k;
                if (i == j) continue;

                double xx = xi - tile_x[k];
                double yy = yi - tile_y[k];
                double zz = zi - tile_z[k];

                if (xx < -sideh) xx += side;
                if (xx >  sideh) xx -= side;
                if (yy < -sideh) yy += side;
                if (yy >  sideh) yy -= side;
                if (zz < -sideh) zz += side;
                if (zz >  sideh) zz -= side;

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
            }
        }
        __syncthreads();
    }

    /* Write per-particle forces — no race (each thread owns its slot) */
    if (i < npart) {
        f[i * 3]     = fxi;
        f[i * 3 + 1] = fyi;
        f[i * 3 + 2] = fzi;
    }

    /* ---- block-level shared-memory reduction for epot and vir ---- */
    __shared__ double s_epot[TILE_SIZE];
    __shared__ double s_vir[TILE_SIZE];

    s_epot[threadIdx.x] = my_epot;
    s_vir[threadIdx.x]  = my_vir;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_epot[threadIdx.x] += s_epot[threadIdx.x + stride];
            s_vir[threadIdx.x]  += s_vir[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(d_epot, s_epot[0]);
        atomicAdd(d_vir,  s_vir[0]);
    }
}

/*
 * mkekin_kernel — scale forces, complete velocity update, and compute
 *                 kinetic energy via parallel map + reduction.
 *
 * Grid : ceil(n3 / BLOCK_SIZE)   Block : BLOCK_SIZE
 * Shared : BLOCK_SIZE doubles for the sum reduction.
 *
 * Map (per element, independent):  f[i] *= hsq2;  vh[i] += f[i];
 * Reduce:  sum += vh[i]²   →  block shared-mem tree reduction → atomicAdd
 */
__global__ void mkekin_kernel(int n3, double *f, double *vh,
                              double hsq2, double *d_sum)
{
    __shared__ double sdata[BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double local_val = 0.0;

    if (i < n3) {
        f[i]  *= hsq2;
        vh[i] += f[i];
        local_val = vh[i] * vh[i];
    }

    sdata[threadIdx.x] = local_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(d_sum, sdata[0]);
}

/*
 * velavg_kernel — compute total speed and count particles above the
 *                 thermal-velocity threshold (dual parallel reduction).
 *
 * Grid : ceil(npart / BLOCK_SIZE)   Block : BLOCK_SIZE
 * Shared : 2 × BLOCK_SIZE doubles (vel and count reductions).
 * Each thread handles one particle (3 consecutive vh components).
 */
__global__ void velavg_kernel(int npart, const double *vh, double vaverh,
                              double *d_vel, double *d_count)
{
    __shared__ double s_vel[BLOCK_SIZE];
    __shared__ double s_cnt[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double local_vel = 0.0;
    double local_cnt = 0.0;

    if (idx < npart) {
        int base = idx * 3;
        double vx = vh[base];
        double vy = vh[base + 1];
        double vz = vh[base + 2];
        double sq = sqrt(vx * vx + vy * vy + vz * vz);
        local_vel = sq;
        if (sq > vaverh)
            local_cnt = 1.0;
    }

    s_vel[threadIdx.x] = local_vel;
    s_cnt[threadIdx.x] = local_cnt;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_vel[threadIdx.x] += s_vel[threadIdx.x + stride];
            s_cnt[threadIdx.x] += s_cnt[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(d_vel,   s_vel[0]);
        atomicAdd(d_count, s_cnt[0]);
    }
}

/* ================================================================
 *  Host-side functions (initialisation & I/O — unchanged from
 *  the original; executed on the CPU)
 * ================================================================ */

void fcc(double x[], int npart, int mm, double a)
{
    int ijk = 0;
    for (int lg = 0; lg < 2; lg++)
        for (int i = 0; i < mm; i++)
            for (int j = 0; j < mm; j++)
                for (int k = 0; k < mm; k++) {
                    x[ijk]     = i * a + lg * a * 0.5;
                    x[ijk + 1] = j * a + lg * a * 0.5;
                    x[ijk + 2] = k * a;
                    ijk += 3;
                }
    for (int lg = 1; lg < 3; lg++)
        for (int i = 0; i < mm; i++)
            for (int j = 0; j < mm; j++)
                for (int k = 0; k < mm; k++) {
                    x[ijk]     = i * a + (2 - lg) * a * 0.5;
                    x[ijk + 1] = j * a + (lg - 1) * a * 0.5;
                    x[ijk + 2] = k * a + a * 0.5;
                    ijk += 3;
                }
}

extern "C" void   srand48(long);
extern "C" double drand48(void);

void mxwell(double vh[], int n3, double h, double tref)
{
    int npart = n3 / 3;
    double r, tscale, v1, v2, s, ekin = 0.0, sp = 0.0, sc;

    srand48(4711L);
    tscale = 16.0 / ((double)npart - 1.0);

    for (int i = 0; i < n3; i += 2) {
        s = 2.0;
        while (s >= 1.0) {
            v1 = 2.0 * drand48() - 1.0;
            v2 = 2.0 * drand48() - 1.0;
            s = v1 * v1 + v2 * v2;
        }
        r = sqrt(-2.0 * log(s) / s);
        vh[i]     = v1 * r;
        vh[i + 1] = v2 * r;
    }

    for (int i = 0; i < n3; i += 3) sp += vh[i];
    sp /= (double)npart;
    for (int i = 0; i < n3; i += 3) { vh[i] -= sp; ekin += vh[i] * vh[i]; }

    sp = 0.0;
    for (int i = 1; i < n3; i += 3) sp += vh[i];
    sp /= (double)npart;
    for (int i = 1; i < n3; i += 3) { vh[i] -= sp; ekin += vh[i] * vh[i]; }

    sp = 0.0;
    for (int i = 2; i < n3; i += 3) sp += vh[i];
    sp /= (double)npart;
    for (int i = 2; i < n3; i += 3) { vh[i] -= sp; ekin += vh[i] * vh[i]; }

    sc = h * sqrt(tref / (tscale * ekin));
    for (int i = 0; i < n3; i++) vh[i] *= sc;
}

void prnout(int move, double ekin, double epot, double tscale,
            double vir, double vel, double count, int npart, double den)
{
    double ek   = 24.0 * ekin;
    epot       *= 4.0;
    double etot = ek + epot;
    double temp = tscale * ekin;
    double pres = den * 16.0 * (ekin - vir) / (double)npart;
    vel        /= (double)npart;
    double rp   = (count / (double)npart) * 100.0;
    printf(" %6d%12.4f%12.4f%12.4f%10.4f%10.4f%10.4f%6.1f\n",
           move, ek, epot, etot, temp, pres, vel, rp);
}

double secnds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ================================================================
 *  Main program
 * ================================================================ */
int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <mm>\n", argv[0]);
        return 1;
    }

    int    mm    = atoi(argv[1]);
    int    npart = 4 * mm * mm * mm;
    int    n3    = 3 * npart;

    /* ---- simulation parameters ---- */
    double den    = 0.83134;
    double side   = pow((double)npart / den, 0.3333333);
    double tref   = 0.722;
    double rcoff  = (double)mm / 4.0;
    double h      = 0.064;
    int    irep   = 10;
    int    istop  = 20;
    int    iprint = 5;
    int    movemx = 20;

    double a      = side / (double)mm;
    double hsq    = h * h;
    double hsq2   = hsq * 0.5;
    double tscale = 16.0 / ((double)npart - 1.0);
    double vaver  = 1.13 * sqrt(tref / 24.0);
    double vaverh = vaver * h;
    double sideh  = 0.5 * side;
    double rcoffs = rcoff * rcoff;

    printf(" Molecular Dynamics Simulation example program (CUDA)\n");
    printf(" ----------------------------------------------------\n");
    printf(" number of particles is ............ %6d\n", npart);
    printf(" side length of the box is ......... %13.6f\n", side);
    printf(" cut off is ........................ %13.6f\n", rcoff);
    printf(" reduced temperature is ............ %13.6f\n", tref);
    printf(" basic timestep is ................. %13.6f\n", h);
    printf(" temperature scale interval ........ %6d\n", irep);
    printf(" stop scaling at move .............. %6d\n", istop);
    printf(" print interval .................... %6d\n", iprint);
    printf(" total no. of steps ................ %6d\n", movemx);

    /* ---- host memory allocation & initialisation ---- */
    double *h_x  = (double *)malloc(n3 * sizeof(double));
    double *h_vh = (double *)malloc(n3 * sizeof(double));
    double *h_f  = (double *)calloc(n3, sizeof(double));

    if (!h_x || !h_vh || !h_f) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    fcc(h_x, npart, mm, a);
    mxwell(h_vh, n3, h, tref);

    /* ---- CUDA device setup ---- */
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf(" CUDA device ...................... %s (SM %d.%d)\n\n",
           prop.name, prop.major, prop.minor);
    CUDA_CHECK(cudaSetDevice(dev));

    /* ---- device global-memory allocation ----
     *
     *  Particle arrays (resident on GPU for the entire simulation):
     *    d_x  — positions          (n3 doubles)
     *    d_vh — half-step velocity (n3 doubles)
     *    d_f  — forces             (n3 doubles)
     *
     *  Reduction scalars (zeroed before each kernel that uses them):
     *    d_epot, d_vir   — potential energy & virial   (forces kernel)
     *    d_sum           — kinetic-energy accumulator  (mkekin kernel)
     *    d_vel, d_count  — velocity & count accumulators (velavg kernel)
     */
    double *d_x, *d_vh, *d_f;
    CUDA_CHECK(cudaMalloc(&d_x,  n3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vh, n3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_f,  n3 * sizeof(double)));

    double *d_epot, *d_vir, *d_sum, *d_vel, *d_count;
    CUDA_CHECK(cudaMalloc(&d_epot,  sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vir,   sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum,   sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel,   sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(double)));

    /* ---- host → device transfer (one-time, at initialisation) ---- */
    CUDA_CHECK(cudaMemcpy(d_x,  h_x,  n3 * sizeof(double),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vh, h_vh, n3 * sizeof(double),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f,  h_f,  n3 * sizeof(double),
                           cudaMemcpyHostToDevice));

    /* Host arrays no longer needed — everything runs on the GPU */
    free(h_x);
    free(h_vh);
    free(h_f);

    /* ---- grid / block dimensions ---- */
    dim3 block_std(BLOCK_SIZE);
    dim3 grid_n3       ((n3    + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_npart    ((npart + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_forces  (TILE_SIZE);
    dim3 grid_forces   ((npart + TILE_SIZE  - 1) / TILE_SIZE);

    /* ---- simulation loop ---- */
    printf("    i       ke         pe            e         temp   "
           "   pres      vel      rp\n  -----  ----------  ----------"
           "  ----------  --------  --------  --------  ----\n");

    double start = secnds();

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

        /* 3. Scale forces, complete velocity update, kinetic energy */
        CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));
        mkekin_kernel<<<grid_n3, block_std>>>(n3, d_f, d_vh, hsq2, d_sum);

        double h_sum;
        CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(double),
                               cudaMemcpyDeviceToHost));
        double ekin = h_sum / hsq;

        /* 4. Average velocity magnitude & count above threshold */
        CUDA_CHECK(cudaMemset(d_vel,   0, sizeof(double)));
        CUDA_CHECK(cudaMemset(d_count, 0, sizeof(double)));
        velavg_kernel<<<grid_npart, block_std>>>(
            npart, d_vh, vaverh, d_vel, d_count);

        double h_vel, h_count;
        CUDA_CHECK(cudaMemcpy(&h_vel,   d_vel,   sizeof(double),
                               cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(double),
                               cudaMemcpyDeviceToHost));
        double vel   = h_vel / h;
        double count = h_count;

        /* 5. Temperature rescaling (every irep steps, up to istop) */
        if (move < istop && (move % irep) == 0) {
            double sc = sqrt(tref / (tscale * ekin));
            dscal_kernel<<<grid_n3, block_std>>>(n3, sc, d_vh);
            ekin = tref / tscale;
        }

        /* 6. Periodic diagnostics output — only here do we pull epot/vir
         *    back from the device (avoiding unnecessary D→H transfers on
         *    non-print steps). */
        if ((move % iprint) == 0) {
            double h_epot, h_vir;
            CUDA_CHECK(cudaMemcpy(&h_epot, d_epot, sizeof(double),
                                   cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_vir,  d_vir,  sizeof(double),
                                   cudaMemcpyDeviceToHost));
            h_epot *= 0.5;
            h_vir  *= 0.5;
            prnout(move, ekin, h_epot, tscale, h_vir, vel, count,
                   npart, den);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = secnds() - start;
    printf("Elapsed time = %f\n", (float)elapsed);

    /* ---- cleanup ---- */
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_vh));
    CUDA_CHECK(cudaFree(d_f));
    CUDA_CHECK(cudaFree(d_epot));
    CUDA_CHECK(cudaFree(d_vir));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_vel));
    CUDA_CHECK(cudaFree(d_count));

    return 0;
}
