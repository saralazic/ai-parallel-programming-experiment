#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double epot;
double vir;
double count;

static inline void checkCuda(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(call) checkCuda((call), #call)

/*
 * CPU helpers
 */
void fcc(double x[], int mm, double a) {
  int ijk = 0;
  int i, j, k, lg;

  for (lg = 0; lg < 2; lg++) {
    for (i = 0; i < mm; i++) {
      for (j = 0; j < mm; j++) {
        for (k = 0; k < mm; k++) {
          x[ijk] = i * a + lg * a * 0.5;
          x[ijk + 1] = j * a + lg * a * 0.5;
          x[ijk + 2] = k * a;
          ijk += 3;
        }
      }
    }
  }

  for (lg = 1; lg < 3; lg++) {
    for (i = 0; i < mm; i++) {
      for (j = 0; j < mm; j++) {
        for (k = 0; k < mm; k++) {
          x[ijk] = i * a + (2 - lg) * a * 0.5;
          x[ijk + 1] = j * a + (lg - 1) * a * 0.5;
          x[ijk + 2] = k * a + a * 0.5;
          ijk += 3;
        }
      }
    }
  }
}

void mxwell(double vh[], int n3, double h, double tref) {
  int i;
  int npart = n3 / 3;
  double r, tscale, v1, v2, s, ekin = 0.0, sp = 0.0, sc;

  srand48(4711);
  tscale = 16.0 / ((double)npart - 1.0);

  for (i = 0; i < n3; i += 2) {
    s = 2.0;
    while (s >= 1.0) {
      v1 = 2.0 * drand48() - 1.0;
      v2 = 2.0 * drand48() - 1.0;
      s = v1 * v1 + v2 * v2;
    }
    r = sqrt(-2.0 * log(s) / s);
    vh[i] = v1 * r;
    vh[i + 1] = v2 * r;
  }

  for (i = 0; i < n3; i += 3) sp += vh[i];
  sp /= (double)npart;
  for (i = 0; i < n3; i += 3) {
    vh[i] -= sp;
    ekin += vh[i] * vh[i];
  }

  sp = 0.0;
  for (i = 1; i < n3; i += 3) sp += vh[i];
  sp /= (double)npart;
  for (i = 1; i < n3; i += 3) {
    vh[i] -= sp;
    ekin += vh[i] * vh[i];
  }

  sp = 0.0;
  for (i = 2; i < n3; i += 3) sp += vh[i];
  sp /= (double)npart;
  for (i = 2; i < n3; i += 3) {
    vh[i] -= sp;
    ekin += vh[i] * vh[i];
  }

  sc = h * sqrt(tref / (tscale * ekin));
  for (i = 0; i < n3; i++) vh[i] *= sc;
}

void prnout(int move, double ekin, double epot_in, double tscale, double vir_in,
            double vel, double count_in, int npart, double den) {
  double ek, etot, temp, pres, rp;
  double ep = epot_in;
  double vr = vir_in;

  ek = 24.0 * ekin;
  ep *= 4.0;
  etot = ek + ep;
  temp = tscale * ekin;
  pres = den * 16.0 * (ekin - vr) / (double)npart;
  vel /= (double)npart;
  rp = (count_in / (double)npart) * 100.0;
  printf(" %6d%12.4f%12.4f%12.4f%10.4f%10.4f%10.4f%6.1f\n", move, ek, ep,
         etot, temp, pres, vel, rp);
}

double secnds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * CUDA kernels
 */
__global__ void domove_kernel(int n3, double *x, double *vh, double *f,
                              double side) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n3) return;

  x[i] += vh[i] + f[i];
  if (x[i] < 0.0) x[i] += side;
  if (x[i] > side) x[i] -= side;

  vh[i] += f[i];
  f[i] = 0.0;
}

__global__ void forces_kernel(int npart, const double *x, double *f, double side,
                              double rcoff, double *epot_i, double *vir_i) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= npart) return;

  extern __shared__ double sh[];
  double *shx = sh;
  double *shy = sh + blockDim.x;
  double *shz = sh + 2 * blockDim.x;

  double sideh = 0.5 * side;
  double rcoffs = rcoff * rcoff;

  int i3 = 3 * i;
  double xi = x[i3];
  double yi = x[i3 + 1];
  double zi = x[i3 + 2];

  double fxi = 0.0;
  double fyi = 0.0;
  double fzi = 0.0;
  double ep = 0.0;
  double vr = 0.0;

  for (int tile = 0; tile < npart; tile += blockDim.x) {
    int j = tile + threadIdx.x;
    if (j < npart) {
      int j3 = 3 * j;
      shx[threadIdx.x] = x[j3];
      shy[threadIdx.x] = x[j3 + 1];
      shz[threadIdx.x] = x[j3 + 2];
    }
    __syncthreads();

    int tileCount = min(blockDim.x, npart - tile);
    for (int k = 0; k < tileCount; k++) {
      int jidx = tile + k;
      if (jidx == i) continue;

      double xx = xi - shx[k];
      double yy = yi - shy[k];
      double zz = zi - shz[k];
      if (xx < -sideh) xx += side;
      if (xx > sideh) xx -= side;
      if (yy < -sideh) yy += side;
      if (yy > sideh) yy -= side;
      if (zz < -sideh) zz += side;
      if (zz > sideh) zz -= side;

      double rd = xx * xx + yy * yy + zz * zz;
      if (rd <= rcoffs) {
        double rrd = 1.0 / rd;
        double rrd2 = rrd * rrd;
        double rrd3 = rrd2 * rrd;
        double rrd4 = rrd2 * rrd2;
        double rrd6 = rrd2 * rrd4;
        double rrd7 = rrd6 * rrd;
        ep += (rrd6 - rrd3);
        double r148 = rrd7 - 0.5 * rrd4;
        vr -= rd * r148;
        fxi += xx * r148;
        fyi += yy * r148;
        fzi += zz * r148;
      }
    }
    __syncthreads();
  }

  f[i3] = fxi;
  f[i3 + 1] = fyi;
  f[i3 + 2] = fzi;
  epot_i[i] = ep;
  vir_i[i] = vr;
}

__global__ void reduce_sum_kernel(const double *in, int n, double *out) {
  extern __shared__ double ssum[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  int stride = blockDim.x * gridDim.x;

  double local = 0.0;
  for (int i = idx; i < n; i += stride) local += in[i];

  ssum[tid] = local;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) ssum[tid] += ssum[tid + s];
    __syncthreads();
  }

  if (tid == 0) atomicAdd(out, ssum[0]);
}

__global__ void mkekin_kernel(int n3, double *f, double *vh, double hsq2,
                              double *ekin_sum) {
  extern __shared__ double ssum[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  int stride = blockDim.x * gridDim.x;

  double local = 0.0;
  for (int i = idx; i < n3; i += stride) {
    f[i] *= hsq2;
    vh[i] += f[i];
    local += vh[i] * vh[i];
  }

  ssum[tid] = local;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) ssum[tid] += ssum[tid + s];
    __syncthreads();
  }

  if (tid == 0) atomicAdd(ekin_sum, ssum[0]);
}

__global__ void velavg_kernel(int npart, const double *vh, double vaverh,
                              double *vel_sum, double *count_sum) {
  extern __shared__ double smem[];
  double *svel = smem;
  double *scnt = smem + blockDim.x;

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  int stride = blockDim.x * gridDim.x;

  double lvel = 0.0;
  double lcnt = 0.0;

  for (int p = idx; p < npart; p += stride) {
    int i = 3 * p;
    double sq = sqrt(vh[i] * vh[i] + vh[i + 1] * vh[i + 1] +
                     vh[i + 2] * vh[i + 2]);
    if (sq > vaverh) lcnt += 1.0;
    lvel += sq;
  }

  svel[tid] = lvel;
  scnt[tid] = lcnt;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      svel[tid] += svel[tid + s];
      scnt[tid] += scnt[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(vel_sum, svel[0]);
    atomicAdd(count_sum, scnt[0]);
  }
}

__global__ void dscal_kernel(int n, double sa, double *sx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) sx[i] *= sa;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <mm>\n", argv[0]);
    return EXIT_FAILURE;
  }

  int mm = atoi(argv[1]);
  int npart = 4 * mm * mm * mm;
  int n3 = 3 * npart;

  double *x = (double *)malloc((size_t)n3 * sizeof(double));
  double *vh = (double *)malloc((size_t)n3 * sizeof(double));
  double *f = (double *)calloc((size_t)n3, sizeof(double));
  if (x == NULL || vh == NULL || f == NULL) {
    fprintf(stderr, "Host allocation failed.\n");
    return EXIT_FAILURE;
  }

  int move;
  double ekin, vel, sc;
  double start, elapsed;

  double den = 0.83134;
  double side = pow((double)npart / den, 0.3333333);
  double tref = 0.722;
  double rcoff = (double)mm / 4.0;
  double h = 0.064;
  int irep = 10;
  int istop = 20;
  int iprint = 5;
  int movemx = 20;

  double a = side / (double)mm;
  double hsq = h * h;
  double hsq2 = hsq * 0.5;
  double tscale = 16.0 / ((double)npart - 1.0);
  double vaver = 1.13 * sqrt(tref / 24.0);

  printf(" Molecular Dynamics Simulation CUDA program\n");
  printf(" ------------------------------------------\n");
  printf(" number of particles is ............ %6d\n", npart);
  printf(" side length of the box is ......... %13.6f\n", side);
  printf(" cut off is ........................ %13.6f\n", rcoff);
  printf(" reduced temperature is ............ %13.6f\n", tref);
  printf(" basic timestep is ................. %13.6f\n", h);
  printf(" temperature scale interval ........ %6d\n", irep);
  printf(" stop scaling at move .............. %6d\n", istop);
  printf(" print interval .................... %6d\n", iprint);
  printf(" total no. of steps ................ %6d\n", movemx);

  fcc(x, mm, a);
  mxwell(vh, n3, h, tref);

  double *d_x, *d_vh, *d_f;
  double *d_epot_i, *d_vir_i;
  double *d_epot, *d_vir, *d_ekin, *d_vel, *d_count;

  CUDA_CHECK(cudaMalloc((void **)&d_x, (size_t)n3 * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_vh, (size_t)n3 * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_f, (size_t)n3 * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_epot_i, (size_t)npart * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_vir_i, (size_t)npart * sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_epot, sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_vir, sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_ekin, sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_vel, sizeof(double)));
  CUDA_CHECK(cudaMalloc((void **)&d_count, sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_x, x, (size_t)n3 * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vh, vh, (size_t)n3 * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_f, f, (size_t)n3 * sizeof(double),
                        cudaMemcpyHostToDevice));

  const int threads = 256;
  const int blocksN3 = (n3 + threads - 1) / threads;
  const int blocksN = (npart + threads - 1) / threads;
  const size_t sharedForce = (size_t)3 * threads * sizeof(double);
  const size_t sharedReduce = (size_t)threads * sizeof(double);
  const size_t sharedVelavg = (size_t)2 * threads * sizeof(double);

  printf("\n    i       ke         pe            e         temp   "
         "   pres      vel      rp\n  -----  ----------  ----------"
         "  ----------  --------  --------  --------  ----\n");

  start = secnds();

  for (move = 1; move <= movemx; move++) {
    domove_kernel<<<blocksN3, threads>>>(n3, d_x, d_vh, d_f, side);
    CUDA_CHECK(cudaGetLastError());

    forces_kernel<<<blocksN, threads, sharedForce>>>(npart, d_x, d_f, side,
                                                      rcoff, d_epot_i, d_vir_i);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemset(d_epot, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_vir, 0, sizeof(double)));
    reduce_sum_kernel<<<blocksN, threads, sharedReduce>>>(d_epot_i, npart,
                                                           d_epot);
    reduce_sum_kernel<<<blocksN, threads, sharedReduce>>>(d_vir_i, npart,
                                                           d_vir);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemset(d_ekin, 0, sizeof(double)));
    mkekin_kernel<<<blocksN3, threads, sharedReduce>>>(n3, d_f, d_vh, hsq2,
                                                        d_ekin);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemset(d_vel, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(double)));
    velavg_kernel<<<blocksN, threads, sharedVelavg>>>(npart, d_vh, vaver * h,
                                                       d_vel, d_count);
    CUDA_CHECK(cudaGetLastError());

    double epot_sum, vir_sum, ekin_sum, vel_sum, count_sum;
    CUDA_CHECK(cudaMemcpy(&epot_sum, d_epot, sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(&vir_sum, d_vir, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(&ekin_sum, d_ekin, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(&vel_sum, d_vel, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&count_sum, d_count, sizeof(double),
                          cudaMemcpyDeviceToHost));

    epot = 0.5 * epot_sum;
    vir = 0.5 * vir_sum;
    ekin = ekin_sum / hsq;
    vel = vel_sum / h;
    count = count_sum;

    if (move < istop && fmod(move, irep) == 0) {
      sc = sqrt(tref / (tscale * ekin));
      dscal_kernel<<<blocksN3, threads>>>(n3, sc, d_vh);
      CUDA_CHECK(cudaGetLastError());
      ekin = tref / tscale;
    }

    if (fmod(move, iprint) == 0)
      prnout(move, ekin, epot, tscale, vir, vel, count, npart, den);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  elapsed = secnds() - start;
  printf("Elapsed time =  %f\n", (float)elapsed);

  CUDA_CHECK(cudaMemcpy(x, d_x, (size_t)n3 * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(vh, d_vh, (size_t)n3 * sizeof(double),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_vh));
  CUDA_CHECK(cudaFree(d_f));
  CUDA_CHECK(cudaFree(d_epot_i));
  CUDA_CHECK(cudaFree(d_vir_i));
  CUDA_CHECK(cudaFree(d_epot));
  CUDA_CHECK(cudaFree(d_vir));
  CUDA_CHECK(cudaFree(d_ekin));
  CUDA_CHECK(cudaFree(d_vel));
  CUDA_CHECK(cudaFree(d_count));

  free(x);
  free(vh);
  free(f);

  return 0;
}
