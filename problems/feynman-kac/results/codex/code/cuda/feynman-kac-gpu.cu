#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err__));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static void timestamp();
__host__ __device__ static inline double potential(double a, double x);
__device__ static inline double r8_uniform_01_device(unsigned int *seed);
__device__ static inline double atomic_add_double(double *address, double val);

__global__ void simulate_paths_kernel(const double *x_values, double *sum_w,
                                      unsigned long long *sum_steps, int n,
                                      double a, double h, double rth) {
  const int i = blockIdx.y;
  const int tid = threadIdx.x;
  const int trajectory = blockIdx.x * blockDim.x + tid;

  extern __shared__ unsigned char shared_raw[];
  double *shared_w = reinterpret_cast<double *>(shared_raw);
  unsigned long long *shared_steps =
      reinterpret_cast<unsigned long long *>(shared_w + blockDim.x);

  double local_w = 0.0;
  unsigned long long local_steps = 0ULL;

  const double x = x_values[i];
  const double test = a * a - x * x;

  if (trajectory < n && test >= 0.0) {
    double x1 = x;
    double w = 1.0;
    double chk = 0.0;

    // Thread-local RNG state (local memory/registers).
    unsigned int seed = 123456789u ^
                        (static_cast<unsigned int>(i + 1) * 747796405u) ^
                        (static_cast<unsigned int>(trajectory + 1) *
                         2891336453u);
    if (seed == 0u) {
      seed = 1u;
    }

    while (chk < 1.0) {
      const double us = r8_uniform_01_device(&seed) - 0.5;
      const double dx = (us < 0.0) ? -rth : rth;

      const double vs = potential(a, x1);
      x1 += dx;
      local_steps += 1ULL;

      const double vh = potential(a, x1);
      const double we = (1.0 - h * vs) * w;
      w = w - 0.5 * h * (vh * we + vs * w);

      chk = (x1 / a) * (x1 / a);
    }
    local_w = w;
  }

  shared_w[tid] = local_w;
  shared_steps[tid] = local_steps;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_w[tid] += shared_w[tid + stride];
      shared_steps[tid] += shared_steps[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0 && test >= 0.0) {
    atomic_add_double(&sum_w[i], shared_w[0]);
    atomicAdd(&sum_steps[i], shared_steps[0]);
  }
}

int main(int argc, char **argv) {
  double a = 2.0;
  double h = 0.0001;
  int n = 10000;
  int ni = 21;
  const int threads_per_block = 256;

  if (argc >= 2) {
    n = atoi(argv[1]);
    if (n <= 0) {
      fprintf(stderr, "Invalid number of trajectories.\n");
      return EXIT_FAILURE;
    }
  }

  timestamp();

  printf("\n");
  printf("FEYNMAN_KAC_1D_GPU:\n");
  printf("  CUDA C++ version.\n");
  printf("\n");
  printf("  Program parameters:\n");
  printf("\n");
  printf("    A = %g\n", a);
  printf("    H = %g\n", h);
  printf("    N trajectories per point = %d\n", n);
  printf("    X coordinate discretized by %d points\n", ni + 2);
  printf("\n");

  const int num_points = ni + 2;
  const double rth = sqrt(h);

  double *h_x = static_cast<double *>(malloc(num_points * sizeof(double)));
  double *h_sum_w = static_cast<double *>(malloc(num_points * sizeof(double)));
  unsigned long long *h_sum_steps =
      static_cast<unsigned long long *>(malloc(num_points * sizeof(unsigned long long)));

  if (h_x == nullptr || h_sum_w == nullptr || h_sum_steps == nullptr) {
    fprintf(stderr, "Host memory allocation failed.\n");
    free(h_x);
    free(h_sum_w);
    free(h_sum_steps);
    return EXIT_FAILURE;
  }

  for (int i = 0; i <= ni + 1; ++i) {
    h_x[i] = (static_cast<double>(ni - i) * (-a) +
              static_cast<double>(i - 1) * a) /
             static_cast<double>(ni - 1);
  }

  // Device global memory.
  double *d_x = nullptr;
  double *d_sum_w = nullptr;
  unsigned long long *d_sum_steps = nullptr;

  CUDA_CHECK(cudaMalloc(&d_x, num_points * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_sum_w, num_points * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_sum_steps, num_points * sizeof(unsigned long long)));

  CUDA_CHECK(cudaMemcpy(d_x, h_x, num_points * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_sum_w, 0, num_points * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_sum_steps, 0, num_points * sizeof(unsigned long long)));

  const int blocks_x = (n + threads_per_block - 1) / threads_per_block;
  const dim3 grid(blocks_x, num_points, 1);
  const dim3 block(threads_per_block, 1, 1);
  const size_t shared_bytes =
      threads_per_block * (sizeof(double) + sizeof(unsigned long long));

  simulate_paths_kernel<<<grid, block, shared_bytes>>>(d_x, d_sum_w, d_sum_steps, n,
                                                        a, h, rth);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_sum_w, d_sum_w, num_points * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_sum_steps, d_sum_steps,
                        num_points * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_sum_w));
  CUDA_CHECK(cudaFree(d_sum_steps));

  double err = 0.0;
  int n_int = 0;
  int k = 0;

  printf("     I     K       X           W exact      W Approx        Error");
  printf("      Ave Steps  Test\n");
  printf("\n");

  for (int i = 0; i <= ni + 1; ++i) {
    const double x = h_x[i];
    const double test = a * a - x * x;
    const double w_exact = (test < 0.0) ? 1.0 : exp((x / a) * (x / a) - 1.0);
    double wt = 1.0;
    int steps_ave = 0;

    ++k;
    if (test >= 0.0) {
      n_int += 1;
      wt = h_sum_w[i] / static_cast<double>(n);
      steps_ave = static_cast<int>(h_sum_steps[i] / static_cast<unsigned long long>(n));
      err += (w_exact - wt) * (w_exact - wt);
    }

    printf("  %4d  %4d  %12g  %12g  %12g  %12g  %8d  %8g\n", i, k, x, w_exact,
           wt, fabs(w_exact - wt), steps_ave, test);
  }

  if (n_int > 0) {
    err = sqrt(err / static_cast<double>(n_int));
  }

  printf("\n");
  printf("  RMS absolute error in solution = %g\n", err);
  printf("\n");
  printf("FEYNMAN_KAC_1D_GPU:\n");
  printf("  Normal end of execution.\n");
  printf("\n");

  free(h_x);
  free(h_sum_w);
  free(h_sum_steps);

  timestamp();

  return 0;
}

__host__ __device__ static inline double potential(double a, double x) {
  return 2.0 * ((x / a / a) * (x / a / a)) + 1.0 / a / a;
}

__device__ static inline double r8_uniform_01_device(unsigned int *seed) {
  const unsigned int q = 127773u;
  const unsigned int a = 16807u;
  const unsigned int r = 2836u;
  const unsigned int m = 2147483647u;

  const unsigned int k = (*seed) / q;
  int next = static_cast<int>(a * ((*seed) - k * q)) - static_cast<int>(k * r);
  if (next < 0) {
    next += static_cast<int>(m);
  }

  *seed = static_cast<unsigned int>(next);
  return static_cast<double>(*seed) * 4.656612875e-10;
}

__device__ static inline double atomic_add_double(double *address, double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  unsigned long long int *address_as_ull =
      reinterpret_cast<unsigned long long int *>(address);
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif
}

static void timestamp(void) {
#define TIME_SIZE 40
  static char time_buffer[TIME_SIZE];
  const struct tm *tm_ptr;
  time_t now;

  now = time(nullptr);
  tm_ptr = localtime(&now);
  strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr);

  printf("%s\n", time_buffer);
#undef TIME_SIZE
}
