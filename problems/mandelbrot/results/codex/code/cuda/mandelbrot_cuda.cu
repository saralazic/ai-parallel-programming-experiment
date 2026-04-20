#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

namespace {

inline void cuda_check(cudaError_t code, const char *context) {
  if (code != cudaSuccess) {
    std::fprintf(stderr, "CUDA error at %s: %s\n", context, cudaGetErrorString(code));
    std::exit(EXIT_FAILURE);
  }
}

void timestamp() {
  static constexpr int kTimeSize = 40;
  char time_buffer[kTimeSize];
  const std::time_t now = std::time(nullptr);
  const std::tm *tm_info = std::localtime(&now);
  std::strftime(time_buffer, kTimeSize, "%d %B %Y %I:%M:%S %p", tm_info);
  std::printf("%s\n", time_buffer);
}

void write_ppm_binary(const char *filename, const uchar3 *rgb, int width, int height) {
  FILE *output = std::fopen(filename, "wb");
  if (!output) {
    std::perror("fopen");
    std::exit(EXIT_FAILURE);
  }

  std::fprintf(output, "P6\n%d %d\n255\n", width, height);
  std::fwrite(rgb, sizeof(uchar3), static_cast<size_t>(width) * static_cast<size_t>(height), output);
  std::fclose(output);
}

__device__ __forceinline__ uchar3 make_color(int iter, int iter_max) {
  if ((iter & 1) == 1) {
    return make_uchar3(255, 255, 255);
  }

  const float normalized = static_cast<float>(iter) / static_cast<float>(iter_max);
  const int c = static_cast<int>(255.0f * sqrtf(sqrtf(sqrtf(normalized))));
  return make_uchar3(static_cast<unsigned char>(3 * c / 5),
                     static_cast<unsigned char>(3 * c / 5),
                     static_cast<unsigned char>(c));
}

__global__ void mandelbrot_kernel(uchar3 *image,
                                  int width,
                                  int height,
                                  int iter_max,
                                  double x_min,
                                  double x_max,
                                  double y_min,
                                  double y_max) {
  // Shared LUT caches the color mapping for all possible iteration counts.
  extern __shared__ uchar3 color_lut[];

  const int local_tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int local_threads = blockDim.x * blockDim.y;
  for (int idx = local_tid; idx <= iter_max; idx += local_threads) {
    color_lut[idx] = make_color(idx, iter_max);
  }
  __syncthreads();

  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= height || j >= width) {
    return;
  }

  const double y = (static_cast<double>(i - 1) * y_max + static_cast<double>(height - i) * y_min) /
                   static_cast<double>(height - 1);
  const double x = (static_cast<double>(j - 1) * x_max + static_cast<double>(width - j) * x_min) /
                   static_cast<double>(width - 1);

  // Local scalar state stays in registers (or spills to local memory if needed).
  double x1 = x;
  double y1 = y;
  int count = 0;

  for (int k = 1; k <= iter_max; ++k) {
    const double x2 = x1 * x1 - y1 * y1 + x;
    const double y2 = 2.0 * x1 * y1 + y;
    if (x2 < -2.0 || x2 > 2.0 || y2 < -2.0 || y2 > 2.0) {
      count = k;
      break;
    }
    x1 = x2;
    y1 = y2;
  }

  image[i * width + j] = color_lut[count];
}

}  // namespace

int main() {
  constexpr int width = 500;
  constexpr int height = 500;
  constexpr int iter_max = 2000;
  constexpr double x_min = -2.25;
  constexpr double x_max = 1.25;
  constexpr double y_min = -1.75;
  constexpr double y_max = 1.75;
  const char *output_filename = "mandelbrot_cuda.ppm";

  timestamp();
  std::printf("\nMANDELBROT_CUDA\n");

  const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
  const size_t image_bytes = pixel_count * sizeof(uchar3);

  uchar3 *h_image = static_cast<uchar3 *>(std::malloc(image_bytes));
  if (!h_image) {
    std::fprintf(stderr, "Host memory allocation failed.\n");
    return EXIT_FAILURE;
  }

  uchar3 *d_image = nullptr;
  cuda_check(cudaMalloc(&d_image, image_bytes), "cudaMalloc(d_image)");

  const dim3 block_dim(16, 16);
  const dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y);
  const size_t shared_bytes = static_cast<size_t>(iter_max + 1) * sizeof(uchar3);

  cudaEvent_t kernel_start, kernel_stop;
  cuda_check(cudaEventCreate(&kernel_start), "cudaEventCreate(kernel_start)");
  cuda_check(cudaEventCreate(&kernel_stop), "cudaEventCreate(kernel_stop)");

  cuda_check(cudaEventRecord(kernel_start), "cudaEventRecord(kernel_start)");
  mandelbrot_kernel<<<grid_dim, block_dim, shared_bytes>>>(
      d_image, width, height, iter_max, x_min, x_max, y_min, y_max);
  cuda_check(cudaGetLastError(), "mandelbrot_kernel launch");
  cuda_check(cudaEventRecord(kernel_stop), "cudaEventRecord(kernel_stop)");
  cuda_check(cudaEventSynchronize(kernel_stop), "cudaEventSynchronize(kernel_stop)");

  float kernel_ms = 0.0f;
  cuda_check(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop), "cudaEventElapsedTime");

  cuda_check(cudaMemcpy(h_image, d_image, image_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

  std::printf("  Kernel time = %.3f ms\n", kernel_ms);

  write_ppm_binary(output_filename, h_image, width, height);
  std::printf("  Graphics data written to \"%s\".\n", output_filename);

  cuda_check(cudaEventDestroy(kernel_start), "cudaEventDestroy(kernel_start)");
  cuda_check(cudaEventDestroy(kernel_stop), "cudaEventDestroy(kernel_stop)");
  cuda_check(cudaFree(d_image), "cudaFree(d_image)");
  std::free(h_image);

  std::printf("\nMANDELBROT_CUDA\n");
  std::printf("  Normal end of execution.\n\n");
  timestamp();
  return EXIT_SUCCESS;
}
