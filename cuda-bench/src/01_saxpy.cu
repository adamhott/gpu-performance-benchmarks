#include "common.cuh"
#include "csv.cuh"
#include "timer.cuh"
#include "verify.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

constexpr int kWarmupIters = 10;
constexpr int kTimedIters = 50;
constexpr int kThreadsPerBlock = 256;
constexpr float kSaxpyAlpha = 1.41421356f;
constexpr float kVerifyRelTol = 1e-4f;

__global__ __launch_bounds__(kThreadsPerBlock, 4) void saxpy_grid_stride(const float* x,
                                                                          float* y,
                                                                          int n,
                                                                          float a) {
  const int stride = blockDim.x * gridDim.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < n; i += stride) {
    y[i] = a * x[i] + y[i];
  }
}

inline int saxpy_num_blocks(int n, int multi_processor_count) {
  const int min_grid = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const int cap = std::max(1, multi_processor_count * 8);
  return std::max(1, std::min(min_grid, cap));
}

}  // namespace

int main() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  bench::query_device_props(device, &prop);

  const double peak_gflops = bench::peak_fp32_gflops(prop);
  const double peak_bw = bench::peak_memory_bandwidth_gb_s(prop);

  constexpr std::int64_t kSizes[] = {1LL << 20, 1LL << 22, 1LL << 24, 1LL << 26,
                                     1LL << 28};

  for (std::int64_t n : kSizes) {
    if (n > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
      std::fprintf(stderr, "Problem size exceeds int range\n");
      return EXIT_FAILURE;
    }
    const int ni = static_cast<int>(n);

    std::vector<float> hx(static_cast<std::size_t>(n));
    std::vector<float> hy(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
      const float v = static_cast<float>(i % 997) * 0.001f;
      hx[static_cast<std::size_t>(i)] = v;
      hy[static_cast<std::size_t>(i)] = v * 0.5f + 0.25f;
    }

    std::vector<float> href = hy;
    bench::saxpy_cpu_reference(hx, href, kSaxpyAlpha);

    float* dx = nullptr;
    float* dy = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&dy, bytes));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy.data(), bytes, cudaMemcpyHostToDevice));

    const int blocks = saxpy_num_blocks(ni, prop.multiProcessorCount);
    const dim3 grid_dim(static_cast<unsigned int>(blocks));
    const dim3 block_dim(static_cast<unsigned int>(kThreadsPerBlock));

    saxpy_grid_stride<<<grid_dim, block_dim>>>(dx, dy, ni, kSaxpyAlpha);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(static_cast<std::size_t>(n));
    CUDA_CHECK(cudaMemcpy(got.data(), dy, bytes, cudaMemcpyDeviceToHost));
    bench::require_relative_match(got, href, kVerifyRelTol, "saxpy");

    const bench::TimingStats stats = bench::time_cuda_kernel(
        [&]() {
          saxpy_grid_stride<<<grid_dim, block_dim>>>(dx, dy, ni, kSaxpyAlpha);
        },
        kWarmupIters, kTimedIters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dy, hy.data(), bytes, cudaMemcpyHostToDevice));

    const double mean_s = stats.mean_ms / 1000.0;
    const double flops = 2.0 * static_cast<double>(n);
    const double bytes_moved = 3.0 * static_cast<double>(bytes);  // read x, read y, write y
    const double gflops = (flops / mean_s) / 1e9;
    const double gb_s = (bytes_moved / mean_s) / 1e9;
    const double pct_flops = (gflops / peak_gflops) * 100.0;
    const double pct_bw = (gb_s / peak_bw) * 100.0;

    bench::append_csv_row("saxpy", prop, n, "grid_stride", stats.mean_ms, stats.stddev_ms,
                          gflops, gb_s, pct_flops, pct_bw);

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
  }

  return 0;
}
