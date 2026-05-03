#include "common.cuh"
#include "csv.cuh"
#include "timer.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

constexpr int kWarmupIters = 10;
constexpr int kTimedIters = 50;
constexpr int kThreads = 256;

__device__ float warp_reduce_sum32(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__global__ __launch_bounds__(kThreads, 4) void reduce_segments(const float* __restrict__ x,
                                                               float* __restrict__ partials,
                                                               long long n, int num_blocks) {
  __shared__ float warp_cache[8];
  const int bid = blockIdx.x;
  const long long chunk_start = (static_cast<long long>(bid) * n) / num_blocks;
  const long long chunk_end = (static_cast<long long>(bid + 1) * n) / num_blocks;
  float s = 0.f;
  for (long long idx = chunk_start + threadIdx.x; idx < chunk_end; idx += blockDim.x) {
    s += x[idx];
  }
  s = warp_reduce_sum32(s);
  if ((threadIdx.x & 31) == 0) {
    warp_cache[threadIdx.x >> 5] = s;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      t += warp_cache[w];
    }
    partials[bid] = t;
  }
}

__global__ __launch_bounds__(kThreads, 4) void reduce_final(const float* __restrict__ part,
                                                            float* __restrict__ out, int B) {
  __shared__ float warp_cache[8];
  float s = 0.f;
  for (int i = threadIdx.x; i < B; i += blockDim.x) {
    s += part[i];
  }
  s = warp_reduce_sum32(s);
  if ((threadIdx.x & 31) == 0) {
    warp_cache[threadIdx.x >> 5] = s;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      t += warp_cache[w];
    }
    *out = t;
  }
}

inline int reduction_num_blocks(const cudaDeviceProp& prop) {
  const int cap = prop.multiProcessorCount * 8;
  if (cap < 1) {
    return 1;
  }
  if (cap > 1024) {
    return 1024;
  }
  return cap;
}

inline void require_sum_close(double cpu_sum, float gpu_sum, long long n, const char* ctx) {
  const double g = static_cast<double>(gpu_sum);
  const double diff = std::fabs(cpu_sum - g);
  const double scale = std::fmax(1.0, std::fabs(cpu_sum));
  // Parallel float reduction order differs from sequential double accumulation; allow
  // a small relative band plus a mild absolute term that grows slowly with n.
  const double tol = 1e-2 * scale + 32.0 * std::sqrt(static_cast<double>(n > 0 ? n : 1));
  if (diff > tol) {
    std::fprintf(stderr,
                 "%s: reduction mismatch cpu=%.8g gpu=%.8g n=%lld diff=%.8g tol=%.8g\n", ctx,
                 cpu_sum, g, static_cast<long long>(n), diff, tol);
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace

int main() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  bench::query_device_props(device, &prop);
  const double peak_gflops = bench::peak_fp32_gflops(prop);
  const double peak_bw = bench::peak_memory_bandwidth_gb_s(prop);

  constexpr std::int64_t kSizes[] = {1LL << 20, 1LL << 22, 1LL << 24, 1LL << 26, 1LL << 28};

  for (std::int64_t n : kSizes) {
    if (n > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
      std::fprintf(stderr, "Problem size exceeds int range\n");
      return EXIT_FAILURE;
    }
    std::vector<float> hx(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
      hx[static_cast<std::size_t>(i)] =
          static_cast<float>((i * 11) % 10007) * 1e-4f - 0.05f;
    }
    double cpu_sum = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
      cpu_sum += static_cast<double>(hx[static_cast<std::size_t>(i)]);
    }

    float* dx = nullptr;
    float* dpart = nullptr;
    float* dout = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));

    const int B = reduction_num_blocks(prop);
    CUDA_CHECK(cudaMalloc(&dpart, static_cast<std::size_t>(B) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dout, sizeof(float)));

    const dim3 seg_grid(static_cast<unsigned int>(B));
    const dim3 seg_block(static_cast<unsigned int>(kThreads));
    const dim3 fin_grid(1);
    const dim3 fin_block(static_cast<unsigned int>(kThreads));

    reduce_segments<<<seg_grid, seg_block>>>(dx, dpart, n, B);
    CUDA_CHECK(cudaGetLastError());
    reduce_final<<<fin_grid, fin_block>>>(dpart, dout, B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_sum = 0.f;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, dout, sizeof(float), cudaMemcpyDeviceToHost));
    require_sum_close(cpu_sum, gpu_sum, n, "reduction");

    const bench::TimingStats stats = bench::time_cuda_kernel(
        [&]() {
          reduce_segments<<<seg_grid, seg_block>>>(dx, dpart, n, B);
          reduce_final<<<fin_grid, fin_block>>>(dpart, dout, B);
        },
        kWarmupIters, kTimedIters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const double mean_s = stats.mean_ms / 1000.0;
    const double flops = static_cast<double>(n > 0 ? n - 1 : 0);
    const double bytes_moved = static_cast<double>(bytes);
    const double gflops = (flops / mean_s) / 1e9;
    const double gb_s = (bytes_moved / mean_s) / 1e9;
    const double pct_flops = (gflops / peak_gflops) * 100.0;
    const double pct_bw = (gb_s / peak_bw) * 100.0;
    bench::append_csv_row("reduction", prop, n, "warp_shuffle", stats.mean_ms, stats.stddev_ms,
                          gflops, gb_s, pct_flops, pct_bw);

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dpart));
    CUDA_CHECK(cudaFree(dout));
  }

  return 0;
}
