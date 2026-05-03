#include "common.cuh"
#include "csv.cuh"
#include "timer.cuh"
#include "verify.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int kWarmupIters = 10;
constexpr int kTimedIters = 50;
constexpr int kThreads = 256;
constexpr int S_TILE = 16;
constexpr float kStencilRelTol = 1e-4f;

__global__ __launch_bounds__(kThreads, 4) void laplacian_naive_global(const float* __restrict__ u,
                                                                       float* __restrict__ out, int N) {
  const int count = N * N;
  const int stride = blockDim.x * gridDim.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (; idx < count; idx += stride) {
    const int i = idx / N;
    const int j = idx % N;
    const float c = u[idx];
    const float um1 = (i > 0) ? u[(i - 1) * N + j] : 0.f;
    const float up1 = (i + 1 < N) ? u[(i + 1) * N + j] : 0.f;
    const float vm1 = (j > 0) ? u[i * N + (j - 1)] : 0.f;
    const float vp1 = (j + 1 < N) ? u[i * N + (j + 1)] : 0.f;
    out[idx] = um1 + up1 + vm1 + vp1 - 4.f * c;
  }
}

__global__ __launch_bounds__(kThreads, 4) void laplacian_shared_tiled(const float* __restrict__ u,
                                                                     float* __restrict__ out, int N) {
  __shared__ float smem[S_TILE + 2][S_TILE + 2];
  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  const int span = S_TILE + 2;
  for (int k = threadIdx.x; k < span * span; k += blockDim.x) {
    const int ly = k / span;
    const int lx = k % span;
    const int gi = tile_y * S_TILE + ly - 1;
    const int gj = tile_x * S_TILE + lx - 1;
    float v = 0.f;
    if (gi >= 0 && gi < N && gj >= 0 && gj < N) {
      v = u[gi * N + gj];
    }
    smem[ly][lx] = v;
  }
  __syncthreads();

  const int tx = threadIdx.x % S_TILE;
  const int ty = threadIdx.x / S_TILE;
  const int go_i = tile_y * S_TILE + ty;
  const int go_j = tile_x * S_TILE + tx;
  if (go_i < N && go_j < N) {
    const float c = smem[ty + 1][tx + 1];
    const float um1 = smem[ty][tx + 1];
    const float up1 = smem[ty + 2][tx + 1];
    const float vm1 = smem[ty + 1][tx];
    const float vp1 = smem[ty + 1][tx + 2];
    out[go_i * N + go_j] = um1 + up1 + vm1 + vp1 - 4.f * c;
  }
}

inline int stencil_num_blocks(int count, int multi_processor_count) {
  const int min_grid = (count + kThreads - 1) / kThreads;
  const int cap = std::max(1, multi_processor_count * 8);
  return std::max(1, std::min(min_grid, cap));
}

inline void laplacian_cpu(const float* u, float* out, int N) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      const float c = u[i * N + j];
      const float um1 = (i > 0) ? u[(i - 1) * N + j] : 0.f;
      const float up1 = (i + 1 < N) ? u[(i + 1) * N + j] : 0.f;
      const float vm1 = (j > 0) ? u[i * N + (j - 1)] : 0.f;
      const float vp1 = (j + 1 < N) ? u[i * N + (j + 1)] : 0.f;
      out[i * N + j] = um1 + up1 + vm1 + vp1 - 4.f * c;
    }
  }
}

inline void append_stencil_row(const char* variant, const bench::TimingStats& stats, int N,
                               const cudaDeviceProp& prop, double peak_gflops, double peak_bw) {
  const std::int64_t problem_size = static_cast<std::int64_t>(N) * static_cast<std::int64_t>(N);
  const double mean_s = stats.mean_ms / 1000.0;
  const double pts = static_cast<double>(problem_size);
  const double flops = 10.0 * pts;
  const double bytes = 6.0 * pts * static_cast<double>(sizeof(float));
  const double gflops = (flops / mean_s) / 1e9;
  const double gb_s = (bytes / mean_s) / 1e9;
  const double pct_flops = (gflops / peak_gflops) * 100.0;
  const double pct_bw = (gb_s / peak_bw) * 100.0;
  bench::append_csv_row("stencil", prop, problem_size, variant, stats.mean_ms, stats.stddev_ms,
                        gflops, gb_s, pct_flops, pct_bw);
}

}  // namespace

int main() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  bench::query_device_props(device, &prop);
  const double peak_gflops = bench::peak_fp32_gflops(prop);
  const double peak_bw = bench::peak_memory_bandwidth_gb_s(prop);

  constexpr int kNs[] = {1024, 2048, 4096, 6144, 8192};

  for (int N : kNs) {
    const std::size_t elems = static_cast<std::size_t>(N) * static_cast<std::size_t>(N);
    std::vector<float> hu(elems);
    for (std::size_t i = 0; i < elems; ++i) {
      hu[i] = static_cast<float>((i * 17) % 503) * 0.001f - 0.25f;
    }
    std::vector<float> href(elems);
    laplacian_cpu(hu.data(), href.data(), N);

    float* du = nullptr;
    float* dout = nullptr;
    const std::size_t bytes = elems * sizeof(float);
    CUDA_CHECK(cudaMalloc(&du, bytes));
    CUDA_CHECK(cudaMalloc(&dout, bytes));
    CUDA_CHECK(cudaMemcpy(du, hu.data(), bytes, cudaMemcpyHostToDevice));

    const int count = N * N;
    const int blocks_naive = stencil_num_blocks(count, prop.multiProcessorCount);
    const dim3 naive_grid(static_cast<unsigned int>(blocks_naive));
    const dim3 naive_block(static_cast<unsigned int>(kThreads));

    laplacian_naive_global<<<naive_grid, naive_block>>>(du, dout, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got_naive(elems);
    CUDA_CHECK(cudaMemcpy(got_naive.data(), dout, bytes, cudaMemcpyDeviceToHost));
    bench::require_relative_match(got_naive, href, kStencilRelTol, "stencil_naive_global");

    const bench::TimingStats naive_stats = bench::time_cuda_kernel(
        [&]() { laplacian_naive_global<<<naive_grid, naive_block>>>(du, dout, N); },
        kWarmupIters, kTimedIters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    append_stencil_row("naive_global", naive_stats, N, prop, peak_gflops, peak_bw);

    const int tiles_x = (N + S_TILE - 1) / S_TILE;
    const int tiles_y = (N + S_TILE - 1) / S_TILE;
    const dim3 shared_grid(static_cast<unsigned int>(tiles_x), static_cast<unsigned int>(tiles_y));
    const dim3 shared_block(static_cast<unsigned int>(kThreads));

    laplacian_shared_tiled<<<shared_grid, shared_block>>>(du, dout, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got_shared(elems);
    CUDA_CHECK(cudaMemcpy(got_shared.data(), dout, bytes, cudaMemcpyDeviceToHost));
    bench::require_relative_match(got_shared, href, kStencilRelTol, "stencil_shared_tiled");

    const bench::TimingStats shared_stats = bench::time_cuda_kernel(
        [&]() { laplacian_shared_tiled<<<shared_grid, shared_block>>>(du, dout, N); },
        kWarmupIters, kTimedIters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    append_stencil_row("shared_tiled", shared_stats, N, prop, peak_gflops, peak_bw);

    CUDA_CHECK(cudaFree(du));
    CUDA_CHECK(cudaFree(dout));
  }

  return 0;
}
