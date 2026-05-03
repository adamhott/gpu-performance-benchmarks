#include "common.cuh"
#include "csv.cuh"
#include "timer.cuh"
#include "verify.cuh"

#include <curand_kernel.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kWarmupIters = 10;
constexpr int kTimedIters = 50;
constexpr int kThreads = 256;

__global__ __launch_bounds__(kThreads, 4) void mc_pi_kernel(long long samples,
                                                            unsigned long long* __restrict__ global_hits,
                                                            unsigned long long seed) {
  __shared__ unsigned int tbuf[kThreads];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nthr = gridDim.x * blockDim.x;
  curandStatePhilox4_32_10_t st;
  curand_init(seed, tid, 0ULL, &st);
  unsigned long long local = 0ULL;
  for (long long i = tid; i < samples; i += nthr) {
    const float x = curand_uniform(&st);
    const float y = curand_uniform(&st);
    if (x * x + y * y <= 1.0f) {
      ++local;
    }
  }
  tbuf[threadIdx.x] = static_cast<unsigned int>(
      local > static_cast<unsigned long long>(0xffffffffULL) ? 0xffffffffULL : local);
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      tbuf[threadIdx.x] += tbuf[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(global_hits, static_cast<unsigned long long>(tbuf[0]));
  }
}

inline int mc_num_blocks(const cudaDeviceProp& prop) {
  return std::min(8192, std::max(1, prop.multiProcessorCount * 16));
}

}  // namespace

int main() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  bench::query_device_props(device, &prop);
  const double peak_gflops = bench::peak_fp32_gflops(prop);
  const double peak_bw = bench::peak_memory_bandwidth_gb_s(prop);

  constexpr long long kSamples[] = {1000000LL, 10000000LL, 100000000LL, 500000000LL,
                                    1000000000LL};

  for (long long samples : kSamples) {
    unsigned long long* dhits = nullptr;
    CUDA_CHECK(cudaMalloc(&dhits, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(dhits, 0, sizeof(unsigned long long)));

    const int blocks = mc_num_blocks(prop);
    const dim3 grid(static_cast<unsigned int>(blocks));
    const dim3 block(static_cast<unsigned int>(kThreads));

    mc_pi_kernel<<<grid, block>>>(samples, dhits, 0xC0FFEEULL ^ static_cast<unsigned long long>(samples));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long hits = 0ULL;
    CUDA_CHECK(cudaMemcpy(&hits, dhits, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    const float pi_est = 4.0f * static_cast<float>(hits) / static_cast<float>(samples);
    bench::require_pi_mc(pi_est, samples, "montecarlo");

    const bench::TimingStats stats = bench::time_cuda_kernel(
        [&]() {
          CUDA_CHECK(cudaMemset(dhits, 0, sizeof(unsigned long long)));
          mc_pi_kernel<<<grid, block>>>(samples, dhits,
                                        0xC0FFEEULL ^ static_cast<unsigned long long>(samples));
        },
        kWarmupIters, kTimedIters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&hits, dhits, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    const float pi_timed = 4.0f * static_cast<float>(hits) / static_cast<float>(samples);
    bench::require_pi_mc(pi_timed, samples, "montecarlo_timed");

    const double mean_s = stats.mean_ms / 1000.0;
    const double flops = 8.0 * static_cast<double>(samples);
    const double bytes = 12.0 * static_cast<double>(samples) * static_cast<double>(sizeof(float));
    const double gflops = (flops / mean_s) / 1e9;
    const double gb_s = (bytes / mean_s) / 1e9;
    const double pct_flops = (gflops / peak_gflops) * 100.0;
    const double pct_bw = (gb_s / peak_bw) * 100.0;
    bench::append_csv_row("montecarlo", prop, samples, "curand_philox", stats.mean_ms,
                          stats.stddev_ms, gflops, gb_s, pct_flops, pct_bw);

    CUDA_CHECK(cudaFree(dhits));
  }

  return 0;
}
