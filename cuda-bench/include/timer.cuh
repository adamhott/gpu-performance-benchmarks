#pragma once

#include "common.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace bench {

struct TimingStats {
  double mean_ms = 0.0;
  double stddev_ms = 0.0;
};

// Warm-up then timed runs using CUDA events. Returns mean and sample stddev (N-1).
template <typename F>
inline TimingStats time_cuda_kernel(F&& f, int warmup_iters, int timed_iters) {
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < warmup_iters; ++i) {
    f();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(timed_iters));

  for (int i = 0; i < timed_iters; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    f();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    samples.push_back(static_cast<double>(ms));
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  double sum = 0.0;
  for (double v : samples) {
    sum += v;
  }
  const double n = static_cast<double>(samples.size());
  const double mean = sum / n;

  double var_acc = 0.0;
  for (double v : samples) {
    const double d = v - mean;
    var_acc += d * d;
  }
  const double denom = (n > 1.0) ? (n - 1.0) : 1.0;
  const double stddev = std::sqrt(var_acc / denom);

  return TimingStats{mean, stddev};
}

}  // namespace bench
