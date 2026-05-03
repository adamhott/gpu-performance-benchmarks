#include "common.cuh"
#include "csv.cuh"
#include "timer.cuh"

#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kWarmupIters = 10;
constexpr int kTimedIters = 50;
constexpr int kThreadsPerBlock = 256;
constexpr float kFftRelTol = 5e-3f;

#define CUFFT_CHECK(expr)                                                        \
  do {                                                                           \
    cufftResult _cufft_status = (expr);                                          \
    if (_cufft_status != CUFFT_SUCCESS) {                                       \
      std::fprintf(stderr, "%s:%d cuFFT error %d\n", __FILE__, __LINE__,         \
                   static_cast<int>(_cufft_status));                            \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                            \
  } while (0)

__global__ __launch_bounds__(kThreadsPerBlock, 4) void scale_fft_inplace(cufftComplex* a,
                                                                         int n,
                                                                         float inv_n) {
  const int stride = blockDim.x * gridDim.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < n; i += stride) {
    a[i].x *= inv_n;
    a[i].y *= inv_n;
  }
}

inline int fft_scale_num_blocks(int n, int multi_processor_count) {
  const int min_grid = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const int cap = std::max(1, multi_processor_count * 8);
  return std::max(1, std::min(min_grid, cap));
}

inline bool complex_close(const cufftComplex& a, const cufftComplex& b, float rel_tol) {
  const auto near = [&](float x, float y) {
    const float s = std::fmax(std::fabs(x), std::fabs(y));
    return std::fabs(x - y) <= rel_tol * std::fmax(1.0f, s);
  };
  return near(a.x, b.x) && near(a.y, b.y);
}

inline void require_fft_roundtrip(const std::vector<cufftComplex>& ref,
                                  const std::vector<cufftComplex>& got, float rel_tol,
                                  const char* ctx) {
  if (ref.size() != got.size()) {
    std::fprintf(stderr, "%s: size mismatch\n", ctx);
    std::exit(EXIT_FAILURE);
  }
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (!complex_close(ref[i], got[i], rel_tol)) {
      std::fprintf(stderr, "%s: mismatch at %zu ref=(%.6g,%.6g) got=(%.6g,%.6g)\n", ctx, i,
                   static_cast<double>(ref[i].x), static_cast<double>(ref[i].y),
                   static_cast<double>(got[i].x), static_cast<double>(got[i].y));
      std::exit(EXIT_FAILURE);
    }
  }
}

inline double fft_flops_estimate(double n) {
  if (n <= 1.0) {
    return 0.0;
  }
  const double lg = std::log2(n);
  return 5.0 * n * lg;
}

inline double fft_bytes_estimate(double n) {
  const double b = static_cast<double>(sizeof(cufftComplex));
  return 4.0 * n * b;
}

// Device-to-device reset before each forward so iterations do not compose FFTs;
// D2D is outside CUDA events so elapsed time is the forward transform only.
inline bench::TimingStats time_fft_forward_only(cufftHandle plan, cufftComplex* d,
                                                 cufftComplex* d_seed, std::size_t bytes,
                                                 int warmup_iters, int timed_iters) {
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < warmup_iters; ++i) {
    CUDA_CHECK(cudaMemcpy(d, d_seed, bytes, cudaMemcpyDeviceToDevice));
    CUFFT_CHECK(cufftExecC2C(plan, d, d, CUFFT_FORWARD));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(timed_iters));
  for (int i = 0; i < timed_iters; ++i) {
    CUDA_CHECK(cudaMemcpy(d, d_seed, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    CUFFT_CHECK(cufftExecC2C(plan, d, d, CUFFT_FORWARD));
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
    const double dv = v - mean;
    var_acc += dv * dv;
  }
  const double denom = (n > 1.0) ? (n - 1.0) : 1.0;
  const double stddev = std::sqrt(var_acc / denom);
  return bench::TimingStats{mean, stddev};
}

}  // namespace

int main() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  bench::query_device_props(device, &prop);
  const double peak_gflops = bench::peak_fp32_gflops(prop);
  const double peak_bw = bench::peak_memory_bandwidth_gb_s(prop);

  constexpr int kSizes[] = {1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26};

  for (int n : kSizes) {
    std::vector<cufftComplex> host(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      const float t = static_cast<float>(i % 997) * 0.001f - 0.05f;
      host[static_cast<std::size_t>(i)] = cufftComplex{t, t * 0.5f + 0.1f};
    }
    const std::vector<cufftComplex> href = host;

    cufftHandle plan{};
    CUFFT_CHECK(cufftCreate(&plan));
    CUFFT_CHECK(cufftPlan1d(&plan, n, CUFFT_C2C, 1));

    cufftComplex* d = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(cufftComplex);
    CUDA_CHECK(cudaMalloc(&d, bytes));
    CUDA_CHECK(cudaMemcpy(d, host.data(), bytes, cudaMemcpyHostToDevice));

    CUFFT_CHECK(cufftExecC2C(plan, d, d, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(plan, d, d, CUFFT_INVERSE));

    const int blocks = fft_scale_num_blocks(n, prop.multiProcessorCount);
    const dim3 grid_dim(static_cast<unsigned int>(blocks));
    const dim3 block_dim(static_cast<unsigned int>(kThreadsPerBlock));
    const float inv_n = 1.0f / static_cast<float>(n);
    scale_fft_inplace<<<grid_dim, block_dim>>>(d, n, inv_n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<cufftComplex> round(static_cast<std::size_t>(n));
    CUDA_CHECK(cudaMemcpy(round.data(), d, bytes, cudaMemcpyDeviceToHost));
    require_fft_roundtrip(href, round, kFftRelTol, "fft_roundtrip");

    cufftComplex* d_seed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_seed, bytes));
    CUDA_CHECK(cudaMemcpy(d_seed, host.data(), bytes, cudaMemcpyHostToDevice));

    const bench::TimingStats stats =
        time_fft_forward_only(plan, d, d_seed, bytes, kWarmupIters, kTimedIters);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_seed));

    const double mean_s = stats.mean_ms / 1000.0;
    const double flops = fft_flops_estimate(static_cast<double>(n));
    const double bytes_moved = fft_bytes_estimate(static_cast<double>(n));
    const double gflops = (flops / mean_s) / 1e9;
    const double gb_s = (bytes_moved / mean_s) / 1e9;
    const double pct_flops = (gflops / peak_gflops) * 100.0;
    const double pct_bw = (gb_s / peak_bw) * 100.0;
    bench::append_csv_row("fft_1d_c2c", prop, static_cast<std::int64_t>(n), "cufft_forward",
                          stats.mean_ms, stats.stddev_ms, gflops, gb_s, pct_flops, pct_bw);

    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(d));
  }

  return 0;
}
