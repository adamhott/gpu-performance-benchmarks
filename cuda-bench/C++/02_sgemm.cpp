// Host C++ port of cuda-bench/src/02_sgemm.cu (naive tiled GEMM → triple-loop SGEMM).
// Matrix sizes 2048/4096 are omitted here: O(N^3) reference and timed naive are impractical
// on a CPU without an external BLAS library.

#include "host_bench.hpp"

#include <cstdio>
#include <vector>

namespace {

constexpr int kWarmupIters = 2;
constexpr int kTimedIters = 5;
constexpr float kAlpha = 1.0f;
constexpr float kBeta = 0.0f;
constexpr float kGemmRelTol = 5e-3f;

inline void sgemm_cpu_row_major(const float* A, const float* B, float* C, int N) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float s = 0.f;
      for (int kk = 0; kk < N; ++kk) {
        s += A[i * N + kk] * B[kk * N + j];
      }
      C[i * N + j] = kAlpha * s + kBeta * C[i * N + j];
    }
  }
}

}  // namespace

int main() {
  using host_bench::append_csv_row;
  using host_bench::HostCsvProps;
  using host_bench::kPlaceholderPeakGbS;
  using host_bench::kPlaceholderPeakGflops;
  using host_bench::require_relative_match;
  using host_bench::time_host_kernel;

  const HostCsvProps prop{};
  constexpr int kNs[] = {256, 512, 1024};

  for (int N : kNs) {
    const std::size_t elems = static_cast<std::size_t>(N) * static_cast<std::size_t>(N);
    std::vector<float> hA(elems);
    std::vector<float> hB(elems);
    std::vector<float> hC(elems, 0.f);
    for (std::size_t i = 0; i < elems; ++i) {
      hA[i] = static_cast<float>((i * 13) % 1000) * 0.0001f - 0.05f;
      hB[i] = static_cast<float>((i * 7) % 1001) * 0.0001f - 0.05f;
    }
    std::vector<float> gold(elems, 0.f);
    sgemm_cpu_row_major(hA.data(), hB.data(), gold.data(), N);

    std::vector<float> got(elems, 0.f);
    sgemm_cpu_row_major(hA.data(), hB.data(), got.data(), N);
    require_relative_match(got, gold, kGemmRelTol, "sgemm_cpu");

    std::vector<float> work(elems, 0.f);
    const auto stats = time_host_kernel(
        [&]() { sgemm_cpu_row_major(hA.data(), hB.data(), work.data(), N); }, kWarmupIters,
        kTimedIters);

    require_relative_match(work, gold, kGemmRelTol, "sgemm_cpu_timed");

    const double mean_s = stats.mean_ms / 1000.0;
    const double n3 = static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
    const double flops = 2.0 * n3;
    const double bytes =
        (2.0 * n3 + static_cast<double>(N) * static_cast<double>(N)) * static_cast<double>(sizeof(float));
    const double gflops = (flops / mean_s) / 1e9;
    const double gb_s = (bytes / mean_s) / 1e9;
    const double pct_flops = (gflops / kPlaceholderPeakGflops) * 100.0;
    const double pct_bw = (gb_s / kPlaceholderPeakGbS) * 100.0;
    append_csv_row("sgemm", prop, N, "cpu_naive", stats.mean_ms, stats.stddev_ms, gflops, gb_s,
                    pct_flops, pct_bw);
  }

  return 0;
}
