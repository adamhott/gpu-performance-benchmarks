// Host C++ port of cuda-bench/src/01_saxpy.cu (grid-stride SAXPY → plain loop).

#include "host_bench.hpp"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

// Fewer repeats than the CUDA bench so host runs finish in reasonable wall time.
constexpr int kWarmupIters = 2;
constexpr int kTimedIters = 5;
constexpr float kSaxpyAlpha = 1.41421356f;
constexpr float kVerifyRelTol = 1e-4f;

inline void saxpy_cpu(float* y, const float* x, int n, float a) {
  for (int i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
}

}  // namespace

int main() {
  using host_bench::append_csv_row;
  using host_bench::HostCsvProps;
  using host_bench::kPlaceholderPeakGbS;
  using host_bench::kPlaceholderPeakGflops;
  using host_bench::require_relative_match;
  using host_bench::saxpy_cpu_reference;
  using host_bench::time_host_kernel;

  const HostCsvProps prop{};
  // Same progression as 01_saxpy.cu except the largest size (2^28) omitted for CPU bandwidth.
  constexpr std::int64_t kSizes[] = {1LL << 20, 1LL << 22, 1LL << 24, 1LL << 26};

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
    saxpy_cpu_reference(hx, href, kSaxpyAlpha);

    std::vector<float> got = hy;
    saxpy_cpu(got.data(), hx.data(), ni, kSaxpyAlpha);
    require_relative_match(got, href, kVerifyRelTol, "saxpy");

    std::vector<float> work = hy;
    const auto stats = time_host_kernel(
        [&]() { saxpy_cpu(work.data(), hx.data(), ni, kSaxpyAlpha); }, kWarmupIters, kTimedIters);

    const double mean_s = stats.mean_ms / 1000.0;
    const double flops = 2.0 * static_cast<double>(n);
    const double bytes = 3.0 * static_cast<double>(n) * static_cast<double>(sizeof(float));
    const double gflops = (flops / mean_s) / 1e9;
    const double gb_s = (bytes / mean_s) / 1e9;
    const double pct_flops = (gflops / kPlaceholderPeakGflops) * 100.0;
    const double pct_bw = (gb_s / kPlaceholderPeakGbS) * 100.0;

    append_csv_row("saxpy", prop, n, "cpu_loop", stats.mean_ms, stats.stddev_ms, gflops, gb_s,
                    pct_flops, pct_bw);
  }

  return 0;
}
