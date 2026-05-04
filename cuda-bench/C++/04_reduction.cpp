// Host C++ port of cuda-bench/src/04_reduction.cu (GPU warp shuffle → double accumulation).

#include "host_bench.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

constexpr int kWarmupIters = 2;
constexpr int kTimedIters = 5;

inline void require_sum_close(double ref_sum, double got_sum, long long n, const char* ctx) {
  const double diff = std::fabs(ref_sum - got_sum);
  const double scale = std::fmax(1.0, std::fabs(ref_sum));
  const double tol = 1e-10 * scale + 1e-6 * std::sqrt(static_cast<double>(n > 0 ? n : 1));
  if (diff > tol) {
    std::fprintf(stderr,
                 "%s: reduction mismatch ref=%.12g got=%.12g n=%lld diff=%.12g tol=%.12g\n", ctx,
                 ref_sum, got_sum, static_cast<long long>(n), diff, tol);
    std::exit(EXIT_FAILURE);
  }
}

inline double reduce_sum_double(const std::vector<float>& x) {
  double s = 0.0;
  for (float v : x) {
    s += static_cast<double>(v);
  }
  return s;
}

}  // namespace

int main() {
  using host_bench::append_csv_row;
  using host_bench::HostCsvProps;
  using host_bench::kPlaceholderPeakGbS;
  using host_bench::kPlaceholderPeakGflops;
  using host_bench::time_host_kernel;

  const HostCsvProps prop{};
  constexpr std::int64_t kSizes[] = {1LL << 20, 1LL << 22, 1LL << 24, 1LL << 26};

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
    double ref_sum = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
      ref_sum += static_cast<double>(hx[static_cast<std::size_t>(i)]);
    }

    const double got = reduce_sum_double(hx);
    require_sum_close(ref_sum, got, n, "reduction");

    const auto stats = time_host_kernel([&]() { (void)reduce_sum_double(hx); }, kWarmupIters,
                                        kTimedIters);

    const double mean_s = stats.mean_ms / 1000.0;
    const double flops = static_cast<double>(n > 0 ? n - 1 : 0);
    const double bytes_moved =
        static_cast<double>(static_cast<std::size_t>(n) * sizeof(float));
    const double gflops = (flops / mean_s) / 1e9;
    const double gb_s = (bytes_moved / mean_s) / 1e9;
    const double pct_flops = (gflops / kPlaceholderPeakGflops) * 100.0;
    const double pct_bw = (gb_s / kPlaceholderPeakGbS) * 100.0;
    append_csv_row("reduction", prop, n, "cpu_double_sum", stats.mean_ms, stats.stddev_ms,
                    gflops, gb_s, pct_flops, pct_bw);
  }

  return 0;
}
