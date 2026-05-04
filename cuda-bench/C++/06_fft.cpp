// Host C++ port of cuda-bench/src/06_fft.cu (cuFFT → radix-2 Cooley–Tukey on std::complex<float>).

#include "host_bench.hpp"
#include "host_fft.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

namespace {

constexpr int kWarmupIters = 2;
constexpr int kTimedIters = 5;
constexpr float kFftRelTol = 5e-3f;

inline bool complex_close(const std::complex<float>& a, const std::complex<float>& b,
                           float rel_tol) {
  const auto near = [&](float x, float y) {
    const float s = std::fmax(std::fabs(x), std::fabs(y));
    return std::fabs(x - y) <= rel_tol * std::fmax(1.0f, s);
  };
  return near(a.real(), b.real()) && near(a.imag(), b.imag());
}

inline void require_fft_roundtrip(const std::vector<std::complex<float>>& ref,
                                  const std::vector<std::complex<float>>& got, float rel_tol,
                                  const char* ctx) {
  if (ref.size() != got.size()) {
    std::fprintf(stderr, "%s: size mismatch\n", ctx);
    std::exit(EXIT_FAILURE);
  }
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (!complex_close(ref[i], got[i], rel_tol)) {
      std::fprintf(stderr, "%s: mismatch at %zu ref=(%.6g,%.6g) got=(%.6g,%.6g)\n", ctx, i,
                   static_cast<double>(ref[i].real()), static_cast<double>(ref[i].imag()),
                   static_cast<double>(got[i].real()), static_cast<double>(got[i].imag()));
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
  const double b = static_cast<double>(sizeof(std::complex<float>));
  return 4.0 * n * b;
}

}  // namespace

int main() {
  using host_bench::append_csv_row;
  using host_bench::fft_power2_inplace;
  using host_bench::HostCsvProps;
  using host_bench::kPlaceholderPeakGbS;
  using host_bench::kPlaceholderPeakGflops;
  using host_bench::time_host_kernel;

  const HostCsvProps prop{};
  constexpr int kSizes[] = {1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26};

  for (int n : kSizes) {
    std::vector<std::complex<float>> host(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      const float t = static_cast<float>(i % 997) * 0.001f - 0.05f;
      host[static_cast<std::size_t>(i)] = std::complex<float>{t, t * 0.5f + 0.1f};
    }
    const std::vector<std::complex<float>> href = host;

    std::vector<std::complex<float>> round = host;
    fft_power2_inplace(round, false);
    fft_power2_inplace(round, true);
    require_fft_roundtrip(href, round, kFftRelTol, "fft_roundtrip");

    const std::vector<std::complex<float>> seed = host;
    std::vector<std::complex<float>> work(static_cast<std::size_t>(n));
    const auto stats = time_host_kernel(
        [&]() {
          std::copy(seed.begin(), seed.end(), work.begin());
          fft_power2_inplace(work, false);
        },
        kWarmupIters, kTimedIters);

    const double mean_s = stats.mean_ms / 1000.0;
    const double flops = fft_flops_estimate(static_cast<double>(n));
    const double bytes_moved = fft_bytes_estimate(static_cast<double>(n));
    const double gflops = (flops / mean_s) / 1e9;
    const double gb_s = (bytes_moved / mean_s) / 1e9;
    const double pct_flops = (gflops / kPlaceholderPeakGflops) * 100.0;
    const double pct_bw = (gb_s / kPlaceholderPeakGbS) * 100.0;
    append_csv_row("fft_1d_c2c", prop, static_cast<std::int64_t>(n), "cpu_fft_forward",
                   stats.mean_ms, stats.stddev_ms, gflops, gb_s, pct_flops, pct_bw);
  }

  return 0;
}
