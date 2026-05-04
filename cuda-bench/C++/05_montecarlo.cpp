// Host C++ port of cuda-bench/src/05_montecarlo.cu (Philox on device → std::mt19937_64 on host).

#include "host_bench.hpp"

#include <cstdint>
#include <cstdio>
#include <random>

namespace {

constexpr int kWarmupIters = 2;
constexpr int kTimedIters = 5;

inline unsigned long long mc_count_hits(long long samples, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  unsigned long long hits = 0ULL;
  for (long long i = 0; i < samples; ++i) {
    const float x = dist(rng);
    const float y = dist(rng);
    if (x * x + y * y <= 1.0f) {
      ++hits;
    }
  }
  return hits;
}

}  // namespace

int main() {
  using host_bench::append_csv_row;
  using host_bench::HostCsvProps;
  using host_bench::kPlaceholderPeakGbS;
  using host_bench::kPlaceholderPeakGflops;
  using host_bench::require_pi_mc;
  using host_bench::time_host_kernel;

  const HostCsvProps prop{};
  constexpr long long kSamples[] = {1000000LL, 10000000LL, 100000000LL, 500000000LL,
                                    1000000000LL};

  for (long long samples : kSamples) {
    const std::uint64_t seed =
        0xC0FFEEULL ^ static_cast<std::uint64_t>(static_cast<std::uint64_t>(samples));
    const unsigned long long h0 = mc_count_hits(samples, seed);
    const float pi_est = 4.0f * static_cast<float>(h0) / static_cast<float>(samples);
    require_pi_mc(pi_est, samples, "montecarlo");

    unsigned long long hits = 0ULL;
    const auto stats = time_host_kernel(
        [&]() { hits = mc_count_hits(samples, seed); }, kWarmupIters, kTimedIters);

    const float pi_timed = 4.0f * static_cast<float>(hits) / static_cast<float>(samples);
    require_pi_mc(pi_timed, samples, "montecarlo_timed");

    const double mean_s = stats.mean_ms / 1000.0;
    const double flops = 8.0 * static_cast<double>(samples);
    const double bytes = 12.0 * static_cast<double>(samples) * static_cast<double>(sizeof(float));
    const double gflops = (flops / mean_s) / 1e9;
    const double gb_s = (bytes / mean_s) / 1e9;
    const double pct_flops = (gflops / kPlaceholderPeakGflops) * 100.0;
    const double pct_bw = (gb_s / kPlaceholderPeakGbS) * 100.0;
    append_csv_row("montecarlo", prop, samples, "mt19937_uniform", stats.mean_ms, stats.stddev_ms,
                    gflops, gb_s, pct_flops, pct_bw);
  }

  return 0;
}
