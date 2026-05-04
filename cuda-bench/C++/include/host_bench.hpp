#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace host_bench {

struct TimingStats {
  double mean_ms = 0.0;
  double stddev_ms = 0.0;
};

template <typename F>
inline TimingStats time_host_kernel(F&& f, int warmup_iters, int timed_iters) {
  for (int i = 0; i < warmup_iters; ++i) {
    f();
  }
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(timed_iters));
  for (int i = 0; i < timed_iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    samples.push_back(ms);
  }
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

// Placeholder roofline for CSV columns (not hardware-probed).
constexpr double kPlaceholderPeakGflops = 512.0;
constexpr double kPlaceholderPeakGbS = 80.0;

struct HostCsvProps {
  const char* device_name = "CPU";
  int major = 0;
  int minor = 0;
};

inline void append_csv_row(const std::string& benchmark, const HostCsvProps& prop,
                           std::int64_t problem_size, const std::string& variant,
                           double mean_ms, double stddev_ms, double gflops, double gb_s,
                           double pct_peak_flops, double pct_peak_bw) {
  namespace fs = std::filesystem;
  constexpr const char* kPath = "results/results_cpp.csv";
  std::error_code ec;
  fs::create_directories("results", ec);
  (void)ec;
  const bool need_header = !fs::exists(kPath) || fs::file_size(kPath) == 0;
  std::ofstream out(kPath, std::ios::app);
  if (!out) {
    std::fprintf(stderr, "Failed to open %s for append\n", kPath);
    std::exit(EXIT_FAILURE);
  }
  if (need_header) {
    out << "benchmark,gpu_name,compute_capability,problem_size,variant,mean_ms,stddev_ms,"
           "gflops,gb_s,pct_peak_flops,pct_peak_bw\n";
  }
  const std::string cc = std::to_string(prop.major) + "." + std::to_string(prop.minor);
  out << benchmark << ',' << prop.device_name << ',' << cc << ',' << problem_size << ','
      << variant << ',' << mean_ms << ',' << stddev_ms << ',' << gflops << ',' << gb_s
      << ',' << pct_peak_flops << ',' << pct_peak_bw << '\n';
}

inline bool nearly_equal_relative(float a, float b, float rel_tol) {
  const float scale = std::fmax(std::fabs(a), std::fabs(b));
  const float abs_tol = rel_tol * std::fmax(1.0f, scale);
  return std::fabs(a - b) <= abs_tol;
}

inline void require_relative_match(const std::vector<float>& got,
                                   const std::vector<float>& ref, float rel_tol,
                                   const char* context) {
  if (got.size() != ref.size()) {
    std::fprintf(stderr, "%s: size mismatch got=%zu ref=%zu\n", context, got.size(),
                 ref.size());
    std::exit(EXIT_FAILURE);
  }
  std::size_t first_bad = got.size();
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (!nearly_equal_relative(got[i], ref[i], rel_tol)) {
      first_bad = i;
      break;
    }
  }
  if (first_bad != got.size()) {
    std::fprintf(stderr,
                 "%s: verification failed at index %zu (got=%.8g ref=%.8g rel_tol=%.8g)\n",
                 context, first_bad, static_cast<double>(got[first_bad]),
                 static_cast<double>(ref[first_bad]), static_cast<double>(rel_tol));
    std::exit(EXIT_FAILURE);
  }
}

inline void saxpy_cpu_reference(const std::vector<float>& x, std::vector<float>& y,
                                float a) {
  for (std::size_t i = 0; i < y.size(); ++i) {
    y[i] = a * x[i] + y[i];
  }
}

inline void require_pi_mc(float pi_est, long long samples, const char* context) {
  constexpr float kPi = 3.14159265f;
  const float abs_floor = 1e-3f;
  const float stat =
      8.0f / std::sqrt(static_cast<float>(samples > 0 ? samples : 1));
  const float tol = std::fmax(abs_floor, stat);
  if (std::fabs(pi_est - kPi) > tol) {
    std::fprintf(stderr,
                 "%s: Pi estimate out of tolerance (est=%.8g pi=%.8g samples=%lld tol=%.8g)\n",
                 context, static_cast<double>(pi_est), static_cast<double>(kPi),
                 static_cast<long long>(samples), static_cast<double>(tol));
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace host_bench
