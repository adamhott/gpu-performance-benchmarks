#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace bench {

inline bool nearly_equal_relative(float a, float b, float rel_tol) {
  const float scale = std::fmax(std::fabs(a), std::fabs(b));
  const float abs_tol = rel_tol * std::fmax(1.0f, scale);
  return std::fabs(a - b) <= abs_tol;
}

inline void require_relative_match(const std::vector<float>& got,
                                   const std::vector<float>& ref, float rel_tol,
                                   const char* context) {
  if (got.size() != ref.size()) {
    std::fprintf(stderr, "%s: size mismatch got=%zu ref=%zu\n", context,
                 got.size(), ref.size());
    std::exit(EXIT_FAILURE);
  }
  std::size_t first_bad = got.size();
  float max_rel_err = 0.0f;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (!nearly_equal_relative(got[i], ref[i], rel_tol)) {
      first_bad = i;
      break;
    }
    const float denom = std::fmax(1.0f, std::fmax(std::fabs(got[i]), std::fabs(ref[i])));
    const float rel = std::fabs(got[i] - ref[i]) / denom;
    max_rel_err = std::fmax(max_rel_err, rel);
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

// Monte Carlo Pi: absolute floor plus sample-count scaling (statistical MC error ~ 1/sqrt(N)).
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

}  // namespace bench
