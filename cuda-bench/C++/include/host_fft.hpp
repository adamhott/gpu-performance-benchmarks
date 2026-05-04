#pragma once

#include <cmath>
#include <complex>
#include <vector>

namespace host_bench {

// In-place radix-2 Cooley–Tukey (DIT) for n a power of two. Internals use double precision
// so forward+inverse round-trips match float inputs within typical CUDA cuFFT tolerances.
inline void fft_power2_inplace(std::vector<std::complex<float>>& af, bool inverse) {
  const int n = static_cast<int>(af.size());
  if (n <= 1) {
    return;
  }
  std::vector<std::complex<double>> a(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    a[static_cast<std::size_t>(i)] =
        std::complex<double>(static_cast<double>(af[static_cast<std::size_t>(i)].real()),
                             static_cast<double>(af[static_cast<std::size_t>(i)].imag()));
  }

  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      std::swap(a[static_cast<std::size_t>(i)], a[static_cast<std::size_t>(j)]);
    }
  }

  constexpr double kPi = 3.14159265358979323846;
  for (int len = 2; len <= n; len <<= 1) {
    const double ang = 2.0 * kPi / static_cast<double>(len) * (inverse ? -1.0 : 1.0);
    const std::complex<double> wlen(std::cos(ang), std::sin(ang));
    for (int i = 0; i < n; i += len) {
      std::complex<double> w(1.0, 0.0);
      for (int j = 0; j < len / 2; ++j) {
        const std::complex<double> u = a[static_cast<std::size_t>(i + j)];
        const std::complex<double> v = a[static_cast<std::size_t>(i + j + len / 2)] * w;
        a[static_cast<std::size_t>(i + j)] = u + v;
        a[static_cast<std::size_t>(i + j + len / 2)] = u - v;
        w *= wlen;
      }
    }
  }
  if (inverse) {
    const double inv_n = 1.0 / static_cast<double>(n);
    for (int i = 0; i < n; ++i) {
      a[static_cast<std::size_t>(i)] *= inv_n;
    }
  }
  for (int i = 0; i < n; ++i) {
    af[static_cast<std::size_t>(i)] =
        std::complex<float>(static_cast<float>(a[static_cast<std::size_t>(i)].real()),
                            static_cast<float>(a[static_cast<std::size_t>(i)].imag()));
  }
}

}  // namespace host_bench
