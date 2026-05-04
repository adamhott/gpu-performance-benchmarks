// Host C++ port of cuda-bench/src/03_stencil.cu (Laplacian: linear index vs tile order).

#include "host_bench.hpp"

#include <cstdio>
#include <vector>

namespace {

constexpr int kWarmupIters = 2;
constexpr int kTimedIters = 5;
constexpr int S_TILE = 16;
constexpr float kStencilRelTol = 1e-4f;

inline void laplacian_naive_global(const float* u, float* out, int N) {
  const int count = N * N;
  for (int idx = 0; idx < count; ++idx) {
    const int i = idx / N;
    const int j = idx % N;
    const float c = u[idx];
    const float um1 = (i > 0) ? u[(i - 1) * N + j] : 0.f;
    const float up1 = (i + 1 < N) ? u[(i + 1) * N + j] : 0.f;
    const float vm1 = (j > 0) ? u[i * N + (j - 1)] : 0.f;
    const float vp1 = (j + 1 < N) ? u[i * N + (j + 1)] : 0.f;
    out[idx] = um1 + up1 + vm1 + vp1 - 4.f * c;
  }
}

inline void laplacian_shared_tiled_order(const float* u, float* out, int N) {
  const int tiles_x = (N + S_TILE - 1) / S_TILE;
  const int tiles_y = (N + S_TILE - 1) / S_TILE;
  for (int tile_y = 0; tile_y < tiles_y; ++tile_y) {
    for (int tile_x = 0; tile_x < tiles_x; ++tile_x) {
      for (int ty = 0; ty < S_TILE; ++ty) {
        for (int tx = 0; tx < S_TILE; ++tx) {
          const int go_i = tile_y * S_TILE + ty;
          const int go_j = tile_x * S_TILE + tx;
          if (go_i < N && go_j < N) {
            const int idx = go_i * N + go_j;
            const float c = u[idx];
            const float um1 = (go_i > 0) ? u[(go_i - 1) * N + go_j] : 0.f;
            const float up1 = (go_i + 1 < N) ? u[(go_i + 1) * N + go_j] : 0.f;
            const float vm1 = (go_j > 0) ? u[go_i * N + (go_j - 1)] : 0.f;
            const float vp1 = (go_j + 1 < N) ? u[go_i * N + (go_j + 1)] : 0.f;
            out[idx] = um1 + up1 + vm1 + vp1 - 4.f * c;
          }
        }
      }
    }
  }
}

inline void laplacian_cpu(const float* u, float* out, int N) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      const float c = u[i * N + j];
      const float um1 = (i > 0) ? u[(i - 1) * N + j] : 0.f;
      const float up1 = (i + 1 < N) ? u[(i + 1) * N + j] : 0.f;
      const float vm1 = (j > 0) ? u[i * N + (j - 1)] : 0.f;
      const float vp1 = (j + 1 < N) ? u[i * N + (j + 1)] : 0.f;
      out[i * N + j] = um1 + up1 + vm1 + vp1 - 4.f * c;
    }
  }
}

inline void append_stencil_row(const char* variant, const host_bench::TimingStats& stats, int N,
                               const host_bench::HostCsvProps& prop) {
  const std::int64_t problem_size = static_cast<std::int64_t>(N) * static_cast<std::int64_t>(N);
  const double mean_s = stats.mean_ms / 1000.0;
  const double pts = static_cast<double>(problem_size);
  const double flops = 10.0 * pts;
  const double bytes = 6.0 * pts * static_cast<double>(sizeof(float));
  const double gflops = (flops / mean_s) / 1e9;
  const double gb_s = (bytes / mean_s) / 1e9;
  const double pct_flops = (gflops / host_bench::kPlaceholderPeakGflops) * 100.0;
  const double pct_bw = (gb_s / host_bench::kPlaceholderPeakGbS) * 100.0;
  host_bench::append_csv_row("stencil", prop, problem_size, variant, stats.mean_ms,
                             stats.stddev_ms, gflops, gb_s, pct_flops, pct_bw);
}

}  // namespace

int main() {
  using host_bench::HostCsvProps;
  using host_bench::require_relative_match;
  using host_bench::time_host_kernel;

  const HostCsvProps prop{};
  constexpr int kNs[] = {1024, 2048, 4096, 6144, 8192};

  for (int N : kNs) {
    const std::size_t elems = static_cast<std::size_t>(N) * static_cast<std::size_t>(N);
    std::vector<float> hu(elems);
    for (std::size_t i = 0; i < elems; ++i) {
      hu[i] = static_cast<float>((i * 17) % 503) * 0.001f - 0.25f;
    }
    std::vector<float> href(elems);
    laplacian_cpu(hu.data(), href.data(), N);

    std::vector<float> got_naive(elems);
    laplacian_naive_global(hu.data(), got_naive.data(), N);
    require_relative_match(got_naive, href, kStencilRelTol, "stencil_naive_global");

    std::vector<float> work_naive(elems);
    const auto naive_stats = time_host_kernel(
        [&]() { laplacian_naive_global(hu.data(), work_naive.data(), N); }, kWarmupIters,
        kTimedIters);
    append_stencil_row("naive_global", naive_stats, N, prop);

    std::vector<float> got_tiled(elems);
    laplacian_shared_tiled_order(hu.data(), got_tiled.data(), N);
    require_relative_match(got_tiled, href, kStencilRelTol, "stencil_shared_tiled");

    std::vector<float> work_tiled(elems);
    const auto shared_stats = time_host_kernel(
        [&]() { laplacian_shared_tiled_order(hu.data(), work_tiled.data(), N); }, kWarmupIters,
        kTimedIters);
    append_stencil_row("shared_tiled", shared_stats, N, prop);
  }

  return 0;
}
