// Host C++ port of cuda-bench/src/07_nbody_tree.cu (GPU BH traversal → same stack walk on CPU).

#include "host_bench.hpp"
#include "nbody_bh_host.hpp"

#include <cstdio>
#include <vector>

namespace {

constexpr int kWarmupIters = 2;
constexpr int kTimedIters = 5;
constexpr float kThetaVerify = 0.12f;
constexpr float kThetaTimed = 0.55f;
constexpr float kSoftFrac = 1e-4f;
constexpr float kAccelRelTol = 0.12f;

}  // namespace

int main() {
  using host_bench::append_csv_row;
  using host_bench::HostCsvProps;
  using host_bench::kPlaceholderPeakGbS;
  using host_bench::kPlaceholderPeakGflops;
  using host_bench::time_host_kernel;

  using nbody_host::bh_accel_all;
  using nbody_host::bh_bytes_estimate;
  using nbody_host::bh_flops_estimate;
  using nbody_host::BarnesHutCpu;
  using nbody_host::Body;
  using nbody_host::direct_accel_cpu;
  using nbody_host::require_accel_match;
  using nbody_host::Vec3;

  const HostCsvProps prop{};
  constexpr int kSizes[] = {2048, 8192, 32768, 131072, 262144};

  for (int n : kSizes) {
    std::vector<Body> host_bodies(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      const unsigned u = static_cast<unsigned>(i) * 1103515245u + 12345u;
      const float fx = static_cast<float>(u & 0xffffu) / 65535.0f;
      const float fy = static_cast<float>((u >> 16) & 0xffffu) / 65535.0f;
      const float fz = static_cast<float>((u >> 8) & 0xffffu) / 65535.0f;
      host_bodies[static_cast<std::size_t>(i)].x = 0.2f + 0.6f * fx;
      host_bodies[static_cast<std::size_t>(i)].y = 0.2f + 0.6f * fy;
      host_bodies[static_cast<std::size_t>(i)].z = 0.2f + 0.6f * fz;
      host_bodies[static_cast<std::size_t>(i)].m =
          0.5f + 0.5f * static_cast<float>((u >> 4) & 0xffu) / 255.0f;
    }

    const float width = 0.6f * std::sqrt(3.0f);
    const float eps2 = (kSoftFrac * width) * (kSoftFrac * width);

    BarnesHutCpu builder(host_bodies);
    std::vector<nbody_host::BHNode> host_tree = builder.build();
    if (host_tree.empty()) {
      std::fprintf(stderr, "nbody_tree: empty tree\n");
      return EXIT_FAILURE;
    }
    const int root = 0;

    if (n <= 4096) {
      std::vector<Vec3> got_acc;
      bh_accel_all(host_tree, host_bodies, n, root, kThetaVerify, eps2, &got_acc);
      std::vector<Vec3> ref_acc;
      direct_accel_cpu(host_bodies, eps2, &ref_acc);
      require_accel_match(ref_acc, got_acc, kAccelRelTol, "nbody_bh_vs_direct");
    }

    std::vector<Vec3> acc;
    const auto stats = time_host_kernel(
        [&]() { bh_accel_all(host_tree, host_bodies, n, root, kThetaTimed, eps2, &acc); },
        kWarmupIters, kTimedIters);

    const double mean_s = stats.mean_ms / 1000.0;
    const double flops = bh_flops_estimate(static_cast<double>(n));
    const double bytes_moved = bh_bytes_estimate(static_cast<double>(n));
    const double gflops = (flops / mean_s) / 1e9;
    const double gb_s = (bytes_moved / mean_s) / 1e9;
    const double pct_flops = (gflops / kPlaceholderPeakGflops) * 100.0;
    const double pct_bw = (gb_s / kPlaceholderPeakGbS) * 100.0;
    append_csv_row("nbody_barnes_hut", prop, static_cast<std::int64_t>(n), "octree_bh_cpu",
                   stats.mean_ms, stats.stddev_ms, gflops, gb_s, pct_flops, pct_bw);
  }

  return 0;
}
