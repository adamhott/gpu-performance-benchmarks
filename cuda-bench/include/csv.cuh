#pragma once

#include "common.cuh"

#include <cstdint>
#include <fstream>
#include <string>

namespace bench {

inline void append_csv_row(const std::string& benchmark, const cudaDeviceProp& prop,
                           std::int64_t problem_size, const std::string& variant,
                           double mean_ms, double stddev_ms, double gflops, double gb_s,
                           double pct_peak_flops, double pct_peak_bw) {
  std::ofstream out("results/results.csv", std::ios::app);
  if (!out) {
    std::fprintf(stderr, "Failed to open results/results.csv for append\n");
    std::exit(EXIT_FAILURE);
  }
  char gpu_name[256] = {};
  device_name_string(prop, gpu_name, sizeof(gpu_name));
  const std::string cc = std::to_string(prop.major) + "." + std::to_string(prop.minor);
  out << benchmark << ',' << gpu_name << ',' << cc << ',' << problem_size << ','
      << variant << ',' << mean_ms << ',' << stddev_ms << ',' << gflops << ',' << gb_s
      << ',' << pct_peak_flops << ',' << pct_peak_bw << '\n';
}

}  // namespace bench
