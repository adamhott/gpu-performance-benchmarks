#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(expr)                                                         \
  do {                                                                           \
    cudaError_t _cuda_check_status = (expr);                                   \
    if (_cuda_check_status != cudaSuccess) {                                    \
      std::fprintf(stderr, "%s:%d CUDA error %s: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorName(_cuda_check_status),                       \
                   cudaGetErrorString(_cuda_check_status));                    \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

namespace bench {

inline void query_device_props(int device, cudaDeviceProp* prop) {
  CUDA_CHECK(cudaGetDeviceProperties(prop, device));
}

// Peak memory bandwidth (GB/s): effective mem clock (DDR) * bus width / 8 bits.
// memoryClockRate is in kHz per CUDA documentation.
inline double peak_memory_bandwidth_gb_s(const cudaDeviceProp& prop) {
  const double mem_khz = static_cast<double>(prop.memoryClockRate);
  const double bus_bits = static_cast<double>(prop.memoryBusWidth);
  const double ddr_factor = 2.0;
  return (mem_khz * 1000.0 * ddr_factor * bus_bits / 8.0) / 1e9;
}

// FP32 FMA throughput per SM per clock cycle by compute capability.
// cudaDeviceProp does not expose "CUDA cores per SM"; this uses only the
// device's (major, minor) from prop — not GPU marketing names.
inline int fp32_fma_ops_per_sm_per_cycle(int major, int minor) {
  const int cc = major * 10 + minor;
  switch (cc) {
    case 86:  // GA102 class (A6000, RTX 3090, etc.)
    case 87:
    case 89:  // AD102 class (RTX 4090)
    case 90:  // GH100 (H100 SXM / PCIe FP32 non-sparse peak uses same SM width)
      return 128;
    default:
      // Conservative fallback: treat unknown CC like Ampere-class FP32 width.
      return 128;
  }
}

// Peak FP32 TFLOP/s: SMs * (FMA ops/SM/cycle) * 2 (FMA = 2 FLOPs) * core clock.
// clockRate in cudaDeviceProp is in kHz (GPU / SM clock, not mem clock).
inline double peak_fp32_tflops(const cudaDeviceProp& prop) {
  const double sm_count = static_cast<double>(prop.multiProcessorCount);
  const double fma_per_sm = static_cast<double>(
      fp32_fma_ops_per_sm_per_cycle(prop.major, prop.minor));
  const double sm_clock_hz = static_cast<double>(prop.clockRate) * 1000.0;
  const double flops_per_cycle = fma_per_sm * 2.0;
  return (sm_count * flops_per_cycle * sm_clock_hz) / 1e12;
}

inline double peak_fp32_gflops(const cudaDeviceProp& prop) {
  return peak_fp32_tflops(prop) * 1000.0;
}

inline void device_name_string(const cudaDeviceProp& prop, char* out, size_t out_len) {
  if (out_len == 0) {
    return;
  }
  std::size_t n = 0;
  while (n < sizeof(prop.name) && prop.name[n] != '\0') {
    ++n;
  }
  const std::size_t max_copy = out_len - 1u;
  const std::size_t copy = n < max_copy ? n : max_copy;
  if (copy > 0u) {
    std::memcpy(out, prop.name, copy);
  }
  out[copy] = '\0';
}

}  // namespace bench
