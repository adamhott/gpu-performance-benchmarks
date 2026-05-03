#include "common.cuh"
#include "csv.cuh"
#include "timer.cuh"
#include "verify.cuh"

#include <cublas_v2.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int kWarmupIters = 10;
constexpr int kTimedIters = 50;
constexpr int BM = 16;
constexpr int BN = 16;
constexpr int BK = 16;
constexpr int kThreads = 256;  // 16 x 16
constexpr float kAlpha = 1.0f;
constexpr float kBeta = 0.0f;
constexpr float kGemmRelTol = 5e-3f;
// Beyond this, a literal triple-nested CPU SGEMM is too slow on typical cloud vCPUs; use
// cuBLAS as the reference result instead (still validates the naive kernel).
constexpr int kCpuSgemmMaxN = 1024;

__global__ __launch_bounds__(kThreads, 2) void sgemm_naive_tiled(const float* __restrict__ A,
                                                                 const float* __restrict__ B,
                                                                 float* __restrict__ C, int N) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x % BN;
  const int ty = threadIdx.x / BN;
  const int Row = by * BM + ty;
  const int Col = bx * BN + tx;
  float acc = 0.f;
  for (int bk = 0; bk < N; bk += BK) {
    const int a_col = bk + tx;
    const int a_row = Row;
    As[ty * BK + tx] = (a_row < N && a_col < N) ? A[a_row * N + a_col] : 0.f;

    const int b_row = bk + ty;
    const int b_col = Col;
    Bs[ty * BN + tx] = (b_row < N && b_col < N) ? B[b_row * N + b_col] : 0.f;

    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      acc += As[ty * BK + k] * Bs[k * BN + tx];
    }
    __syncthreads();
  }
  if (Row < N && Col < N) {
    C[Row * N + Col] = acc;
  }
}

inline void sgemm_cpu_row_major(const float* A, const float* B, float* C, int N) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float s = 0.f;
      for (int kk = 0; kk < N; ++kk) {
        s += A[i * N + kk] * B[kk * N + j];
      }
      C[i * N + j] = kAlpha * s + kBeta * C[i * N + j];
    }
  }
}

inline void row_major_to_col_major(const float* row, int rows, int cols, float* col) {
  for (int c = 0; c < cols; ++c) {
    for (int r = 0; r < rows; ++r) {
      col[c * rows + r] = row[r * cols + c];
    }
  }
}

inline void col_major_to_row_major(const float* col, int rows, int cols, float* row) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      row[r * cols + c] = col[c * rows + r];
    }
  }
}

inline void run_variant_csv(const char* variant_label, const bench::TimingStats& stats,
                            int N, const cudaDeviceProp& prop, double peak_gflops,
                            double peak_bw) {
  const double mean_s = stats.mean_ms / 1000.0;
  const double n3 = static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
  const double flops = 2.0 * n3;
  const double bytes =
      (2.0 * n3 + static_cast<double>(N) * static_cast<double>(N)) * static_cast<double>(sizeof(float));
  const double gflops = (flops / mean_s) / 1e9;
  const double gb_s = (bytes / mean_s) / 1e9;
  const double pct_flops = (gflops / peak_gflops) * 100.0;
  const double pct_bw = (gb_s / peak_bw) * 100.0;
  bench::append_csv_row("sgemm", prop, N, variant_label, stats.mean_ms, stats.stddev_ms, gflops,
                        gb_s, pct_flops, pct_bw);
}

}  // namespace

int main() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  bench::query_device_props(device, &prop);
  const double peak_gflops = bench::peak_fp32_gflops(prop);
  const double peak_bw = bench::peak_memory_bandwidth_gb_s(prop);

  constexpr int kNs[] = {256, 512, 1024, 2048, 4096};

  cublasHandle_t handle{};
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cublasCreate failed\n");
    return EXIT_FAILURE;
  }

  for (int N : kNs) {
    const std::size_t elems = static_cast<std::size_t>(N) * static_cast<std::size_t>(N);
    std::vector<float> hA(elems);
    std::vector<float> hB(elems);
    std::vector<float> hC(elems, 0.f);
    for (std::size_t i = 0; i < elems; ++i) {
      hA[i] = static_cast<float>((i * 13) % 1000) * 0.0001f - 0.05f;
      hB[i] = static_cast<float>((i * 7) % 1001) * 0.0001f - 0.05f;
    }
    std::vector<float> gold(elems, 0.f);
    if (N <= kCpuSgemmMaxN) {
      sgemm_cpu_row_major(hA.data(), hB.data(), gold.data(), N);
    }

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;
    const std::size_t bytes = elems * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    std::vector<float> hA_col(elems);
    std::vector<float> hB_col(elems);
    std::vector<float> hC_col(elems, 0.f);
    row_major_to_col_major(hA.data(), N, N, hA_col.data());
    row_major_to_col_major(hB.data(), N, N, hB_col.data());

    if (N > kCpuSgemmMaxN) {
      CUDA_CHECK(cudaMemcpy(dA, hA_col.data(), bytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dB, hB_col.data(), bytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemset(dC, 0, bytes));
      if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &kAlpha, dA, N, dB, N, &kBeta, dC,
                      N) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasSgemm (reference) failed\n");
        return EXIT_FAILURE;
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(hC_col.data(), dC, bytes, cudaMemcpyDeviceToHost));
      col_major_to_row_major(hC_col.data(), N, N, gold.data());
    }

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, bytes));

    const dim3 block_dim(static_cast<unsigned int>(kThreads));
    const dim3 grid_dim(static_cast<unsigned int>((N + BN - 1) / BN),
                        static_cast<unsigned int>((N + BM - 1) / BM));

    sgemm_naive_tiled<<<grid_dim, block_dim>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(elems);
    CUDA_CHECK(cudaMemcpy(got.data(), dC, bytes, cudaMemcpyDeviceToHost));
    bench::require_relative_match(got, gold, kGemmRelTol, "sgemm_naive");

    const bench::TimingStats naive_stats = bench::time_cuda_kernel(
        [&]() { sgemm_naive_tiled<<<grid_dim, block_dim>>>(dA, dB, dC, N); },
        kWarmupIters, kTimedIters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    run_variant_csv("naive", naive_stats, N, prop, peak_gflops, peak_bw);

    CUDA_CHECK(cudaMemcpy(dA, hA_col.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB_col.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, bytes));

    if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &kAlpha, dA, N, dB, N, &kBeta, dC,
                    N) != CUBLAS_STATUS_SUCCESS) {
      std::fprintf(stderr, "cublasSgemm failed\n");
      return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC_col.data(), dC, bytes, cudaMemcpyDeviceToHost));
    std::vector<float> cublas_row(elems);
    col_major_to_row_major(hC_col.data(), N, N, cublas_row.data());
    bench::require_relative_match(cublas_row, gold, kGemmRelTol, "sgemm_cublas");

    CUDA_CHECK(cudaMemcpy(dA, hA_col.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB_col.data(), bytes, cudaMemcpyHostToDevice));

    const bench::TimingStats cublas_stats = bench::time_cuda_kernel(
        [&]() {
          cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &kAlpha, dA, N, dB, N, &kBeta, dC, N);
        },
        kWarmupIters, kTimedIters);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_variant_csv("cublas", cublas_stats, N, prop, peak_gflops, peak_bw);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
  }

  cublasDestroy(handle);
  return 0;
}
