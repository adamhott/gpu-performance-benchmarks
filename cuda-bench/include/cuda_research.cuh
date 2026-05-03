#pragma once

#include <cuda_runtime.h>

// Umbrella header for GPU research numerics (NR / cosmology-style PDEs, linear algebra,
// spectral methods, Monte Carlo, and optional quantum tensor-network paths).
// Link the `cuda_bench_research` CMake target so symbols resolve.
//
// Thrust / CUB ship with the CUDA Toolkit (headers; device algorithms compile with nvcc).

#include <cublas_v2.h>
#include <cufft.h>
#include <curand.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#if defined(BENCH_HAVE_CUQUANTUM)
#include <custatevec.h>
#include <cutensornet.h>
#endif

#if defined(BENCH_HAVE_CUTENSOR)
#include <cutensor.h>
#endif
