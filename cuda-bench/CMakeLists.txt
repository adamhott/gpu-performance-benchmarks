cmake_minimum_required(VERSION 3.20)
project(cuda-bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Multi-arch for Ampere (A6000), Ada (4090), Hopper (H100). Plain `86` form is
# compatible with CMake 3.20+; newer CMake can switch to `86-real` etc. to trim PTX.
set(CMAKE_CUDA_ARCHITECTURES 86 89 90)

find_package(CUDAToolkit REQUIRED)

# Shared GPU math stack for benchmarks: BLAS/RNG (02, 05), FFT (06); solver and
# sparse for future kernels. N-body (07) is hand-written CUDA; it still links this
# target for a uniform bench binary layout.
add_library(cuda_bench_math INTERFACE)
target_link_libraries(
  cuda_bench_math
  INTERFACE CUDA::cudart
            CUDA::cublas
            CUDA::curand
            CUDA::cusolver
            CUDA::cusparse
            CUDA::cufft)
if(TARGET CUDA::cublasLt)
  target_link_libraries(cuda_bench_math INTERFACE CUDA::cublasLt)
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/BenchResearchLibs.cmake")

add_compile_options(
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# nvcc forwards host compiler warnings; keep device code warning-clean.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall,-Wextra -O3 --expt-relaxed-constexpr")

function(add_cuda_bench_executable name src)
  add_executable(${name} ${src})
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(${name} PRIVATE cuda_bench_research)
  set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endfunction()

add_cuda_bench_executable(cuda_bench_01_saxpy src/01_saxpy.cu)
add_cuda_bench_executable(cuda_bench_02_sgemm src/02_sgemm.cu)
add_cuda_bench_executable(cuda_bench_03_stencil src/03_stencil.cu)
add_cuda_bench_executable(cuda_bench_04_reduction src/04_reduction.cu)
add_cuda_bench_executable(cuda_bench_05_montecarlo src/05_montecarlo.cu)
add_cuda_bench_executable(cuda_bench_06_fft src/06_fft.cu)
add_cuda_bench_executable(cuda_bench_07_nbody_tree src/07_nbody_tree.cu)
