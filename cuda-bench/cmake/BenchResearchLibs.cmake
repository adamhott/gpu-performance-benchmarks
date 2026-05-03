# Optional NVIDIA libraries commonly used beyond core toolkit math:
# - cuQuantum (custatevec, cutensornet): quantum circuits, tensor networks, entanglement numerics
# - cuTENSOR: high-performance tensor contractions (many-body / lattice-style models)

option(BENCH_WITH_CUQUANTUM "Discover cuQuantum from CUQUANTUM_ROOT (custatevec + cutensornet)" ON)
option(BENCH_WITH_CUTENSOR "Discover cuTENSOR from CUTENSOR_ROOT (libcutensor)" ON)

set(BENCH_HAVE_CUQUANTUM OFF)
set(BENCH_HAVE_CUTENSOR OFF)

if(BENCH_WITH_CUQUANTUM)
  find_path(
    BENCH_CUQUANTUM_INCLUDE_DIR
    NAMES custatevec.h
    HINTS "$ENV{CUQUANTUM_ROOT}" "$ENV{cuquantum_ROOT}"
    PATH_SUFFIXES include)

  find_library(
    BENCH_CUSTATEVEC_LIBRARY
    NAMES custatevec
    HINTS "$ENV{CUQUANTUM_ROOT}" "$ENV{cuquantum_ROOT}"
    PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib)

  find_library(
    BENCH_CUTENSORNET_LIBRARY
    NAMES cutensornet
    HINTS "$ENV{CUQUANTUM_ROOT}" "$ENV{cuquantum_ROOT}"
    PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib)

  if(BENCH_CUQUANTUM_INCLUDE_DIR
     AND BENCH_CUSTATEVEC_LIBRARY
     AND BENCH_CUTENSORNET_LIBRARY)
    set(BENCH_HAVE_CUQUANTUM ON)
  else()
    message(
      STATUS
        "cuQuantum not found (optional). Install SDK and export CUQUANTUM_ROOT to enable.")
  endif()
endif()

if(BENCH_WITH_CUTENSOR)
  find_path(
    BENCH_CUTENSOR_INCLUDE_DIR
    NAMES cutensor.h
    HINTS "$ENV{CUTENSOR_ROOT}" "$ENV{CUQUANTUM_ROOT}"
    PATH_SUFFIXES include)

  find_library(
    BENCH_CUTENSOR_LIBRARY
    NAMES cutensor
    HINTS "$ENV{CUTENSOR_ROOT}" "$ENV{CUQUANTUM_ROOT}"
    PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib)

  if(BENCH_CUTENSOR_INCLUDE_DIR AND BENCH_CUTENSOR_LIBRARY)
    set(BENCH_HAVE_CUTENSOR ON)
  else()
    message(
      STATUS
        "cuTENSOR not found (optional). Install NVIDIA cuTENSOR and export CUTENSOR_ROOT to enable.")
  endif()
endif()

add_library(cuda_bench_research INTERFACE)
target_link_libraries(cuda_bench_research INTERFACE cuda_bench_math)

if(BENCH_HAVE_CUQUANTUM)
  target_include_directories(cuda_bench_research INTERFACE "${BENCH_CUQUANTUM_INCLUDE_DIR}")
  target_link_libraries(cuda_bench_research INTERFACE "${BENCH_CUSTATEVEC_LIBRARY}"
                                                     "${BENCH_CUTENSORNET_LIBRARY}")
  target_compile_definitions(cuda_bench_research INTERFACE BENCH_HAVE_CUQUANTUM=1)
endif()

if(BENCH_HAVE_CUTENSOR)
  target_include_directories(cuda_bench_research INTERFACE "${BENCH_CUTENSOR_INCLUDE_DIR}")
  target_link_libraries(cuda_bench_research INTERFACE "${BENCH_CUTENSOR_LIBRARY}")
  target_compile_definitions(cuda_bench_research INTERFACE BENCH_HAVE_CUTENSOR=1)
endif()

message(
  STATUS
  "cuda_bench_research: cuQuantum=${BENCH_HAVE_CUQUANTUM} cuTENSOR=${BENCH_HAVE_CUTENSOR}")
