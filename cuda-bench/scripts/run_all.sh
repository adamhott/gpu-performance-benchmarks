#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

mkdir -p results build

# Lock the GPU to a fixed graphics clock before timing so boost behavior does not
# skew A/B comparisons across runs or hosts. Reset after the suite so the GPU
# returns to its default dynamic clocks for normal workloads.
# Many cloud pods (RunPod, Docker) deny nvidia-smi -lgc; in that case we continue
# with a warning—benchmarks still run, clocks may vary slightly with boost.
CLOCK_LOCKED=0
LOCK_MHZ="$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || true)"
if [[ -n "${LOCK_MHZ}" ]] && nvidia-smi -lgc "${LOCK_MHZ},${LOCK_MHZ}" &>/dev/null; then
  CLOCK_LOCKED=1
  trap 'if [[ "${CLOCK_LOCKED}" -eq 1 ]]; then nvidia-smi -rgc 2>/dev/null || true; fi' EXIT
else
  echo "run_all.sh: skipping GPU clock lock (-lgc not permitted or unavailable); timings may include boost variance." >&2
  trap - EXIT
fi

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

HEADER="benchmark,gpu_name,compute_capability,problem_size,variant,mean_ms,stddev_ms,gflops,gb_s,pct_peak_flops,pct_peak_bw"
printf '%s\n' "${HEADER}" > results/results.csv

echo ">> cuda_bench_01_saxpy" >&2
./build/cuda_bench_01_saxpy

echo ">> cuda_bench_02_sgemm" >&2
./build/cuda_bench_02_sgemm

echo ">> cuda_bench_03_stencil" >&2
./build/cuda_bench_03_stencil

echo ">> cuda_bench_04_reduction" >&2
./build/cuda_bench_04_reduction

echo ">> cuda_bench_05_montecarlo (largest sample 1e9 × 60 iters — can take several minutes)" >&2
./build/cuda_bench_05_montecarlo

echo ">> cuda_bench_06_fft" >&2
./build/cuda_bench_06_fft

echo ">> cuda_bench_07_nbody_tree (CPU octree build + GPU Barnes–Hut walk per size)" >&2
./build/cuda_bench_07_nbody_tree

echo ">> done" >&2
