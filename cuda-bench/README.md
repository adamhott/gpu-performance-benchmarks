# cuda-bench

CUDA microbenchmarks for comparing NVIDIA GPUs with reproducible timing and CSV output for later plotting and cost analysis. Archived runs in this repo target **RTX A4000**, **GeForce RTX 4090**, and **H100 80GB HBM3** (Hopper); the build also covers other **`sm_86` / `sm_89` / `sm_90`** parts such as **L40S** or **RTX A6000**-class boards without extra CMake flags.

## Requirements

- Ubuntu 22.04 (or compatible Linux with NVIDIA driver)
- CUDA Toolkit 12.4 or newer (`nvcc` on `PATH`)
- CMake 3.20 or newer
- NVIDIA GPU with driver supporting your toolkit version

All benchmark binaries link the **`cuda_bench_research`** CMake target, which wraps:

- **`cuda_bench_math` (toolkit):** **cuBLAS**, optional **cuBLASLt**, **cuRAND**, **cuSOLVER**, **cuSPARSE**, **cuFFT** — dense/sparse linear algebra, decompositions, FFTs, and RNG (PDE discretizations, hydro-style operators, spectral / CMB map numerics, Monte Carlo).
- **Optional SDKs (auto-detected):** **cuQuantum** (`custatevec`, `cutensornet`) if `CUQUANTUM_ROOT` points at an install — quantum circuits, tensor contractions, and entanglement-focused many-body numerics. **cuTENSOR** if `CUTENSOR_ROOT` is set — high-performance tensor contractions common in lattice-style and tensor-network workflows.
- **Headers:** `include/cuda_research.cuh` pulls the toolkit C APIs plus **Thrust** (ships with CUDA) for scans, reductions, and device containers. Optional cuQuantum / cuTENSOR includes are guarded by `BENCH_HAVE_CUQUANTUM` and `BENCH_HAVE_CUTENSOR` (defined by CMake when those libraries are found).

Turn off discovery with `-DBENCH_WITH_CUQUANTUM=OFF` or `-DBENCH_WITH_CUTENSOR=OFF` if you want a quieter configure on machines without those SDKs.

The build embeds SASS for `sm_86`, `sm_89`, and `sm_90`. The driver JIT-falls back when a binary is missing an architecture, but for best performance ship a binary built on or for the same major generation you will run. On RunPod, pick a GPU type that matches one of those arches—for example **RTX A4000** (`sm_86`), **GeForce RTX 4090** or **L40S** (`sm_89`), or **H100 SXM** (`sm_90`)—without changing CMake.

## Build

From this directory:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Executables are named `cuda_bench_<id>_<name>` (for example `build/cuda_bench_01_saxpy`).

Compiler flags aim for a clean build with host warnings enabled (`-Wall -Wextra -O3` forwarded where appropriate for `nvcc`).

## Running benchmarks

`scripts/run_all.sh` configures the environment, builds, writes the CSV header, and runs all benchmarks listed in the script:

```bash
chmod +x scripts/run_all.sh
./scripts/run_all.sh
```

Each invocation of `run_all.sh` recreates `results/results.csv` with the header row, then benchmark binaries append one row per measurement for that run. The **N-body** benchmark (`07`) spends **CPU time building an octree** each size (not included in the GPU kernel timing); the CSV row reflects **only** the Barnes–Hut GPU traversal. The optional **`C++/`** host ports write CPU rows to **`results/results_cpp.csv`** so they never overwrite the GPU CSV from this script.

## Running on RunPod

1. Pick a template with CUDA 12.x and a matching driver for your GPU (RunPod’s PyTorch or CUDA base images usually work).
2. Clone or copy this repository into the pod and `cd cuda-bench` (if you use a network volume mounted at `/workspace`, keep the repo under that path so it survives pod restarts).
3. Install CMake if missing (`apt-get update && apt-get install -y cmake` as root when `sudo` is not installed in the image).
4. Run `./scripts/run_all.sh` once the GPU is visible (`nvidia-smi` should succeed).
5. Copy `results/results.csv` off the pod (Jupyter download, `scp` from your Mac, or a sync’d volume) before terminating the instance if you want to keep the numbers.

For cost tracking, record the pod’s GPU type, hourly rate, and the wall time reported by the script or your session timer so you can join those fields with `gflops` in later analysis.

### C++ host benchmarks on RunPod

The **`C++/`** tree is **CPU-only** (no `nvcc`); it is useful for host-side reference timings on the **same pod** you use for CUDA. PyTorch/CUDA **devel** templates on RunPod usually include **`g++`**; if `g++ --version` fails, install a toolchain as root (images often have no `sudo`):

```bash
apt-get update && apt-get install -y build-essential
```

Still from **`cuda-bench/`** (same clone and `cd` as above):

```bash
mkdir -p results
cmake -S C++ -B build_cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpp -j
```

Each binary **appends** rows to **`results/results_cpp.csv`**. Remove that file first if you want a single clean sweep without mixing older runs:

```bash
rm -f results/results_cpp.csv
for exe in build_cpp/cpp_bench_*; do "./${exe}"; done
```

The Monte Carlo port can take noticeable wall time on large CPU settings; you can run individual **`build_cpp/cpp_bench_0N_*`** executables if you need a quicker partial pass.

Copy **`results/results_cpp.csv`** off the pod before you delete the instance. To version numbers next to your GPU class, follow **`../performance-results/C++/README.md`** (copy into **`performance-results/C++/RTX-4090/`**, **`…/H100-80GB-HBM3/`**, etc., matching the pod SKU, then run **`python3 ../format_tables.py`** from that folder).

### Example environments for archived snapshots

The checked-in CSVs under **`../performance-results/CUDA/`** were captured on RunPod **secure cloud** hosts summarized below. RunPod’s UI may say **H100 SXM** while the CSV uses **`gpu_name`** = **`NVIDIA H100 80GB HBM3`**—same GPU class. SKUs, clocks, and pricing change over time; treat these rows as **documentation of the runs that produced each snapshot**, not a guarantee for future pods. CPU **`cuda-bench/C++`** archives for the same three GPU classes live under **`../performance-results/C++/`** (see **`../performance-results/C++/README.md`**).

#### RTX A4000 (`../performance-results/CUDA/RTX-A4000/`)

| Field | Value |
| --- | --- |
| **GPU** | 1× **NVIDIA RTX A4000** (compute capability **8.6**, `sm_86`) — CSV: `NVIDIA RTX A4000` |
| **vCPU** | 16 (AMD EPYC 7702 64-Core Processor host) |
| **RAM** | 62 GB |
| **List pricing (snapshot)** | Compute **$0.25/hr**; 20 GB container storage **$0.003/hr**; **~$0.25/hr** total in UI at capture (confirm current RunPod billing). |

#### GeForce RTX 4090 (`../performance-results/CUDA/RTX-4090/`)

| Field | Value |
| --- | --- |
| **GPU** | 1× **NVIDIA GeForce RTX 4090** (compute capability **8.9**, `sm_89`) — CSV: `NVIDIA GeForce RTX 4090` |
| **vCPU** | 16 (AMD EPYC 75F3 32-Core Processor host) |
| **RAM** | 62 GB |
| **List pricing (snapshot)** | Compute **$0.69/hr**; 20 GB container storage **$0.003/hr**; **~$0.70/hr** total in UI (confirm current RunPod billing). |

See [**`../performance-results/CUDA/RTX-4090/README.md`**](../performance-results/CUDA/RTX-4090/README.md) for the snapshot notes and **`python3 format_tables.py`** refresh.

#### H100 80GB HBM3 / Hopper (`../performance-results/CUDA/H100-80GB-HBM3/`)

| Field | Value |
| --- | --- |
| **GPU** | 1× **NVIDIA H100 SXM** (80 GB HBM3, `sm_90`) — CSV: `NVIDIA H100 80GB HBM3` |
| **vCPU** | 20 (Intel Xeon Platinum 8460Y+ host) |
| **RAM** | 251 GB |
| **Driver / CUDA (session)** | **550.163.01** / CUDA **12.4** (`nvcc` 12.4.x) |
| **List pricing (snapshot)** | Compute **$2.99/hr**; container + volume storage **$0.003/hr** each (20 GB lines in UI); **~$3.00/hr** total (confirm current RunPod billing). |

See [**`../performance-results/CUDA/H100-80GB-HBM3/README.md`**](../performance-results/CUDA/H100-80GB-HBM3/README.md) for the snapshot notes and **`python3 format_tables.py`** refresh.

## Research directions (beyond this repo’s microbenchmarks)

CUDA here is a **numeric substrate**. For **numerical relativity / binary black holes**, production science codes are usually separate applications (often MPI + AMR) that *use* GPUs: for example **AMReX** (block-structured AMR with CUDA), community stacks around **Einstein Toolkit**, or the **Simulating eXtreme Spacetimes (SXS)** / **SpEC** ecosystem — pair those with the same math libraries you already link. For **cosmology / N-body / hydro**, look at **Gadget**, **Arepo**, **SWIFT**, or **PKDGRAV3** GPU paths as examples of full applications. For **quantum many-body / circuits**, use **cuQuantum** (wired above when installed) alongside or instead of hand-rolled tensor kernels. This repository stays a **timing and A/B harness**; the README and `cuda_research.cuh` are meant to align your CMake link line with where those larger codes sit in the NVIDIA stack.

## Repository layout

- `include/` — timing, verification helpers, device peak metrics, `cuda_research.cuh` (research-oriented umbrella includes)
- `cmake/` — optional library discovery (`BenchResearchLibs.cmake`)
- `src/` — one executable per benchmark (`01_saxpy` … `07_nbody_tree`)
- `Math/` — per-benchmark math summaries and LaTeX equations ([`Math/README.md`](Math/README.md))
- `scripts/` — orchestration and plotting (plotting added when more benchmarks land)
- `results/` — CSV output consumed by plotting scripts
