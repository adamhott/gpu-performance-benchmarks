# `src/` — benchmark sources

Each `NN_name.cu` file builds to a standalone executable (`cuda_bench_NN_name`) via `CMakeLists.txt`. All binaries share timing, CSV, and optional research-library wiring from `../include/` and `../scripts/run_all.sh`.

| File | Kernel / library focus |
| --- | --- |
| `01_saxpy.cu` | `y = a*x + y`, grid-stride SAXPY |
| `02_sgemm.cu` | Naive tiled SGEMM vs **cuBLAS** |
| `03_stencil.cu` | 2D 5-point Laplacian: global vs shared-tiled |
| `04_reduction.cu` | Float sum, warp-shuffle + two-stage reduction |
| `05_montecarlo.cu` | π via **cuRAND** Philox + quarter-circle hits |
| `06_fft.cu` | 1D complex-to-complex **cuFFT** forward transform (round-trip verify) |
| `07_nbody_tree.cu` | **Barnes–Hut** octree (CPU build) + GPU tree walk for softened mutual gravity |

## Reference GPU / host environment (measured runs)

Archived CSVs and tables live under **`../../performance-results/`** — see [**`../../performance-results/README.md`**](../../performance-results/README.md). **RTX A4000** → **`RTX-A4000/`**; **RTX 4090** → **`RTX-4090/`**; **H100 80GB HBM3** → **`H100-80GB-HBM3/`**.

The RTX A4000 CSV snapshots checked in under `../../performance-results/RTX-A4000/` were captured on a RunPod **secure cloud** pod with the following **Details** (see RunPod UI for live values):

| Field | Value |
| --- | --- |
| **Pod name** | `dragon_wheel_gpu` |
| **Pod ID** | `qadeh24smpubsq` |
| **Region / class** | EUR-IS-1 (secure cloud) |
| **GPU** | 1× **NVIDIA RTX A4000** (compute capability **8.6**, `sm_86`) |
| **vCPU** | 16 (AMD EPYC 7702 64-Core Processor host) |
| **RAM** | 62 GB |
| **Container disk** | 20 GB |
| **Network volume** | `fish_pond`, 20 GB, mount **`/workspace`** |
| **Container image** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| **Template** | `runpod-torch-v240` |
| **List pricing (snapshot)** | Compute **$0.25/hr**; 20 GB container storage **$0.003/hr** (confirm in RunPod billing). |

### RTX 4090 pod (`fire_aquarium_pod`)

| Field | Value |
| --- | --- |
| **Pod name** | `fire_aquarium_pod` |
| **Pod ID** | `2qvoqmjtkp7y1n` |
| **Region / class** | US-NC-1 (secure cloud) |
| **GPU** | 1× **NVIDIA GeForce RTX 4090** (`sm_89`) — CSV: `NVIDIA GeForce RTX 4090` |
| **vCPU** | 16 (AMD EPYC 75F3 32-Core Processor host) |
| **RAM** | 62 GB |
| **Container disk** | 20 GB |
| **Container image** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| **Template** | `runpod-torch-v240` |
| **List pricing (snapshot)** | Compute **$0.69/hr**; container storage **$0.003/hr**; **~$0.70/hr** total in UI. |

### H100 80GB HBM3 pod (`mundo-fire-torch`)

| Field | Value |
| --- | --- |
| **Pod name** | `mundo-fire-torch` |
| **Pod ID** | `fuf2s04cy2ag28` |
| **Region / class** | AP-JP-1 (secure cloud) |
| **GPU** | 1× **NVIDIA H100 SXM** (80 GB HBM3, `sm_90`) — CSV: `NVIDIA H100 80GB HBM3` |
| **vCPU** | 20 (Intel Xeon Platinum 8460Y+ host) |
| **RAM** | 251 GB |
| **Container disk** | 20 GB |
| **Container image** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| **Template** | `runpod-torch-v240` |
| **List pricing (snapshot)** | Compute **$2.99/hr**; storage lines **$0.003/hr** each in UI; **~$3.00/hr** total. |

Clock locking via `nvidia-smi -lgc` is often **unavailable** in containers; `run_all.sh` skips it with a warning when the driver denies the call.

Different pods or regions will change performance slightly; update these tables when you publish numbers from a new environment.
