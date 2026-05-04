# gpu-performance-benchmarks

CUDA microbenchmarks and recorded GPU numbers for **A/B-style** comparisons across NVIDIA hardware (Ampere / Ada / Hopper-class targets). The main code lives under `**cuda-bench/`**: reproducible timings, CSV output, and optional links to NVIDIA math and research libraries.

## Contents


| Path                                                        | Role                                                                                                                                                                                                                                                                          |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `**[cuda-bench/](cuda-bench/README.md)`**                   | CMake project, seven benchmarks (SAXPY, SGEMM, stencil, reduction, Monte Carlo, 1D FFT, Barnes–Hut N-body), `scripts/run_all.sh`, and build/run docs (including RunPod).                                                                                                      |
| `**[performance-results/](performance-results/README.md)**` | Archived **CUDA** and **C++** results per GPU class under **`CUDA/`** and **`C++/`** (see the index README). |


## Quick start

```bash
cd cuda-bench
./scripts/run_all.sh
```

See `**[cuda-bench/README.md](cuda-bench/README.md)**` for requirements, library stack, and cloud notes.

## Renting GPUs in the cloud (options)

If you want to **reproduce or extend** the numbers in this repo, you typically rent a **Linux host with an NVIDIA GPU** and a driver that matches a recent **CUDA toolkit**. Vendors differ mainly in **billing model**, **contract length**, and **how much platform glue** you get around the bare GPU.


| Model                        | What it usually means                                                                                                                              | Tradeoffs                                                                                                                                  |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **On-demand / hourly**       | Pay for a VM or container **while it runs** (per minute or per hour). Spin up, run benchmarks, tear down.                                          | Highest flexibility; **$/hr** is often higher than reserved pricing; capacity can be scarce for popular SKUs (e.g. H100) at peak times.    |
| **Spot / preemptible**       | Deep discounts in exchange for **possible interruption** when capacity is reclaimed.                                                               | Excellent for fault-tolerant batch jobs; **poor default choice** for long single-run timing suites unless you snapshot results frequently. |
| **Reserved / committed use** | **1–3 year** (or similar) commit to a GPU instance family on a hyperscaler, or a provider’s **annual plan**, in return for a lower effective rate. | Best when you have **steady** GPU load; upfront or contractual obligation; less attractive for occasional microbenchmark work.             |
| **Bare-metal or colo lease** | **Fixed-term** (months–years) on dedicated servers you control.                                                                                    | Lowest **$/GPU-hour** at scale; highest operational burden (OS, image, remote hands)—usually overkill for this repository’s workflow.      |


**Where people often rent:**

- **Hyperscalers** (**[AWS](https://aws.amazon.com/ec2/instance-types/)**, **[Google Cloud](https://cloud.google.com/gpu)**, **[Azure](https://azure.microsoft.com/en-us/products/virtual-machines/linux/#gpu)**): Broad GPU catalogs, enterprise networking and identity, **reserved / savings plans** vs **on-demand** pricing. More steps to get a simple CUDA dev shell than on a GPU-focused hoster.
- **GPU-centric clouds** (e.g. **[Lambda](https://lambdalabs.com/service/gpu-cloud)**, **[CoreWeave](https://www.coreweave.com/)**, **[RunPod](https://www.runpod.io/)**, others): Emphasis on **fast allocation**, **ML/CUDA images**, and **per-hour** clarity for researchers and small teams.
- **Marketplaces / aggregators** (e.g. **[Vast.ai](https://vast.ai/)**-style supply): **Supply-driven pricing** and heterogeneous hardware; great for experimentation if you read listings carefully and accept variable quality.

This repo does not endorse a single vendor for every workload—only document what worked for **our** reproducible benchmark snapshots.

## Why RunPod for this project

**RunPod** was chosen here because it matches how `**cuda-bench`** is meant to be used: **short-lived, GPU-bound** sessions where you care about **(a)** a known CUDA image, **(b)** quick **SSH / browser terminal** access, **(c)** attaching **network storage** so clones and CSVs survive pod restarts, and **(d)** **transparent list pricing** in the UI for cost notes next to results. The **[cuda-bench/README.md](cuda-bench/README.md)** RunPod section and the **`performance-results/CUDA/*/README.md`** tables were written around that workflow.

Other platforms are equally capable of running the same binaries; we standardize on **RunPod in the docs** to keep **one** concrete path for readers (image name, region naming, volume mount at `**/workspace`**) without maintaining parallel instructions for every CSP console.

## About RunPod (brief)

**[RunPod](https://www.runpod.io/)** is a cloud platform for renting **GPU compute** (and related CPU/RAM/disk) as **pods**—typically containers from templates such as `**runpod/pytorch:…-cuda12.x-devel-ubuntu22.04`**, with optional network volumes, multiple regions, and per-pod billing shown in the dashboard. You connect over SSH or the web terminal, run `**nvidia-smi`** and `**./scripts/run_all.sh**`, then copy `**cuda-bench/results/results.csv**` off the instance.

- **Product / pricing overview:** [https://www.runpod.io/](https://www.runpod.io/)  
- **Documentation (pods, storage, CLI, billing):** [https://docs.runpod.io/](https://docs.runpod.io/)

## Measured RunPod environments

Archived **`results.csv`** / **`consolidated.md`** pairs live under **`[performance-results/](performance-results/README.md)`**: **GPU CUDA runs** under **`CUDA/`**, **CPU `cuda-bench/C++` runs** from the same pod classes under **`C++/`**. Each **`CUDA/*/README.md`** summarizes the **hardware and list-pricing snapshot** for the GPU CSV and how to regenerate **`consolidated.md`** with **`format_tables.py`**:


| CUDA snapshot folder | RunPod GPU SKU | CSV `gpu_name` (from `cuda-bench`) |
| --- | --- | --- |
| **[`CUDA/RTX-A4000/`](performance-results/CUDA/RTX-A4000/README.md)** | 1× **NVIDIA RTX A4000** | `NVIDIA RTX A4000` |
| **[`CUDA/RTX-4090/`](performance-results/CUDA/RTX-4090/README.md)** | 1× **NVIDIA GeForce RTX 4090** | `NVIDIA GeForce RTX 4090` |
| **[`CUDA/H100-80GB-HBM3/`](performance-results/CUDA/H100-80GB-HBM3/README.md)** | 1× **NVIDIA H100 SXM** (80 GB HBM3) | `NVIDIA H100 80GB HBM3` |

C++ host snapshots (same folder names under **`[performance-results/C++/](performance-results/C++/README.md)`**): **[`C++/RTX-A4000/`](performance-results/C++/RTX-A4000/README.md)**, **[`C++/RTX-4090/`](performance-results/C++/RTX-4090/README.md)**, **[`C++/H100-80GB-HBM3/`](performance-results/C++/H100-80GB-HBM3/README.md)**.


RunPod’s UI label **“H100 SXM”** is the same class of part the driver reports as `**NVIDIA H100 80GB HBM3`** in the CSV.