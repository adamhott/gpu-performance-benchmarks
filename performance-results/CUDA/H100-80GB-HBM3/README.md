# H100 80GB HBM3 — performance snapshots

This folder holds a full **`cuda-bench`** run on **Hopper**: RunPod lists the SKU as **NVIDIA H100 SXM** (80 GB HBM3). The driver reports **`gpu_name`** = **`NVIDIA H100 80GB HBM3`** and **`compute_capability`** = **9.0** (`sm_90`) in **`results.csv`** — same hardware, different naming strings.

## RunPod hardware (recorded environment)

Summary of the host used when the checked-in **`results.csv`** was captured:

| Field | Value |
| --- | --- |
| **GPU** | 1× **NVIDIA H100 SXM** (80 GB HBM3, `sm_90`) — CSV: `NVIDIA H100 80GB HBM3` |
| **vCPU** | 20 (Intel Xeon Platinum 8460Y+ host) |
| **RAM** | 251 GB |
| **Driver / CUDA (session)** | **550.163.01** / CUDA **12.4** (`nvcc` 12.4.x) |
| **List pricing (snapshot)** | Compute **$2.99/hr**; container + volume storage **$0.003/hr** each (20 GB lines in UI); **~$3.00/hr** total (confirm current billing). |

## Checked-in results

The suite includes **seven** benchmarks (**45** data rows + CSV header when **`07_nbody_tree`** is present).

## Refreshing

```bash
cd performance-results/CUDA/H100-80GB-HBM3
python3 format_tables.py
```
