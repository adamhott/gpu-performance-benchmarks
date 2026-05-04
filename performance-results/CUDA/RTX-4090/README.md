# RTX 4090 — performance snapshots

This folder is for **`cuda-bench`** runs on an **NVIDIA GeForce RTX 4090** (compute capability **8.9**, `sm_89`). The checked-in CSV uses **`gpu_name`** = **`NVIDIA GeForce RTX 4090`** on every row.

## RunPod hardware (recorded environment)

Summary of the host used when the checked-in **`results.csv`** was captured:

| Field | Value |
| --- | --- |
| **GPU** | 1× **NVIDIA GeForce RTX 4090** — CSV: `NVIDIA GeForce RTX 4090` |
| **vCPU** | 16 (AMD EPYC 75F3 32-Core Processor host) |
| **RAM** | 62 GB |
| **List pricing (snapshot)** | Compute **$0.69/hr**; 20 GB container storage **$0.003/hr**; **~$0.70/hr** total in UI (confirm current billing). |

## Checked-in results

This directory includes **`results.csv`** and **`consolidated.md`** from a full **`cuda-bench`** run matching the hardware above. A run that includes **`07_nbody_tree`** produces **45** data rows plus the CSV header (**five** sizes each for seven benchmarks); older CSVs without N-body have **40** data rows.

To refresh after a new run on the same or another **4090** pod:

1. Copy **`cuda-bench/results/results.csv`** from the pod and confirm **`gpu_name`** is **`NVIDIA GeForce RTX 4090`** and **`compute_capability`** is **8.9** on every row.
2. Regenerate the summary:

   ```bash
   cd performance-results/CUDA/RTX-4090
   python3 format_tables.py
   ```

3. Commit **`results.csv`** and **`consolidated.md`** when you want the snapshot versioned.
