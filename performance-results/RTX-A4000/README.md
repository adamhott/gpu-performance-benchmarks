# RTX A4000 — performance snapshots

This folder holds **human-readable summaries** of full `cuda-bench` runs on an **NVIDIA RTX A4000** (compute capability **8.6**, `sm_86`). The CSV **`gpu_name`** column is **`NVIDIA RTX A4000`**.

The checked-in `results.csv` should match a real **`cuda-bench`** run (e.g. copy from the pod or download via Jupyter). Regenerate `consolidated.md` after every update.

## RunPod hardware (recorded environment)

Summary of the host used when the checked-in **`results.csv`** was captured:

| Field | Value |
| --- | --- |
| **GPU** | 1× **NVIDIA RTX A4000** — CSV: `NVIDIA RTX A4000` |
| **vCPU** | 16 (AMD EPYC 7702 64-Core Processor host) |
| **RAM** | 62 GB |
| **List pricing (snapshot)** | Compute **$0.25/hr**; 20 GB container storage **$0.003/hr**; **~$0.25/hr** total in UI at time of capture (verify in-console billing). |

Future pods may differ; update this table when you record a new environment.

## Refreshing

```bash
cd performance-results/RTX-A4000
python3 format_tables.py
```
