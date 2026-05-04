# Performance snapshots

Archived benchmark outputs are split by **how they were measured** and by **GPU class** (the pod / machine that hosted the run).

| Branch | Contents |
| --- | --- |
| [**`CUDA/`**](CUDA/README.md) | **`results.csv`** from full **`cuda-bench`** GPU runs (`cuda-bench/scripts/run_all.sh`), plus **`consolidated.md`** from **`format_tables.py`**. |
| [**`C++/`**](C++/README.md) | **`results.csv`** from the portable **`cuda-bench/C++`** host ports (CPU timing), one folder per GPU class so numbers stay paired with the same RunPod hardware notes as the CUDA snapshot. |

## CUDA snapshots (`CUDA/`)

| Directory | GPU |
| --- | --- |
| [**`CUDA/RTX-A4000/`**](CUDA/RTX-A4000/README.md) | **RTX A4000** (`sm_86`) — CSV `NVIDIA RTX A4000` |
| [**`CUDA/RTX-4090/`**](CUDA/RTX-4090/README.md) | **GeForce RTX 4090** (`sm_89`) — CSV `NVIDIA GeForce RTX 4090` |
| [**`CUDA/H100-80GB-HBM3/`**](CUDA/H100-80GB-HBM3/README.md) | **H100 SXM** 80 GB HBM3 (`sm_90`) — CSV `NVIDIA H100 80GB HBM3` |

Regenerate **`consolidated.md`** from the snapshot directory that contains **`results.csv`**:

```bash
cd performance-results/CUDA/RTX-A4000   # or RTX-4090 / H100-80GB-HBM3
python3 format_tables.py
```

## C++ host snapshots (`C++/`)

Same three folder names under **`C++/`** hold CPU **`results.csv`** (and optional **`consolidated.md`**) for runs captured on hosts in the same GPU class. See [**`C++/README.md`**](C++/README.md).
