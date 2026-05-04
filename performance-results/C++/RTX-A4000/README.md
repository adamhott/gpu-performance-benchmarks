# RTX A4000 — C++ host snapshots

CPU **`cuda-bench/C++`** timings captured on a host with an **NVIDIA RTX A4000** (same class as [**`../../CUDA/RTX-A4000/README.md`**](../../CUDA/RTX-A4000/README.md)).

Place **`results.csv`** here (usually by copying **`cuda-bench/results/results_cpp.csv`** after a full C++ run on that pod), then:

```bash
cd performance-results/C++/RTX-A4000
python3 ../format_tables.py
```

See [**`../README.md`**](../README.md) for the full workflow.
