# H100 80GB HBM3 — C++ host snapshots

CPU **`cuda-bench/C++`** timings captured on a host with an **NVIDIA H100 80GB HBM3** (same class as [**`../../CUDA/H100-80GB-HBM3/README.md`**](../../CUDA/H100-80GB-HBM3/README.md)).

Place **`results.csv`** here (usually by copying **`cuda-bench/results/results_cpp.csv`** after a full C++ run on that pod), then:

```bash
cd performance-results/C++/H100-80GB-HBM3
python3 ../format_tables.py
```

See [**`../README.md`**](../README.md) for the full workflow.
