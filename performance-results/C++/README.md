# C++ host performance snapshots

Each **`RTX-A4000/`**, **`RTX-4090/`**, and **`H100-80GB-HBM3/`** subdirectory holds **`results.csv`** from the **`cuda-bench/C++`** ports (CPU-only), captured on a machine in the **same GPU class** as the paired CUDA archive under **`../CUDA/<same-name>/`**. That way A4000 vs 4090 vs H100 **host CPU / memory** differences stay separated even though the CSV always reports `gpu_name` = **`CPU`**.

## Capturing a snapshot

From **`cuda-bench/`** on the pod (after building the C++ targets, same repo layout as the GPU run):

```bash
cd cuda-bench
cmake -S C++ -B build_cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpp -j
# run the suite (your loop or individual binaries); results append to results/results_cpp.csv
```

Copy the CSV into the matching archive folder (repo root relative paths):

```bash
cp cuda-bench/results/results_cpp.csv performance-results/C++/RTX-4090/results.csv
cd performance-results/C++/RTX-4090
python3 ../format_tables.py
```

Use the folder that matches the **GPU SKU** of the pod, not the word “CPU” in the CSV.

Shared formatter: **`format_tables.py`** in this directory (run from a GPU subfolder so the script can infer the archive name from the current directory).

Parent index: [**`../README.md`**](../README.md).
