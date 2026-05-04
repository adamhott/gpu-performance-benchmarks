# Benchmark math notes

Per-benchmark summaries: what is computed, why it is in the suite, and the governing equations in **LaTeX** form. Use **inline** math as `$ … $` and **display** math as `$$` on their own lines (blank line before and after each `$$` block). That is what [GitHub’s math renderer](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions) and most Markdown previews expect (not `\(` … `\)`).

| # | Source | Notes |
| --- | --- | --- |
| 01 | [`01_saxpy.md`](01_saxpy.md) | Level-1 BLAS SAXPY, memory-bound baseline |
| 02 | [`02_sgemm.md`](02_sgemm.md) | Dense matrix multiply, library vs hand kernel |
| 03 | [`03_stencil.md`](03_stencil.md) | 5-point Laplacian stencil, PDE-style locality |
| 04 | [`04_reduction.md`](04_reduction.md) | Global sum, parallel aggregation |
| 05 | [`05_montecarlo.md`](05_montecarlo.md) | Monte Carlo quadrature for $\pi$, RNG throughput |
| 06 | [`06_fft.md`](06_fft.md) | Complex 1D FFT via cuFFT |
| 07 | [`07_nbody_tree.md`](07_nbody_tree.md) | Barnes–Hut-style $N$-body accelerations |

## Problem sizes and timing caps

These limits are set in `cuda-bench/src/0N_*.cu` (and echoed in `scripts/run_all.sh` for Monte Carlo). They bound wall time on cloud GPUs while still spanning a useful performance range.

### Shared timing (all of 01–07)

- **Warmup iterations:** 10 (`kWarmupIters`)
- **Timed iterations:** 50 (`kTimedIters`) per problem size (after warmups)

### Largest problem size per benchmark

| Bench | Parameter | Values swept (largest last) | Upper bound |
| --- | --- | --- | --- |
| 01 SAXPY | Vector length $n$ | $2^{20}, 2^{22}, 2^{24}, 2^{26}, 2^{28}$ | **$n = 2^{28}$** floats (~268M). Sizes must fit in a signed **`int`** index. |
| 02 SGEMM | Square dimension $N$ | 256, 512, 1024, 2048, 4096 | **$N = 4096$** (naive + cuBLAS). CPU reference SGEMM is only used for $N \le 1024$ (`kCpuSgemmMaxN`); larger checks use cuBLAS. |
| 03 stencil | Grid edge $N$ ($N\times N$ points) | 1024, 2048, 4096, 6144, 8192 | **$N = 8192$** (two CUDA variants timed per $N$). |
| 04 reduction | Vector length $n$ | $2^{20}, 2^{22}, 2^{24}, 2^{26}, 2^{28}$ | **$n = 2^{28}$** (same cap as 01). |
| 05 Monte Carlo | Samples $S$ | $10^6$, $10^7$, $10^8$, $5\times 10^8$, $10^9$ | **$S = 10^9$** (dominant cost; full suite can run several minutes on some GPUs). |
| 06 FFT | Complex length $n$ | $2^{18}$, $2^{20}$, $2^{22}$, $2^{24}$, $2^{26}$ | **$n = 2^{26}$** (~64M) complex points. CSV timing is **forward C2C only** (device reset outside the timer). |
| 07 N-body | Body count $N$ | 2048, 8192, 32768, 131072, 262144 | **$N = 262144$**. Direct $O(N^2)$ CPU verification vs Barnes–Hut runs only for **$N \le 4096$**. |

### Launch and algorithm caps (secondary)

- **Grid launch caps:** several kernels cap the grid at about **`8 ×` SM count** (and never below the minimum grid needed). **04 reduction** also caps the first-stage block count at **1024**. **05 Monte Carlo** uses **`min(8192, max(1, 16 × SM count))`** blocks.
- **07:** Barnes–Hut tree walk stack depth **`kMaxStack = 192`**; timed kernel uses opening parameter **`kThetaTimed = 0.55`** (stricter **`kThetaVerify = 0.12`** for the small-$N$ vs direct check).
