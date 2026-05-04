# Benchmark math notes

Per-benchmark summaries: what is computed, why it is in the suite, and the governing equations in **LaTeX** form. Equations use GitHub-style math delimiters (`$$ … $$` for display, `\( … \)` for inline); they render on [github.com](https://github.com) when viewing these files in the web UI.

| # | Source | Notes |
| --- | --- | --- |
| 01 | [`01_saxpy.md`](01_saxpy.md) | Level-1 BLAS SAXPY, memory-bound baseline |
| 02 | [`02_sgemm.md`](02_sgemm.md) | Dense matrix multiply, library vs hand kernel |
| 03 | [`03_stencil.md`](03_stencil.md) | 5-point Laplacian stencil, PDE-style locality |
| 04 | [`04_reduction.md`](04_reduction.md) | Global sum, parallel aggregation |
| 05 | [`05_montecarlo.md`](05_montecarlo.md) | Monte Carlo quadrature for \( \pi \), RNG throughput |
| 06 | [`06_fft.md`](06_fft.md) | Complex 1D FFT via cuFFT |
| 07 | [`07_nbody_tree.md`](07_nbody_tree.md) | Barnes–Hut–style \( N \)-body accelerations |
