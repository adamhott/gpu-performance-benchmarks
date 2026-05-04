# 03 — 2D five-point Laplacian stencil

## Summary

Benchmark **`03_stencil`** applies a **five-point discrete Laplacian** on an \(N\times N\) periodic-grid-style update (interior uses four neighbors; **Dirichlet-style zeros** are used **outside** the domain in this implementation: missing neighbors contribute \(0\)). Two CUDA variants are timed: a **global-memory** kernel and a **shared-memory tiled** kernel. The host reference uses the same stencil formula on the CPU.

## Why we run it

Stencil updates model **elliptic PDE discretizations**, **diffusion** steps, and **Jacobi/Gauss–Seidel**-style smoothers. They are **memory-latency** and **halo-exchange** sensitive and reward **spatial locality** and **shared-memory tiling**. This benchmark separates **naive** (one read stream per neighbor) from **tiled** (halo rows cached in `__shared__`) performance on a structured grid.

## Math and verification

Let \(u_{i,j}\) denote the grid value at row \(i\), column \(j\) for \(0 \le i,j < N\). The code computes an output grid \(v\) with the **standard five-point Laplacian** (unit grid spacing \(h=1\) in the discrete formula):

$$
v_{i,j} \;=\; u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4\,u_{i,j},
$$

where terms with indices outside \([0,N-1]\) are replaced by **\(0\)** (as in the CUDA and CPU paths in `03_stencil.cu`), not periodic wrap-around.

**Operation count (for accounting in this repo):** the CSV builder uses **\(10\,N^2\)** FP ops for an \(N\times N\) field (five neighbor terms plus scaling/combine as counted in the benchmark harness).

Relative verification tolerance is on the order of **\(10^{-4}\)** against the CPU stencil.
