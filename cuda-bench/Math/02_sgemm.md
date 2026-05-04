# 02 — SGEMM (single-precision general matrix multiply)

## Summary

Benchmark **`02_sgemm`** times **square** dense matrix multiplies $C = \alpha AB + \beta C$ with **`float`** data, $\alpha = 1$, $\beta = 0$, so effectively $C = AB$, for matrix orders $N \in \{256,512,\ldots,4096\}$. It runs a **naive tiled** CUDA kernel and **cuBLAS** (`cublasSgemm`), converts layout where needed (cuBLAS uses column-major), and checks against a CPU or cuBLAS reference.

## Why we run it

Matrix multiply is the core of **Level-3 BLAS** and dominates many HPC and ML workloads. This benchmark contrasts a **hand-written** blocked kernel (shared-memory tiling) with a **highly optimized library**, exposing how much headroom remains on a given chip. It stresses **compute intensity**, **register/shared memory pressure**, and **library stack** quality.

## Math and verification

Let $A,B,C \in \mathbb{R}^{N\times N}$. In index form with row-major layout as in the reference loops,

$$
C_{ij} = \sum_{k=0}^{N-1} A_{ik} B_{kj}.
$$

With general $\alpha,\beta$:

$$
C_{ij} = \alpha \sum_{k=0}^{N-1} A_{ik} B_{kj} + \beta C_{ij}^{\text{(input)}}.
$$

This repo fixes $\alpha = 1$, $\beta = 0$.

**Operation count:** each output element requires $N$ multiply–add pairs → **$2N^3$** multiply-adds (FP32) for the multiply–accumulate phase (standard GEMM FLOP count).

The CSV reports **GFLOP/s** and **GB/s** using that $2N^3$ model and an approximate byte footprint for reads/writes of $A$, $B$, and $C$.
