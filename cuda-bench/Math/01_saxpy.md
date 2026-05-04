# 01 — SAXPY (scalar multiply–add)

## Summary

Benchmark **`01_saxpy`** times a **grid-stride** CUDA kernel that applies a **SAXPY** (Single-precision **A** **X** **P**lus **Y**) update on length-$n$ vectors $\mathbf{x},\mathbf{y}$ in device memory. Each thread walks a stride of `blockDim.x * gridDim.x` indices so the whole vector is covered with tunable occupancy.

## Why we run it

SAXPY is the simplest **Level-1 BLAS** pattern: one multiply and one add per element, with **two reads and one write** per update (plus the constant $a$ in registers). It isolates **memory bandwidth** and **streaming stride** behavior before more arithmetic-heavy kernels. It is the usual first check that large-vector timings and CSV **GFLOP/s** / **GB/s** ratios look sane for a given GPU.

## Math and verification

Let $x_i, y_i \in \mathbb{R}$ (stored as `float`). The benchmark uses a fixed scalar $a = \sqrt{2}$ (`kSaxpyAlpha`). The kernel implements the BLAS SAXPY update in place on $\mathbf{y}$:

$$
y_i \leftarrow a x_i + y_i, \qquad i = 0,\ldots,n-1.
$$

**Operation count (per element):** one multiplication, one addition → **2 floating-point ops** per index $i$, hence **$2n$** useful FP32 ops for the full vector (matches how throughput is often quoted for SAXPY).

The host compares the device $\mathbf{y}$ to a CPU reference applying the same formula elementwise (relative tolerance $\sim 10^{-4}$).
