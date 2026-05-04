# 04 — Vector reduction (global sum)

## Summary

Benchmark **`04_reduction`** computes the **sum** of a large length-\(n\) `float` vector \(\mathbf{x}\) on the GPU using a **two-kernel** strategy: blocks first reduce contiguous **chunks** of \(\mathbf{x}\) with **warp shuffle** reductions into per-block partial sums, then a single block reduces those partials to one scalar. The result is compared to a sequential **double-precision** CPU sum (with a tolerance that allows **reordering** of floating-point additions).

## Why we run it

Global reductions appear in **norms**, **dot products**, **residuals**, and **aggregations** across parallel codes. This benchmark stresses **warp-level primitives** (`__shfl_down_sync`), **shared-memory staging**, and **multi-kernel composition**—patterns every CUDA programmer hits when moving beyond embarrassingly parallel maps.

## Math and verification

Given \(x_0,\ldots,x_{n-1}\), the mathematical quantity is

$$
S \;=\; \sum_{i=0}^{n-1} x_i.
$$

Associativity does not hold for finite-precision floats, so parallel tree reductions may differ slightly from a left-to-right accumulator; the test allows a **mixed absolute/relative** error band tied to \(n\).

**FLOP accounting in the CSV:** the harness uses **\(n-1\)** additions as a simple scalar model for reported **GFLOP/s** (not counting warp-shuffle and memory traffic explicitly).
