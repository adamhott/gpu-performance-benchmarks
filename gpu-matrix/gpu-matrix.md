```markdown
# GPU CUDA Benchmark Analysis Report
**Repository:** `adamhott/gpu-performance-benchmarks` — `performance-results/CUDA/`  
**Date of Analysis:** 2026-05-04  
**GPUs Evaluated:** NVIDIA H100 80GB HBM3 · NVIDIA GeForce RTX 4090 · NVIDIA RTX A4000

---

## 1. Repository Structure

Each GPU subdirectory contains four files:

| File | Purpose |
|------|---------|
| `README.md` | Hardware environment, RunPod host specs, and pricing notes |
| `results.csv` | Raw benchmark data — one row per (benchmark × problem_size × variant) |
| `consolidated.md` | Human-readable tables generated from the CSV |
| `format_tables.py` | Script that regenerates `consolidated.md` from `results.csv` |

---

## 2. Hardware Reference

| Field | H100 80GB HBM3 | RTX 4090 | RTX A4000 |
|---|---|---|---|
| **GPU Name (CSV)** | NVIDIA H100 80GB HBM3 | NVIDIA GeForce RTX 4090 | NVIDIA RTX A4000 |
| **Architecture** | Hopper | Ada Lovelace | Ampere |
| **Compute Capability** | 9.0 (sm_90) | 8.9 (sm_89) | 8.6 (sm_86) |
| **VRAM** | 80 GB HBM3 | 24 GB GDDR6X | 16 GB GDDR6 |
| **vCPU (RunPod)** | 20 (Intel Xeon Platinum 8460Y+) | 16 (AMD EPYC 75F3) | 16 (AMD EPYC 7702) |
| **Host RAM** | 251 GB | 62 GB | 62 GB |
| **CUDA Driver** | 550.163.01 / CUDA 12.4 | — | — |
| **RunPod Compute Price** | ~$3.00/hr | ~$0.70/hr | ~$0.25/hr |

---

## 3. Metric Definitions, Summaries, and Practical Use Cases

The benchmark CSV contains eleven columns. The six performance metrics are described below.

### 3.1 `mean_ms` — Mean Kernel Execution Time (milliseconds)

**Definition.** The arithmetic mean of repeated kernel invocation times, measured in milliseconds (ms), typically using CUDA events placed around the device-side execution.

**Summary.** This is the raw wall-clock cost of executing the kernel at a given problem size. Lower is faster. It is strongly affected by problem size and algorithmic complexity, so it is most useful for absolute latency comparisons at a fixed problem size.

**Practical Use Case for Complex GPU Math.** In iterative solvers (e.g., conjugate gradient, Jacobi iterations for PDEs), each iteration involves a matrix-vector product or stencil sweep. `mean_ms` directly tells you how long one step takes, making it the primary metric for estimating total time-to-solution when the iteration count is known. For real-time simulation (e.g., fluid dynamics at fixed timesteps), it tells you whether a kernel fits within the timestep budget.

**Why It Matters.** It is the ground-truth answer to "how fast is this?" No other derived metric can substitute for it when you need an absolute latency figure. Extremely important for latency-sensitive pipelines.

---

### 3.2 `stddev_ms` — Standard Deviation of Execution Time (milliseconds)

**Definition.** The sample standard deviation across repeated kernel runs, in milliseconds. A high `stddev_ms` relative to `mean_ms` indicates inconsistent execution.

**Summary.** Measures the run-to-run variability of the kernel. Small values indicate a stable, reproducible workload. Large values (e.g., `stddev_ms / mean_ms > 5%`) may indicate GPU clock throttling, OS jitter, cache-warming effects, or non-deterministic memory access patterns.

**Practical Use Case for Complex GPU Math.** In Monte Carlo simulations or neural-network training where many kernel invocations are batched, a high stddev degrades throughput predictability and makes profiling unreliable. For production numerical workloads (e.g., reservoir simulation, weather modelling), low variance ensures that run-time estimates given to downstream pipeline stages are trustworthy. For benchmarking, a tight stddev validates that the mean is meaningful rather than an average of outliers.

**Why It Matters.** A metric that is often ignored but crucial for scientific reproducibility and production reliability. When choosing between two implementations with similar `mean_ms`, the one with lower `stddev_ms` is preferable because its worst-case behaviour is closer to its average-case behaviour.

---

### 3.3 `gflops` — Achieved Floating-Point Throughput (GFLOP/s)

**Definition.** Giga floating-point operations per second actually performed by the kernel, computed as `(floating_point_ops_in_problem) / (mean_ms × 10⁻³) / 10⁹`. The numerator is derived analytically from the algorithm (e.g., for an N×N SGEMM: `2N³` FLOPs for multiply-accumulate).

**Summary.** Expresses how much arithmetic work the GPU is completing per unit time. Higher is better. It is algorithm-specific: a memory-bound kernel (SAXPY) will naturally report far fewer GFLOP/s than a compute-bound kernel (SGEMM) because most of its time is spent waiting on memory, not executing arithmetic units.

**Practical Use Case for Complex GPU Math.** GFLOP/s is the central performance currency for dense linear algebra (BLAS level-3), FFT, and spectral solvers. When fitting a model to hardware: an N-body simulation doing O(N²) force evaluations, or a turbulence simulation using spectral methods, GFLOP/s determines how large a problem you can solve in a fixed time budget. It also lets you compare across different hardware generations with different clock speeds — a raw time comparison would unfairly favour faster-clocked consumer GPUs over slower-clocked professional cards.

**Why It Matters.** It is the standard currency for comparing dense compute workloads. For FP32 SGEMM, you want GFLOP/s to approach the theoretical peak (which pct_peak_flops quantifies). For memory-bound kernels, tracking GFLOP/s alongside GB/s helps identify the performance bottleneck.

---

### 3.4 `gb_s` — Achieved Memory Bandwidth (GB/s)

**Definition.** Gigabytes of data transferred between GPU global memory (DRAM) and the streaming multiprocessors per second, computed as `(bytes_read + bytes_written) / (mean_ms × 10⁻³) / 10⁹`.

**Summary.** Measures how efficiently the kernel exploits the GPU's memory subsystem. For memory-bound kernels (SAXPY, reduction, stencil), this is the dominant performance metric — the kernel is idle most of the time waiting for data from DRAM. For compute-bound kernels (large SGEMM with cuBLAS), the GPU's arithmetic units are the bottleneck and `gb_s` is less diagnostic.

**Practical Use Case for Complex GPU Math.** Many scientific workloads are memory-bandwidth limited: sparse matrix-vector products, stencil codes for finite-difference PDE solvers, graph analytics, and reduction passes in iterative methods. For these, `gb_s` directly governs throughput — if you achieve 80% of peak memory bandwidth, you are close to optimal for the algorithm. Bandwidth also matters for data movement across GPU–CPU or multi-GPU boundaries (though the benchmark measures device-only BW here).

**Why It Matters.** For bandwidth-limited kernels, `gb_s` is the performance metric. The ratio of `gflops` to `gb_s` (arithmetic intensity) determines which resource limits throughput — a value below the machine's "roofline knee" (FP32 peak GFLOP/s divided by peak GB/s) implies a memory-bound kernel. H100's HBM3 memory makes this metric especially critical: its peak bandwidth is more than 3× that of the RTX A4000's GDDR6, and this directly drives the gap in memory-bound workload performance.

---

### 3.5 `pct_peak_flops` — Percentage of Theoretical Peak FP32 Throughput

**Definition.** `pct_peak_flops = (achieved_gflops / theoretical_peak_gflops) × 100`. The theoretical peak is the GPU's published FP32 TFLOP/s figure (H100 SXM: ~66.9 TFLOP/s; RTX 4090: ~82.6 TFLOP/s; RTX A4000: ~19.2 TFLOP/s).

**Summary.** A normalised efficiency measure that answers "what fraction of the GPU's maximum arithmetic capability is this kernel using?" Values above 70% are considered excellent for dense BLAS. Most real workloads achieve 10–50%. A very low value (e.g., <5%) indicates the kernel is compute-underutilised — it is memory-bound, latency-bound, or poorly implemented.

**Practical Use Case for Complex GPU Math.** When optimising a custom CUDA kernel for dense numerical computation (e.g., a custom tensor contraction for quantum chemistry, or a mixed-precision solver), `pct_peak_flops` tells you how much headroom remains. If `pct_peak_flops` is already 75%, further algorithmic tuning is unlikely to yield much — hardware is near its limit. If it is 10%, there is substantial potential improvement by improving data reuse (tiling, shared memory) or restructuring the computation to increase arithmetic intensity.

**Why It Matters.** Allows apples-to-apples comparison of kernel efficiency across different GPU generations, independent of absolute clock-speed differences. A kernel achieving 77% on an H100 is using the hardware almost as efficiently as one achieving 68% on an RTX 4090 — even though the absolute GFLOP/s figures differ enormously. Essential when deciding whether to rewrite a kernel or whether it is already hardware-limited.

---

### 3.6 `pct_peak_bw` — Percentage of Theoretical Peak Memory Bandwidth

**Definition.** `pct_peak_bw = (achieved_gb_s / theoretical_peak_gb_s) × 100`. Peak values used by the benchmark: H100 SXM HBM3 ~3,350 GB/s; RTX 4090 GDDR6X ~1,008 GB/s; RTX A4000 GDDR6 ~448 GB/s.

**Summary.** Mirrors `pct_peak_flops` but for the memory subsystem. Values consistently above 80% indicate that a memory-bound kernel is very well-optimised. Values well below 50% despite a memory-bound workload suggest coalescing problems, excessive cache misses, strided access patterns, or insufficient parallelism to saturate the memory bus.

**Practical Use Case for Complex GPU Math.** For stencil codes (finite-difference, finite-element), the Roofline model says performance is bounded by `min(peak_GFLOP/s / arithmetic_intensity, peak_GB/s)`. A stencil with low arithmetic intensity will be constrained by `pct_peak_bw`. Similarly, reduction operations (summing a large array for dot products, norms, energy sums in MD simulations) are bandwidth-limited; you want `pct_peak_bw` to be high. If it is low despite correct code, the fix is usually improving memory access patterns (coalescing) or increasing the thread-block occupancy.

**Why It Matters.** For memory-bound kernels this is THE metric. The H100's >3,000 GB/s HBM3 bandwidth versus the RTX A4000's ~448 GB/s GDDR6 means the H100 can process 7× more data per second for memory-bound workloads — even if its compute advantage is "only" ~3.5×. Choosing the right GPU for a memory-bandwidth-limited workload can determine feasibility of real-time operation.

---

## 4. Per-Benchmark Analysis

### 4.1 SAXPY (Single-Precision A·X Plus Y)

**Description.** Executes `Y[i] = a * X[i] + Y[i]` across a vector of floats using a grid-stride loop. This is BLAS Level-1 `saxpy`. Two reads and one write per element with two FLOPs — extremely low arithmetic intensity (≈0.33 FLOP/byte), making it a canonical memory-bandwidth benchmark.

**Results at largest problem size (268M elements):**

| GPU | mean_ms | gflops | gb_s | pct_peak_flops | pct_peak_bw |
|---|---|---|---|---|---|
| H100 80GB HBM3 | 1.146 | 468.4 | 2,810 | 0.70% | 83.8% |
| RTX 4090 | 3.681 | 145.8 | 875 | 0.18% | 86.8% |
| RTX A4000 | 8.358 | 64.2 | 385 | 0.34% | 86.0% |

**Key Observation.** All three GPUs reach ~84–87% of their respective peak memory bandwidth, confirming the kernel is well-optimised and the test is genuinely memory-bound. The absolute bandwidth hierarchy (H100 ≫ 4090 > A4000) drives the 3.2× and 7.3× absolute throughput advantage of the H100 over the 4090 and A4000 respectively. `pct_peak_flops` is near-zero everywhere as expected — this kernel barely uses the FP32 ALUs.

**Mathematical Relevance.** SAXPY is the backbone of iterative Krylov solvers (CG, GMRES, BiCGSTAB), vector scaling in gradient descent, and update steps in explicit ODE integrators. A GPU with higher memory bandwidth processes larger vectors or more iterations per second in these methods.

---

### 4.2 SGEMM (Single-Precision General Matrix-Matrix Multiplication)

**Description.** Computes `C = A × B` for square matrices of size N×N. Two variants are tested: `naive` (direct nested-loop kernel with poor data reuse) and `cublas` (NVIDIA's highly optimised cuBLAS implementation using tensor cores and shared memory tiling).

**Results at N=4096 (cuBLAS variant):**

| GPU | mean_ms | gflops | gb_s | pct_peak_flops | pct_peak_bw |
|---|---|---|---|---|---|
| H100 80GB HBM3 | 2.637 | 52,110 | 208,466 | 77.9% | 6,219% |
| RTX 4090 | 2.612 | 52,627 | 210,534 | 63.7% | 20,884% |
| RTX A4000 | 14.067 | 9,770 | 39,085 | 51.0% | 8,723% |

*Note: `gb_s` > 100% of real peak reflects that the benchmark counts re-reads from cache as "bandwidth" — the matrix operands are reused extensively from L2/shared memory. The absolute GFLOP/s is the valid figure here.*

**Key Observation.** At large N, cuBLAS on the H100 and RTX 4090 deliver remarkably similar absolute GFLOP/s (~52 TFLOP/s) but the H100 reaches a higher percentage of its peak (77.9% vs 63.7%), reflecting Hopper's superior Tensor Core utilisation. The RTX A4000 trails badly at ~9.8 TFLOP/s because it has far fewer SMs and narrower memory. The naive kernel across all GPUs saturates at approximately 6,000–6,400 GFLOP/s for the H100 and ~4,800–5,100 for the others at large N — illustrating that without shared-memory tiling, all GPUs are severely bandwidth-throttled even for this compute-bound kernel.

**Mathematical Relevance.** SGEMM is the most important primitive in numerical linear algebra, deep learning (forward/backward passes are batched GEMMs), spectral methods, and covariance matrix computations in statistics. It is the "workhorse kernel" and its pct_peak_flops directly reflects how well you will utilise a GPU for dense AI/ML or scientific computing workloads. High pct_peak_flops here is the key figure.

---

### 4.3 Stencil (5-Point 1D Stencil)

**Description.** Applies a 5-point averaging stencil `out[i] = 0.2*(in[i-2]+in[i-1]+in[i]+in[i+1]+in[i+2])` across a 1D array. Two variants: `naive_global` (reads directly from global DRAM) and `shared_tiled` (caches a tile in shared memory to reduce redundant global reads).

**Results at N=67M (largest common size):**

| GPU | Variant | mean_ms | gflops | gb_s | pct_peak_bw |
|---|---|---|---|---|---|
| H100 | naive_global | 0.267 | 2,516 | 6,038 | 180% |
| H100 | shared_tiled | 0.397 | 1,691 | 4,059 | 121% |
| RTX 4090 | naive_global | 0.704 | 953 | 2,287 | 227% |
| RTX 4090 | shared_tiled | 0.605 | 1,110 | 2,663 | 264% |
| RTX A4000 | naive_global | 2.317 | 290 | 695 | 155% |
| RTX A4000 | shared_tiled | 1.653 | 406 | 974 | 217% |

*Note: `pct_peak_bw` > 100% again indicates L2 cache hits being counted in the bandwidth tally — some elements are reused from L2 rather than DRAM.*

**Key Observation.** An important and counter-intuitive result: `naive_global` is *faster* than `shared_tiled` on the H100 (0.267 ms vs 0.397 ms). The H100's extremely high HBM3 bandwidth (>3 TB/s) and large L2 cache make the naive approach memory-saturating without needing the overhead of explicit shared-memory management. On the RTX 4090 and A4000, the shared_tiled variant is faster (or competitive), as expected, because their memory bandwidth is lower and explicit reuse helps. This is a crucial insight for kernel design: **shared-memory tiling is not universally beneficial — on bandwidth-rich GPUs, the synchronization overhead can outweigh the reuse benefit.**

**Mathematical Relevance.** Stencil kernels are the numerical core of finite-difference methods for PDEs: heat equation, wave equation, Navier-Stokes (explicit schemes), electromagnetic simulation (FDTD), and weather forecasting (finite-difference atmosphere models). For these applications, `gb_s` is the governing performance metric and the stencil benchmark reveals exactly how well each GPU's memory subsystem handles repetitive, spatially-local access patterns.

---

### 4.4 Reduction (Parallel Sum via Warp Shuffle)

**Description.** Reduces a large float array to a single sum using CUDA warp-level shuffle intrinsics (`__shfl_down_sync`). This tests the GPU's ability to perform tree-reduction with near-zero memory traffic beyond the input read.

**Results at N=268M elements:**

| GPU | mean_ms | gflops | gb_s | pct_peak_bw |
|---|---|---|---|---|
| H100 80GB HBM3 | 0.349 | 769.9 | 3,080 | 91.9% |
| RTX 4090 | 1.128 | 238.1 | 952 | 94.5% |
| RTX A4000 | 3.052 | 88.0 | 352 | 78.5% |

**Key Observation.** The reduction benchmark achieves very high `pct_peak_bw` on H100 (~92%) and RTX 4090 (~95%), confirming that warp-shuffle reduction is among the best implementations possible for this algorithm — it nearly saturates the memory bus. The RTX A4000 lags at 78.5%, possibly due to fewer warp schedulers or lower occupancy at this problem size. The H100's 3× bandwidth advantage over the RTX 4090 translates to a ~3.2× faster reduction time.

**Mathematical Relevance.** Reduction is ubiquitous: computing vector norms (for convergence checks in iterative solvers), dot products (CG method update steps), summing energy in molecular dynamics, softmax denominators in neural networks, and pooling in deep learning. A fast reduction is the difference between a negligible overhead and a bottleneck in hybrid iterative algorithms. The near-peak bandwidth utilisation here means algorithmic improvements (rather than implementation improvements) are the only remaining lever.

---

### 4.5 Monte Carlo (Pi Estimation via cuRAND Philox PRNG)

**Description.** Generates N random (x,y) pairs using the cuRAND Philox counter-based PRNG and counts how many fall inside the unit circle. Primarily tests the throughput of pseudo-random number generation and the GPU's arithmetic pipelines under a mixture of random number generation, multiply, add, and branch (counter) workloads.

**Results at N=1B samples:**

| GPU | mean_ms | gflops | gb_s | pct_peak_flops |
|---|---|---|---|---|
| H100 80GB HBM3 | 2.968 | 2,695 | 16,170 | 4.03% |
| RTX 4090 | 2.222 | 3,601 | 21,605 | 4.36% |
| RTX A4000 | 9.143 | 875 | 5,250 | 4.57% |

**Key Observation.** The RTX 4090 is notably *faster* than the H100 (2.22 ms vs 2.97 ms), despite the H100's larger compute capability. Both reach similar `pct_peak_flops` (~4%), indicating neither is truly compute-bound — the PRNG generation and random number consumption are balanced across both compute and memory resources, and the RTX 4090's Ada Lovelace architecture may have superior throughput for this specific Philox PRNG kernel. The A4000 is 3–4× slower than both. `gb_s` here reflects the rate of consuming and discarding generated numbers, not DRAM bandwidth to a stored array.

**Mathematical Relevance.** Monte Carlo methods underpin financial risk modelling (option pricing, VaR), quantum chemistry (VMC/DMC), particle transport (radiation therapy planning), Bayesian inference via MCMC, and stochastic physics simulations. PRNG throughput in samples/second is the governing metric. The surprise of the RTX 4090 outperforming the H100 here illustrates that architectural generation and specific kernel design matter — the H100 is not uniformly faster at everything.

---

### 4.6 FFT 1D C2C (1D Complex-to-Complex Fast Fourier Transform)

**Description.** Performs a 1D complex-to-complex forward FFT of size N using NVIDIA cuFFT. The arithmetic intensity of FFT is O(N log N) FLOPs over O(N) data, making it moderately compute-bound at large sizes.

*Note: RTX A4000 has no FFT results in the checked-in CSV.*

**Results at N=16M complex elements:**

| GPU | mean_ms | gflops | gb_s | pct_peak_flops |
|---|---|---|---|---|
| H100 80GB HBM3 | 0.282 | 7,127 | 1,901 | 10.65% |
| RTX 4090 | 0.912 | 2,206 | 588 | 2.67% |

**Results at N=4M:**

| GPU | mean_ms | gflops | gb_s | pct_peak_flops |
|---|---|---|---|---|
| H100 80GB HBM3 | 0.073 | 6,344 | 1,846 | 9.48% |
| RTX 4090 | 0.078 | 5,950 | 1,731 | 7.21% |

**Key Observation.** At N=4M the two GPUs perform nearly identically in absolute time (0.073 ms vs 0.078 ms). At N=16M the H100 is ~3.2× faster. The H100 achieves a significantly higher `pct_peak_flops` (~10.65% vs 2.67% at 16M), suggesting the Hopper architecture's HBM3 bandwidth sustains cuFFT's radix butterfly passes at scale more effectively. The `pct_peak_bw` values are low relative to SAXPY — FFT is neither purely memory- nor compute-bound; it cycles between global memory passes and arithmetic, leading to moderate utilisation of both resources.

**Mathematical Relevance.** FFT is foundational to spectral solvers for fluid dynamics, signal processing, image reconstruction (MRI/CT), crystallography, convolution-based deep learning, and GPS/radar signal processing. Many climate and ocean models use spectral methods where FFT dominates runtime. The H100's FFT throughput advantage grows with problem size, meaning it is the appropriate choice for large-scale spectral workloads.

---

### 4.7 N-Body Barnes-Hut (Gravitational Tree Code)

**Description.** Implements the Barnes-Hut octree approximation for N-body gravitational simulation, complexity O(N log N). The tree-traversal is irregular, pointer-chasing, and has high branch divergence — exactly the pattern that GPUs handle least efficiently compared to regular compute.

*Note: Only H100 results are present in the CSV for this benchmark.*

**Results (H100 only):**

| N | mean_ms | gflops | gb_s | pct_peak_flops | pct_peak_bw |
|---|---|---|---|---|---|
| 2,048 | 0.248 | 7.27 | 0.76 | 0.011% | 0.023% |
| 32,768 | 0.936 | 42.0 | 3.22 | 0.063% | 0.096% |
| 262,144 | 8.048 | 46.9 | 3.00 | 0.070% | 0.089% |

**Key Observation.** The extraordinarily low `pct_peak_flops` (~0.01–0.07%) and `pct_peak_bw` (~0.02–0.09%) confirm that the Barnes-Hut tree traversal is severely bottlenecked by irregular memory access patterns (cache-busting pointer chasing), warp divergence (threads in the same warp traverse different tree paths), and the overhead of octree construction. Even on the H100 with its enormous bandwidth and cache, the hardware is almost entirely idle — the bottleneck is memory latency, not throughput. This is consistent with published literature on tree-code GPU performance.

**Mathematical Relevance.** N-body simulations are used in astrophysics (galaxy formation), molecular dynamics (long-range electrostatics via Barnes-Hut or FMM), plasma physics, and vortex methods in fluid dynamics. The data shows that irregular tree algorithms stress GPUs very differently from structured kernels — GPU-optimised tree codes require careful attention to tree layout in memory (BVH, Morton codes), warp-level batching of traversals, and often hybrid CPU-GPU decompositions. The H100's massive L2 cache partially mitigates the issue, but all three metrics remain extremely low.

---

## 5. Cross-GPU Comparative Analysis

### 5.1 Compute Capability and Architecture Lineage

| GPU | Architecture | SM Count | FP32 Peak | Mem Type | Peak BW | VRAM |
|---|---|---|---|---|---|---|
| H100 SXM 80GB | Hopper (sm_90) | 132 SMs | ~66.9 TFLOP/s | HBM3 | ~3,350 GB/s | 80 GB |
| RTX 4090 | Ada Lovelace (sm_89) | 128 SMs | ~82.6 TFLOP/s | GDDR6X | ~1,008 GB/s | 24 GB |
| RTX A4000 | Ampere (sm_86) | 48 SMs | ~19.2 TFLOP/s | GDDR6 | ~448 GB/s | 16 GB |

*Note: The RTX 4090 actually has a higher theoretical FP32 peak than the H100 (using standard FP32 tensor-free throughput). However, the H100's advantage in Tensor Core FP16/BF16/FP8 workloads and HBM3 bandwidth makes it the dominant scientific computing platform.*

### 5.2 Benchmark-by-Benchmark Winner Summary

| Benchmark | Fastest GPU | Notes |
|---|---|---|
| SAXPY (memory BW) | **H100** | 3.2× faster than 4090; 7.3× faster than A4000 |
| SGEMM cuBLAS large N | **H100 ≈ RTX 4090** | Both ~52 TFLOP/s; H100 wins on pct_peak |
| Stencil naive | **H100** | 2.6× faster than 4090; 8.7× faster than A4000 |
| Stencil shared_tiled | **RTX 4090 > H100** | 4090 benefits more from tiling; H100's overhead larger |
| Reduction | **H100** | ~3.2× faster than 4090; ~8.7× faster than A4000 |
| Monte Carlo | **RTX 4090** | 25% faster than H100; A4000 ~4× behind |
| FFT (large N) | **H100** | 3.2× faster than 4090 at 16M |
| FFT (small N) | **H100 ≈ RTX 4090** | ~8% difference at 4M |
| N-Body Barnes-Hut | **H100 only tested** | All GPUs expected to be similarly bottlenecked |

### 5.3 Roofline and Bottleneck Analysis

The roofline model distinguishes compute-bound from memory-bound regimes by comparing a kernel's arithmetic intensity (FLOP/byte) against the machine's peak FLOP:byte ratio (the "ridge point").

Approximate ridge points:
- **H100:** 66,900 GFLOP/s ÷ 3,350 GB/s ≈ **20 FLOP/byte**
- **RTX 4090:** 82,600 GFLOP/s ÷ 1,008 GB/s ≈ **82 FLOP/byte**
- **RTX A4000:** 19,200 GFLOP/s ÷ 448 GB/s ≈ **43 FLOP/byte**

| Benchmark | Arithmetic Intensity | H100 Regime | 4090 Regime | A4000 Regime |
|---|---|---|---|---|
| SAXPY | ~0.33 FLOP/byte | Memory-bound | Memory-bound | Memory-bound |
| SGEMM (N=4096) | ~2,730 FLOP/byte | Compute-bound | Compute-bound | Compute-bound |
| Stencil | ~5 FLOP/byte | Memory-bound | Memory-bound | Memory-bound |
| Reduction | ~1 FLOP/byte | Memory-bound | Memory-bound | Memory-bound |
| Monte Carlo | ~50 FLOP/byte | Memory-bound | Compute-bound | Compute-bound |
| FFT | ~35 FLOP/byte | Memory-bound | Compute-bound | Memory-bound |
| N-Body BH | irregular | Latency-bound | Latency-bound | Latency-bound |

**Critical Insight.** The H100's much lower ridge point (20 FLOP/byte vs 82 for RTX 4090) means that *many more workloads are in the compute-bound regime* on the H100 — kernels that are memory-bound on a 4090 become compute-bound on an H100. This is why the H100 delivers a far larger absolute speedup on bandwidth-hungry workloads (SAXPY, reduction, stencil) than its raw FP32 compute ratio would predict.

### 5.4 Speedup Tables (H100 as baseline)

**Speedup of H100 over RTX 4090 (mean_ms, lower is better → speedup = 4090_ms / H100_ms):**

| Benchmark | Problem Size | 4090 ms | H100 ms | Speedup |
|---|---|---|---|---|
| SAXPY | 268M | 3.681 | 1.146 | **3.21×** |
| SGEMM cuBLAS | 4096 | 2.612 | 2.637 | **0.99×** (tie) |
| Stencil naive | 67M | 0.704 | 0.267 | **2.64×** |
| Stencil shared | 67M | 0.605 | 0.397 | **1.52×** |
| Reduction | 268M | 1.128 | 0.349 | **3.23×** |
| Monte Carlo | 1B | 2.222 | 2.968 | **0.75×** (4090 wins) |
| FFT | 16M | 0.912 | 0.282 | **3.23×** |

**Speedup of H100 over RTX A4000:**

| Benchmark | Problem Size | A4000 ms | H100 ms | Speedup |
|---|---|---|---|---|
| SAXPY | 268M | 8.358 | 1.146 | **7.29×** |
| SGEMM cuBLAS | 4096 | 14.067 | 2.637 | **5.33×** |
| Stencil naive | 67M | 2.317 | 0.267 | **8.67×** |
| Stencil shared | 67M | 1.653 | 0.397 | **4.16×** |
| Reduction | 268M | 3.052 | 0.349 | **8.75×** |
| Monte Carlo | 1B | 9.143 | 2.968 | **3.08×** |

### 5.5 Efficiency vs. Cost Analysis

Using the RunPod compute prices captured in the READMEs:

| GPU | Price/hr | SGEMM Peak GFLOP/s (cuBLAS, N=4096) | GFLOP/s per Dollar |
|---|---|---|---|
| H100 80GB HBM3 | $3.00 | 52,110 | **17,370 GFLOP/$/hr** |
| RTX 4090 | $0.70 | 52,627 | **75,181 GFLOP/$/hr** |
| RTX A4000 | $0.25 | 9,770 | **39,080 GFLOP/$/hr** |

| GPU | Price/hr | SAXPY Peak BW (GB/s, 268M) | GB/s per Dollar |
|---|---|---|---|
| H100 80GB HBM3 | $3.00 | 2,810 | **937 (GB/s)/$/hr** |
| RTX 4090 | $0.70 | 875 | **1,250 (GB/s)/$/hr** |
| RTX A4000 | $0.25 | 385 | **1,540 (GB/s)/$/hr** |

**Observation.** The H100 is the *least cost-efficient* GPU by raw FLOP/$ or BW/$, but this metric misses the point for production workloads where throughput capacity is the binding constraint (time-to-solution, not $/FLOP). For a fixed wall-clock time budget, the H100 handles 3–9× more data or computation than the alternatives.

---

## 6. Trade-Off Analysis

### 6.1 H100 80GB HBM3

**Strengths:**
- Dominant on all bandwidth-bound workloads — SAXPY, reduction, stencil: 3–9× faster than the alternatives
- Best-in-class cuBLAS GEMM efficiency (77.9% of theoretical FP32 peak, the highest pct_peak_flops recorded)
- 80 GB VRAM enables problem sizes impossible on the other cards (large-scale CFD meshes, full-model deep learning, genomics)
- HBM3 architecture delivers very tight run-to-run variance (lowest stddev_ms) on structured workloads
- Hopper sm_90 introduces wgmma/TMA instructions for even faster tensor operations not directly benchmarked here
- Best for: large-scale scientific computing, AI training, dense linear algebra, spectral methods, large-N Monte Carlo

**Weaknesses:**
- Monte Carlo (PRNG-heavy) was *slower* than the RTX 4090 — Hopper architecture appears to be suboptimal for Philox PRNG generation in this configuration
- 12× more expensive than RTX A4000, 4.3× more expensive than RTX 4090 on RunPod
- Cost-per-GFLOP is the worst of the three
- Excessive for small problem sizes where latency dominates — all three GPUs perform similarly at N <1M elements

**Best suited for:** Massively parallel structured math — PDE solvers, BLAS-3, cuBLAS-dependent deep learning, large spectral transforms, large-N reductions and dot products.

---

### 6.2 NVIDIA GeForce RTX 4090

**Strengths:**
- Best-in-class for Monte Carlo PRNG workloads — 25% faster than the H100 at 1B samples
- Nearly matches the H100 on cuBLAS SGEMM absolute GFLOP/s (~52 TFLOP/s) at a fraction of the cost
- Excellent `pct_peak_flops` for large SGEMM (63.7%) — second-best efficiency of all three cards
- High GDDR6X bandwidth (1,008 GB/s peak) with best `pct_peak_bw` for reduction (94.5%)
- Shared-memory stencil benefits more here than on the H100 — good signal for optimised finite-difference codes
- 4.3× cheaper than H100 per hour, making it the best cost/TFLOP card for GEMM workloads

**Weaknesses:**
- Only 24 GB VRAM — severely limits problem size for training large models or running large meshes
- Memory bandwidth (1,008 GB/s) is only 30% of the H100's — devastating for bandwidth-heavy workloads
- GDDR6X has higher latency than HBM3, making it worse for irregular access patterns
- No ECC memory by default (consumer card) — may introduce silent errors in long-running scientific jobs

**Best suited for:** Moderately-sized AI inference, GEMM-heavy workloads (deep learning training with mixed precision), Monte Carlo financial or physics simulations, and any workload where 24 GB VRAM is sufficient and cost sensitivity matters.

---

### 6.3 NVIDIA RTX A4000

**Strengths:**
- Extremely cost-effective at $0.25/hr — best GB/s