# 06 — 1D complex-to-complex FFT (cuFFT)

## Summary

Benchmark **`06_fft`** builds a length-$n$ **complex** vector $z_j \in \mathbb{C}$ (as `cufftComplex`), runs a **forward** **C2C** FFT in place with **cuFFT**, and for correctness runs **inverse + scaling** so the data should match the original within a relative tolerance. **Timing** in the CSV is **forward-only**: each timed iteration copies a fixed seed buffer device-to-device, then records CUDA events around **`cufftExecC2C(..., CUFFT_FORWARD)`** only.

## Why we run it

Fast Fourier transforms underpin **spectral PDE methods**, **convolutions via FFT**, **signal processing**, and many **quantum / many-body** workflows. Library FFTs are among the most tuned codes on GPUs; this benchmark measures **cuFFT** forward throughput at several powers-of-two lengths.

## Math and verification

Let $\omega_n = e^{-2\pi i/n}$ be a primitive $n$-th root of unity. The **forward** discrete Fourier transform of $z_0,\ldots,z_{n-1}$ is

$$
\hat{z}_k = \sum_{j=0}^{n-1} z_j \omega_n^{jk}, \qquad k = 0,\ldots,n-1.
$$

The **inverse** transform (up to normalization conventions) satisfies

$$
z_j = \frac{1}{n} \sum_{k=0}^{n-1} \hat{z}_k \omega_n^{-jk}.
$$

The benchmark applies **`1/n`** scaling in a small CUDA kernel after **`CUFFT_INVERSE`** so the round-trip matches the original $z_j$ within **`kFftRelTol`**.

**Complexity / accounting:** the harness uses a textbook-style **$5 n \log_2 n$** model for complex FFT floating work when reporting **GFLOP/s**, and an approximate **byte traffic** model for **GB/s** (see `fft_flops_estimate` / `fft_bytes_estimate` in `06_fft.cu`).
