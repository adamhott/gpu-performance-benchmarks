# 05 — Monte Carlo estimate of $\pi$

## Summary

Benchmark **`05_montecarlo`** draws **uniform** random points $(X,Y)$ in the unit square $[0,1)^2$ using **cuRAND** Philox in the kernel. For each sample it tests membership in the **quarter disk** $x^2+y^2 \le 1$. After counting hits $H$ over $S$ trials, it forms $\hat{\pi} = 4H/S$ and checks statistical consistency. Timing wraps only the GPU kernel launches that refill the hit counter.

## Why we run it

This pattern isolates **random number generation throughput**, **branchy** inner loops, and **atomic** accumulation—common in **particle methods**, **MCMC**, and **risk** simulations. It complements deterministic linear-algebra kernels by exercising **curand** and **warp/block reductions** before a global atomic add.

## Math and verification

Let $(X_\ell,Y_\ell)$ for $\ell=1,\ldots,S$ be i.i.d. $\mathrm{Uniform}(0,1)^2$. Define the indicator

$$
H = \sum_{\ell=1}^{S} \mathbf{1}\{ X_\ell^2 + Y_\ell^2 \le 1 \}.
$$

Then

$$
\hat{\pi} = \frac{4H}{S}
$$

is an unbiased estimator of $\pi$ because the quarter circle of radius 1 has area $\pi/4$ inside the unit square.

**Throughput accounting in this repo:** the CSV uses a fixed **$8S$** “FLOP” model and a **byte** model tied to random variates and arithmetic (see `05_montecarlo.cu`) for rough **GFLOP/s** / **GB/s** reporting—not a literal count of every micro-op.

Verification uses a **confidence-style** tolerance combining a fixed floor and a $1/\sqrt{S}$ scaling, appropriate for Monte Carlo noise.
