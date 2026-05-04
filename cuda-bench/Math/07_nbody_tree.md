# 07 — Barnes–Hut \(N\)-body accelerations (softened gravity)

## Summary

Benchmark **`07_nbody_tree`** builds a **Barnes–Hut octree** on the **CPU** from \(N\) point masses with positions \(\mathbf{r}_j \in \mathbb{R}^3\) and masses \(m_j\). A CUDA kernel then walks that tree for each body \(i\) and accumulates a **softened gravitational acceleration** \(\mathbf{a}_i\), using an opening angle \(\theta\) to decide when to treat a cell as a point mass at its center of mass. For small \(N\), results are checked against a **direct** \(O(N^2)\) sum on the CPU. CSV timing reflects **only the GPU traversal kernel** (not the CPU tree build).

## Why we run it

Cosmology and stellar dynamics codes often use **hierarchical** \(N\)-body or hybrid **tree + PM** methods. This benchmark captures **irregular control flow** (stack-based tree walk), **pointer-chasing** on a GPU-unfriendly structure laid flat in arrays, and **scientific kernels** beyond dense linear algebra—while still grounding forces in a classical **inverse-square** law with softening.

## Math and verification

Let \(\mathbf{r}_i\) be the position of body \(i\) and \(\mathbf{r}_j\) another body’s position. Write \(\mathbf{d}_{ij} = \mathbf{r}_j - \mathbf{r}_i\). With a softening parameter \(\varepsilon > 0\) (the code uses \(\varepsilon^2 = \texttt{eps2}\)), define the squared softened distance

$$
\rho_{ij}^2 \;=\; \|\mathbf{d}_{ij}\|^2 + \varepsilon^2.
$$

With gravitational constant \(G\) (here **`kG = 1`**), the contribution of mass \(m_j\) to the acceleration of body \(i\) is the softened inverse-square law

$$
\mathbf{a}_{i} \;\mathrel{+}= \; G\, m_j\, \frac{\mathbf{d}_{ij}}{\rho_{ij}^{3}}.
$$

The **direct** reference sums the above over all \(j \ne i\).

The **Barnes–Hut** kernel approximates distant groups of bodies by their total mass \(M\) and center-of-mass \(\mathbf{R}\) when the cell’s spatial size \(s\) and distance \(d\) satisfy an **opening criterion** (implemented as comparing \(s/d\) to \(\theta\)); otherwise the cell is opened and its children are pushed on a stack. **Leaves** apply the same pairwise softened force using the actual body in the leaf (skipping self-interaction).

Verification (for \(N \le 4096\)) compares BH accelerations to the direct sum with a **relative** tolerance on the order of **12%** componentwise (`kAccelRelTol`) because the tree is an approximation unless \(\theta\) is small.

**Reporting model:** the CSV uses an **\(80\,N \log_2 N\)** FLOP estimate and a simple per-body byte model for **GFLOP/s** / **GB/s** (see `bh_flops_estimate` in `07_nbody_tree.cu`); treat these as **rough** throughput indicators, not exact operation counts of the walk.
