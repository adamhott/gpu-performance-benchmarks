# NVIDIA GeForce RTX 4090 — consolidated `cuda-bench` results

_Generated 2026-05-03 from `results.csv`._

> **Provenance:** All figures are derived from `results.csv` in this directory (typically copied from `cuda-bench/results/results.csv` after a run). Regenerate this file with `python3 format_tables.py` whenever `results.csv` changes.

## Run context

| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 |
| Compute capability | 8.9 |
| Suite | `cuda-bench` (`scripts/run_all.sh`) |
| Notes | RunPod-style container (GPU clock lock often unavailable). |

## Summary tables

### saxpy

| problem_size | variant | mean_ms | stddev_ms | gflops | gb_s | % peak F | % peak BW |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,048,576 | grid_stride | 0.0061 | 0.0004 | 345.9460 | 2075.6800 | 0.4189 | 205.9010 |
| 4,194,304 | grid_stride | 0.0143 | 0.0004 | 586.7670 | 3520.6000 | 0.7106 | 349.2330 |
| 16,777,216 | grid_stride | 0.2268 | 0.0257 | 147.9630 | 887.7790 | 0.1792 | 88.0649 |
| 67,108,864 | grid_stride | 0.9037 | 0.0052 | 148.5220 | 891.1340 | 0.1799 | 88.3977 |
| 268,435,456 | grid_stride | 3.6812 | 0.0107 | 145.8410 | 875.0470 | 0.1766 | 86.8020 |

### sgemm

| problem_size | variant | mean_ms | stddev_ms | gflops | gb_s | % peak F | % peak BW |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | naive | 0.0122 | 0.0003 | 2753.9000 | 11037.1000 | 3.3350 | 1094.8500 |
| 256 | cublas | 0.0102 | 0.0005 | 3289.5500 | 13183.9000 | 3.9837 | 1307.8000 |
| 512 | naive | 0.0637 | 0.0051 | 4216.4400 | 16882.2000 | 5.1062 | 1674.6700 |
| 512 | cublas | 0.0162 | 0.0006 | 16582.2000 | 66393.6000 | 20.0813 | 6586.0400 |
| 1,024 | naive | 0.4892 | 0.0009 | 4389.3600 | 17566.0000 | 5.3156 | 1742.4900 |
| 1,024 | cublas | 0.0498 | 0.0023 | 43154.6000 | 172703.0000 | 52.2609 | 17131.6000 |
| 2,048 | naive | 3.7694 | 0.1160 | 4557.7800 | 18235.6000 | 5.5195 | 1808.9100 |
| 2,048 | cublas | 0.3050 | 0.0017 | 56320.6000 | 225338.0000 | 68.2051 | 22352.8000 |
| 4,096 | naive | 28.2772 | 0.0153 | 4860.4200 | 19444.0000 | 5.8860 | 1928.7900 |
| 4,096 | cublas | 2.6116 | 0.0718 | 52627.2000 | 210534.0000 | 63.7323 | 20884.4000 |

### stencil

| problem_size | variant | mean_ms | stddev_ms | gflops | gb_s | % peak F | % peak BW |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,048,576 | naive_global | 0.0071 | 0.0008 | 1486.4800 | 3567.5600 | 1.8001 | 353.8900 |
| 1,048,576 | shared_tiled | 0.0070 | 0.0004 | 1506.7100 | 3616.1100 | 1.8247 | 358.7070 |
| 4,194,304 | naive_global | 0.0155 | 0.0003 | 2704.8600 | 6491.6600 | 3.2756 | 643.9520 |
| 4,194,304 | shared_tiled | 0.0179 | 0.0005 | 2346.9400 | 5632.6600 | 2.8422 | 558.7420 |
| 16,777,216 | naive_global | 0.1554 | 0.0011 | 1079.5900 | 2591.0200 | 1.3074 | 257.0210 |
| 16,777,216 | shared_tiled | 0.1518 | 0.0005 | 1105.4400 | 2653.0600 | 1.3387 | 263.1750 |
| 37,748,736 | naive_global | 0.3792 | 0.0045 | 995.3690 | 2388.8900 | 1.2054 | 236.9700 |
| 37,748,736 | shared_tiled | 0.3404 | 0.0013 | 1109.0500 | 2661.7300 | 1.3431 | 264.0350 |
| 67,108,864 | naive_global | 0.7043 | 0.0063 | 952.8970 | 2286.9500 | 1.1540 | 226.8590 |
| 67,108,864 | shared_tiled | 0.6048 | 0.0006 | 1109.6100 | 2663.0600 | 1.3438 | 264.1670 |

### reduction

| problem_size | variant | mean_ms | stddev_ms | gflops | gb_s | % peak F | % peak BW |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,048,576 | warp_shuffle | 0.0069 | 0.0005 | 152.7360 | 610.9440 | 0.1850 | 60.6038 |
| 4,194,304 | warp_shuffle | 0.0104 | 0.0008 | 403.8450 | 1615.3800 | 0.4891 | 160.2410 |
| 16,777,216 | warp_shuffle | 0.0209 | 0.0005 | 802.0560 | 3208.2200 | 0.9713 | 318.2460 |
| 67,108,864 | warp_shuffle | 0.2863 | 0.0005 | 234.3640 | 937.4570 | 0.2838 | 92.9928 |
| 268,435,456 | warp_shuffle | 1.1275 | 0.0019 | 238.0750 | 952.2980 | 0.2883 | 94.4650 |

### montecarlo

| problem_size | variant | mean_ms | stddev_ms | gflops | gb_s | % peak F | % peak BW |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000,000 | curand_philox | 0.0101 | 0.0045 | 790.9390 | 4745.6300 | 0.9578 | 470.7520 |
| 10,000,000 | curand_philox | 0.0292 | 0.0005 | 2742.1900 | 16453.1000 | 3.3208 | 1632.1000 |
| 100,000,000 | curand_philox | 0.2275 | 0.0007 | 3516.7300 | 21100.4000 | 4.2588 | 2093.0900 |
| 500,000,000 | curand_philox | 1.1107 | 0.0026 | 3601.2500 | 21607.5000 | 4.3612 | 2143.4000 |
| 1,000,000,000 | curand_philox | 2.2217 | 0.0089 | 3600.9000 | 21605.4000 | 4.3607 | 2143.1900 |

### fft_1d_c2c

| problem_size | variant | mean_ms | stddev_ms | gflops | gb_s | % peak F | % peak BW |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 262,144 | cufft_forward | 0.0093 | 0.0003 | 2535.3500 | 901.4580 | 3.0703 | 89.4218 |
| 1,048,576 | cufft_forward | 0.0187 | 0.0004 | 5598.6900 | 1791.5800 | 6.7801 | 177.7190 |
| 4,194,304 | cufft_forward | 0.0775 | 0.0013 | 5949.8000 | 1730.8500 | 7.2053 | 171.6950 |
| 16,777,216 | cufft_forward | 0.9125 | 0.0356 | 2206.3800 | 588.3670 | 2.6720 | 58.3642 |
| 67,108,864 | cufft_forward | 3.9491 | 0.0175 | 2209.1700 | 543.7970 | 2.6753 | 53.9429 |

## Raw CSV

```csv
benchmark,gpu_name,compute_capability,problem_size,variant,mean_ms,stddev_ms,gflops,gb_s,pct_peak_flops,pct_peak_bw
saxpy,NVIDIA GeForce RTX 4090,8.9,1048576,grid_stride,0.00606208,0.000363488,345.946,2075.68,0.418946,205.901
saxpy,NVIDIA GeForce RTX 4090,8.9,4194304,grid_stride,0.0142963,0.000372988,586.767,3520.6,0.710584,349.233
saxpy,NVIDIA GeForce RTX 4090,8.9,16777216,grid_stride,0.226776,0.0256665,147.963,887.779,0.179186,88.0649
saxpy,NVIDIA GeForce RTX 4090,8.9,67108864,grid_stride,0.903687,0.00515693,148.522,891.134,0.179863,88.3977
saxpy,NVIDIA GeForce RTX 4090,8.9,268435456,grid_stride,3.6812,0.0107388,145.841,875.047,0.176616,86.802
sgemm,NVIDIA GeForce RTX 4090,8.9,256,naive,0.0121843,0.000334968,2753.9,11037.1,3.33502,1094.85
sgemm,NVIDIA GeForce RTX 4090,8.9,256,cublas,0.0102003,0.000490894,3289.55,13183.9,3.98369,1307.8
sgemm,NVIDIA GeForce RTX 4090,8.9,512,naive,0.063664,0.00508234,4216.44,16882.2,5.10617,1674.67
sgemm,NVIDIA GeForce RTX 4090,8.9,512,cublas,0.0161882,0.000605491,16582.2,66393.6,20.0813,6586.04
sgemm,NVIDIA GeForce RTX 4090,8.9,1024,naive,0.489248,0.000905976,4389.36,17566,5.31558,1742.49
sgemm,NVIDIA GeForce RTX 4090,8.9,1024,cublas,0.0497626,0.00232938,43154.6,172703,52.2609,17131.6
sgemm,NVIDIA GeForce RTX 4090,8.9,2048,naive,3.76935,0.116015,4557.78,18235.6,5.51953,1808.91
sgemm,NVIDIA GeForce RTX 4090,8.9,2048,cublas,0.305037,0.00171335,56320.6,225338,68.2051,22352.8
sgemm,NVIDIA GeForce RTX 4090,8.9,4096,naive,28.2772,0.0153476,4860.42,19444,5.88604,1928.79
sgemm,NVIDIA GeForce RTX 4090,8.9,4096,cublas,2.61156,0.0718193,52627.2,210534,63.7323,20884.4
stencil,NVIDIA GeForce RTX 4090,8.9,1048576,naive_global,0.00705408,0.000800487,1486.48,3567.56,1.80015,353.89
stencil,NVIDIA GeForce RTX 4090,8.9,1048576,shared_tiled,0.00695936,0.000395065,1506.71,3616.11,1.82465,358.707
stencil,NVIDIA GeForce RTX 4090,8.9,4194304,naive_global,0.0155066,0.000319418,2704.86,6491.66,3.27562,643.952
stencil,NVIDIA GeForce RTX 4090,8.9,4194304,shared_tiled,0.0178714,0.000478025,2346.94,5632.66,2.84218,558.742
stencil,NVIDIA GeForce RTX 4090,8.9,16777216,naive_global,0.155404,0.00109049,1079.59,2591.02,1.3074,257.021
stencil,NVIDIA GeForce RTX 4090,8.9,16777216,shared_tiled,0.15177,0.000540821,1105.44,2653.06,1.3387,263.175
stencil,NVIDIA GeForce RTX 4090,8.9,37748736,naive_global,0.379244,0.00446195,995.369,2388.89,1.20541,236.97
stencil,NVIDIA GeForce RTX 4090,8.9,37748736,shared_tiled,0.340369,0.00128603,1109.05,2661.73,1.34308,264.035
stencil,NVIDIA GeForce RTX 4090,8.9,67108864,naive_global,0.704262,0.00625636,952.897,2286.95,1.15397,226.859
stencil,NVIDIA GeForce RTX 4090,8.9,67108864,shared_tiled,0.604799,0.000558121,1109.61,2663.06,1.34375,264.167
reduction,NVIDIA GeForce RTX 4090,8.9,1048576,warp_shuffle,0.00686528,0.000548084,152.736,610.944,0.184966,60.6038
reduction,NVIDIA GeForce RTX 4090,8.9,4194304,warp_shuffle,0.0103859,0.000761418,403.845,1615.38,0.489062,160.241
reduction,NVIDIA GeForce RTX 4090,8.9,16777216,warp_shuffle,0.0209178,0.00053735,802.056,3208.22,0.971302,318.246
reduction,NVIDIA GeForce RTX 4090,8.9,67108864,warp_shuffle,0.286344,0.000474586,234.364,937.457,0.283819,92.9928
reduction,NVIDIA GeForce RTX 4090,8.9,268435456,warp_shuffle,1.12753,0.00190889,238.075,952.298,0.288312,94.465
montecarlo,NVIDIA GeForce RTX 4090,8.9,1000000,curand_philox,0.0101146,0.00448297,790.939,4745.63,0.957839,470.752
montecarlo,NVIDIA GeForce RTX 4090,8.9,10000000,curand_philox,0.0291738,0.000468363,2742.19,16453.1,3.32083,1632.1
montecarlo,NVIDIA GeForce RTX 4090,8.9,100000000,curand_philox,0.227484,0.000661125,3516.73,21100.4,4.25881,2093.09
montecarlo,NVIDIA GeForce RTX 4090,8.9,500000000,curand_philox,1.11072,0.00263774,3601.25,21607.5,4.36117,2143.4
montecarlo,NVIDIA GeForce RTX 4090,8.9,1000000000,curand_philox,2.22167,0.00890908,3600.9,21605.4,4.36074,2143.19
fft_1d_c2c,NVIDIA GeForce RTX 4090,8.9,262144,cufft_forward,0.0093056,0.000298581,2535.35,901.458,3.07035,89.4218
fft_1d_c2c,NVIDIA GeForce RTX 4090,8.9,1048576,cufft_forward,0.018729,0.000419777,5598.69,1791.58,6.78009,177.719
fft_1d_c2c,NVIDIA GeForce RTX 4090,8.9,4194304,cufft_forward,0.0775443,0.00130853,5949.8,1730.85,7.2053,171.695
fft_1d_c2c,NVIDIA GeForce RTX 4090,8.9,16777216,cufft_forward,0.912476,0.0355828,2206.38,588.367,2.67196,58.3642
fft_1d_c2c,NVIDIA GeForce RTX 4090,8.9,67108864,cufft_forward,3.94906,0.0174681,2209.17,543.797,2.67534,53.9429
```
