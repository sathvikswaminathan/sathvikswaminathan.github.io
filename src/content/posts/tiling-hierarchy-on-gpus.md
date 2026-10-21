---
author: Sathvik Swaminathan
pubDatetime: 2026-10-20T05:00:00Z
title: Tiling Hierarchy on GPUs
featured: true
tags:
  - gpu
  - performance
  - memory-hierarchy
  - gemm
description: GPUs are equipped with a tremendous amount of compute and a tiered memory architecture. How data is placed and accessed across these tiers determines how much of that compute is actually used.
---

Matrix multiplication, the workload that dominates GPU time in deep learning, will run much slower than what the hardware is actually capable of when written in the straightforward way. This is not because the GPU lacks compute cores; rather, the compute cores remain idle while waiting for data to arrive.

GPUs have a tiered memory hierarchy, with each level being larger but slower than the previous one: register file, shared memory/L1 cache, L2 cache, and global memory (HBM). The compute cores remain idle if data is fetched from slower memory tiers too often, or if memory requests stall due to inefficient access patterns such as shared-memory bank conflicts or uncoalesced global loads. The fix is to organize data across these tiers such that data loaded into a faster tier is reused as many times as possible before being evicted, and to group accesses so that they are conflict-free. This way, the compute cores are fed from the fastest tier with minimal memory stalls.

Throughout this post we will consider an FP32 matrix multiply `C = A × B`, where `A` is `M×K`, `B` is `K×N`, and `C` is `M×N`, with `M = N = K = 4096`.

## Table of contents

## Where the FLOPs come from

Every SM is divided into 4 subpartitions known as SMSPs. Each SMSP has a warp scheduler which can issue one warp-wide instruction every cycle. A warp comprises 32 threads, and every thread's position in a warp is referred to as its lane (`0–31`). Each SMSP is also equipped with CUDA cores. On an H100, each SMSP hosts 32 FP32 CUDA cores; the A100 SMSP hosts 16 FP32 CUDA cores. The core count per SMSP is what decides how long a warp occupies the cores. Since a warp issues 32 FMAs per instruction, the H100's 32 FP32 cores retire a warp-wide FMA in a single cycle, matching the scheduler's issue rate. The A100's 16 cores need two cycles per warp, so its scheduler can only issue an FP32 FMA every other cycle and has to find other work, or run another warp, to fill the gap.

Each FP32 CUDA core can execute the FMA (fused multiply-add) instruction:

```text
c = a * b + c
```

Essentially, every FMA results in `2 FLOPs` — one multiplication and one addition. The CUDA cores are pipelined, allowing every SMSP to perform `2 FLOPs` per cycle per core, resulting in a peak throughput of `2 FLOPs × 32 = 64 FLOPs` per cycle. The whole SM can sustain a peak throughput of `4 × 64 FLOPs = 256 FLOPs`, or `128 FMAs`, per cycle.

## Arithmetic Intensity of Matrix Multiplication

In the FMA operation `c = a * b + c`, the accumulator `c` is already present in the register file. The threads only need to load their `a` and `b` values into the register file to execute the FMA.

Arithmetic intensity (AI) is the ratio of computation performed to data moved from memory.

$$
\text{Arithmetic Intensity} = \frac{\text{Number of arithmetic operations}}{\text{Bytes transferred from memory}}
$$

For a single FMA, the AI is `2 FLOPs / 64 bytes` = `0.03125 FLOPs per byte`, or 2 loads per FMA. A higher AI helps keep the cores busy for longer.

Now, let's compute the AI for matrix multiplication.

There are two ways of computing `C = A × B`: 

### Inner Product

$$
C_{rc} = \sum_{k=0}^{K-1} A_{rk} B_{kc}
\qquad (0 \le r < M,\; 0 \le c < N)
$$

Computing each output of `C` requires `K` multiplies and `K` adds, resulting in `K` FMAs. So the whole GEMM would require `M × N × K` FMAs, or `2MNK` FLOPs.

To compute each output, the algorithm walks along the `k` axis of both `A` and `B`, loading `K` values of `A` and `K` values of `B`. Computing the next output would require loading the next set of `A` and `B` values. Therefore, for `K` FMAs, the algorithm is loading `2K` FP32 values, resulting in an AI of `0.5` FMAs per load.

### Outer Product

$$
C = \sum_{k=0}^{K-1} A_{:,k} \, B_{k,:}
\qquad (A_{:,k} \in \mathbb{R}^{M},\; B_{k,:} \in \mathbb{R}^{N})
$$

Here we swap the order of traversal and compute a partial product for all the output elements of `C` in steps of `k`. In every `k` step, the `k`-th column of `A` and the `k`-th row of `B` are fetched to compute a partial product for all the output elements. After `K` steps, all the output elements will be fully computed.

Therefore, for `M × N` partial products, this algorithm performs `M × N` FMAs while loading only `M` values of `A` and `N` values of `B`. This results in an AI of

$$
\frac{MN}{M+N}
$$

Maximizing this requires `M = N`, resulting in an AI of `0.5 × M` FMAs for every load. This is why square matrices are typically chosen — they have a higher AI.

![Inner product loads a row of A and a column of B to produce one output element, while the outer product loads a column of A and a row of B to update every output element of C](./inner-vs-outer-product.png)

Since both `M` and `N` are typically large, the outer product has a far higher AI than the inner product due to higher data reuse.

## Register tiling

How much AI does it take to saturate the CUDA cores?

As we will see in the next section, CUDA threads load data from GMEM into SMEM, so every load into the register file is served by shared memory.

On an H100, an SM sustains a peak throughput of `128` FMAs per cycle, and each FMA needs `2` FP32 operands. For a lane to issue an FMA in the next cycle, its operands must already be sitting in registers. But the SM's shared-memory pipe only delivers about `128 bytes` (`32` FP32 values) per cycle into the register file. If every FMA used unique operands, the SM would need `128 × 2 = 256` FP32 values every cycle. Those `32` loaded values must therefore be shared across all `128` FMAs, which pins the required arithmetic intensity at `128 FMAs / 32 loads = 4 FMAs` per load.

![Without reuse, 32 loaded FP32 values feed only 16 FMAs and 112 of the SM's 128 lanes sit idle; with each loaded value feeding 4 FMAs, all 128 lanes stay busy](./required-arithmetic-intensity.png)

Now compare this against the outer product. A single `k` step needs `M` values of `A` and `N` values of `B` in the registers. Since `M` and `N` are both far too large to fit in the register file, every CUDA thread instead computes a small `m×n` tile of the output matrix and loads only `m + n` values. Assuming `m = n`, each thread computes an `m×m` output tile from `2m` loaded values, resulting in an AI of `m² / 2m = m/2`.

Therefore, for the outer product to sustain the SM's peak throughput,

`m / 2 = 4` → `m = 8`.

This means every thread must load at least `8` values of `A` and `8` values of `B` into its registers and compute an `8×8` output tile. The register budget for this is `2m = 16` registers for the inputs plus `m² = 64` registers for the accumulators — `80` registers per thread. This is well under the 255-register-per-thread limit.

Every thread is now repsonsible for computing a `8x8` tile of the output matrix. This is referred to as a **thread tile** (or a lane tile) and this technique is referred to as **register tiling**. 

![Arithmetic intensity grows as m over 2 with the thread tile size, crossing the required intensity of 4 at m equals 8, where a thread loads 8 values of A and 8 values of B to compute an 8 by 8 tile of 64 accumulators](./thread-tile-8x8.png)