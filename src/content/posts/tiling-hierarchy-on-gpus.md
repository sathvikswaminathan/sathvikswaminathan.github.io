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

GPUs have a tiered memory hierarchy, with each level being slower than the previous one: register file, shared memory/L1 cache, L2 cache, and global memory (HBM). The compute cores remain idle if data is fetched from slower memory tiers too often, or if memory requests stall due to inefficient access patterns such as shared-memory bank conflicts or uncoalesced global loads. The fix is to organize data across these tiers such that data loaded into a faster tier is reused as many times as possible before being evicted, and to group accesses so that they are conflict-free. This way, the compute cores are fed from the fastest tier with minimal memory stalls.

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

For a single FMA, the AI is `2 FLOPs / 8 bytes` = `0.25 FLOPs per byte`, or 2 loads per FMA. A higher AI helps keep the cores busy for longer.

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

### Register Tiling

How much AI does it take to saturate the CUDA cores?

As we will see later, CUDA threads load data from GMEM into SMEM, so every load into the register file is served by shared memory.

On an H100, an SM sustains a peak throughput of `128` FMAs per cycle, and each FMA needs `2` FP32 operands. For a lane to issue an FMA in the next cycle, its operands must already be sitting in registers. But the SM's shared-memory pipe only delivers about `128 bytes` (`32` FP32 values) per cycle into the register file. If every FMA used unique operands, the SM would need `128 × 2 = 256` FP32 values every cycle. Those `32` loaded values must therefore be shared across all `128` FMAs, which pins the required arithmetic intensity at `128 FMAs / 32 loads = 4 FMAs` per load.

![Without reuse, 32 loaded FP32 values feed only 16 FMAs and 112 of the SM's 128 lanes sit idle; with each loaded value feeding 4 FMAs, all 128 lanes stay busy](./required-arithmetic-intensity.png)

Now compare this against the outer product. A single `k` step needs `M` values of `A` and `N` values of `B` in the registers. Since `M` and `N` are both far too large to fit in the register file, every CUDA thread instead computes a small `m×n` tile of the output matrix and loads only `m + n` values. Assuming `m = n`, each thread computes an `m×m` output tile from `2m` loaded values, resulting in an AI of `m² / 2m = m/2`.

Therefore, for the outer product to sustain the SM's peak throughput,

`m / 2 = 4` → `m = 8`.

This means every thread must load at least `8` values of `A` and `8` values of `B` into its registers and compute an `8×8` output tile. The register budget for this is `2m = 16` registers for the inputs plus `m² = 64` registers for the accumulators — `80` registers per thread. This is well under the 255-register-per-thread limit.

Every thread is now repsonsible for computing a `8x8` tile of the output matrix. This is referred to as a **thread tile (TM, TN)** (or a lane tile) and this technique is referred to as **register tiling**. 

Here is what the kernel looks like from a single thread's perspective:

```c
for k:
    load a[0..7], b[0..7]        
    64 FFMAs using a, b          
```

![Arithmetic intensity grows as m over 2 with the thread tile size, crossing the required intensity of 4 at m equals 8, where a thread loads 8 values of A and 8 values of B to compute an 8 by 8 tile of 64 accumulators](./thread-tile-8x8.png)

## Issue Slot Efficiency

The previous section went over how to load data into registers to hit an SM's peak compute throughput. That only matters, however, if the warp scheduler actually has an FMA to issue in every cycle; if it does not, register tiling does not matter. There are two reasons an FMA may not be issued in a given issue slot:

| Issue slot | What happens in that cycle | Example causes |
| --- | --- | --- |
| Overhead slot | Something issued, but it wasn't an FFMA | `LDS`, address math, loop branch, `__syncthreads` |
| Idle slot | Nothing issued: every warp on the SMSP is stalled | Waiting for a load to return, shared-memory pipe congested, barrier |

It is therefore useful to quantify the fraction of issue slots that actually carry an FMA, since that is the metric we want to maximize:

```text
issue-slot efficiency η = cycles that issued an FMA / total cycles
```

Overhead slots cannot be eliminated entirely: every thread still has to load its operands, compute addresses, advance the loop, and synchronize with its peers. Their cost can only be amortized by doing more compute per overhead instruction, which is exactly what a larger thread tile gives us.

### Eliminating Idle slots

Idle slots, on the other hand, can be reduced by writing better kernel code.

Consider the code from the previous section:

```c
for k:
    load a[0..7], b[0..7]        // 16 LDS
    64 FFMAs using a, b          // must wait for the loads        
```

On an H100, a shared-memory load (`LDS`) takes about 30 clock cycles when there are no bank conflicts. Tracing one `k` step through the scheduler:

```text
cycle   0-15  : issue 16 LDS                          (16 overhead slots)
cycle  16-37  : nothing to issue: the first FFMA needs
                a0 and b0, and b0 (issued at cycle 8)
                comes back at ~38                     (~22 idle slots)
cycle  38-101 : 64 FFMAs                              (64 useful slots)
-------------------------------------------------------------
~102 cycles for 64 FFMAs  ->  η ≈ 63%
```

The CUDA cores are idle during the 22 idle slots, waiting for the data to be loaded.

Register prefetching removes them.

#### Register prefetching

The idea is to issue the `LDS` instructions for step `k+1` *before* running the FMAs for step `k`. Those loads take `16 + 30 = 46` cycles to land, while the 64 FMAs of step `k` take 64 cycles, so the memory load latency of step `k+1` is completely overlapped with  the compute of step `k`. When the loop reaches step `k+1`, its operands are already sitting in registers: the scheduler issues the `LDS` for step `k+2` and then immediately issues the FMAs.

```c
load a_next, b_next for k = 0
for k:
    a, b = a_next, b_next             
    issue 16 LDS for k+1 → a_next, b_next
    64 FFMAs using a, b               
```

Scheduler trace during step k:

```text
cycle  0-15 : issue 16 LDS for k+1      (overhead)
cycle 16-79 : 64 FFMAs for step k       (useful)
              k+1's loads, issued 0-15, are back by ~45, well before cycle 80
-------------------------------------------------------------
80 cycles for 64 FFMAs  ->  η = 80%
```

Register prefetching raises η from ~63% to 80%. However, this comes at a cost. Each thread must now hold two sets of operands at once — the one it is computing with and the one it is prefetching, increasing the register budget from 80 to 96 registers per thread. This is referred to as **register double buffering**.

These `LDS` instructions are issued as a warp-wide load instruction by the whole warp to shared memory. They carry addresses requested by all 32 threads, each of which is loading data for its `8×8` tile. The effective bandwidth with which SMEM delivers this data to the warp depends on which tiles the 32 threads are trying to compute. If they are grouped such that the `LDS` leads to a lot of bank conflicts, that would again lead to idle slots, as the `LDS` for step k + 1 can take longer than the compute in step `k`. This is what warp tiling addresses.

#### Warp Tiling

Every one of the 32 threads in a warp computes a `TM × TN` = 8×8 thread tile of the output. A warp therefore computes 32 such thread tiles. How these 32 tiles are arranged within the output matrix determines which addresses are grouped into a warp's `LDS` instruction. This, in turn, determines whether the instruction is serviced by shared memory with bank conflicts, can take advantage of broadcasts, and so on.

If bank conflicts occur, the `LDS` instruction takes longer to deliver all the required data, resulting in idle slots. On the other hand, if there are no bank conflicts and the data is shared across threads such that it can be broadcast, `LDS` can deliver the data much more efficiently and its latency can often be overlapped with compute.

Therefore, the placement of these 32 thread tiles within a warp is critical for minimizing idle slots and increasing issue-slot efficiency.

Now, assume the thread block (CTA) containing this warp computes a `BM × BN` = 128×128 output tile. At 8×8 per thread, the CTA needs `(BM · BN) / (TM · TN) = 16,384 / 64 = 256` threads, or 8 warps. Within a warp, the 32 lanes can be laid out as any `LM × LN` grid with `LM · LN = 32`, which makes the warp responsible for a **warp tile** of `WM = TM · LM` rows by `WN = TN · LN` columns. Each (row, column) position in the `LM × LN` lane grid refers to the `TM x TN` /  `8x8` thread tile owned by a single lane.

Below are different possible arrangements of `LM x LN`

| Lane grid `LM × LN` | Warp tile `WM × WN` | Unique `A` (= `WM`) | Unique `B` (= `WN`) | Total unique floats (Unique `A` + Unique `B`) | FMAs per float |
|---|---|---|---|---|---|
| 1×32 | 8×256 | 8 | 256 | 264 | 7.8 ✗ |
| 2×16 | 16×128 | 16 | 128 | 144 | 14.2 |
| 4×8 | 32×64 | 32 | 64 | **96** | **21.3** |
| 8×4 | 64×32 | 64 | 32 | **96** | **21.3** |
| 16×2 | 128×16 | 128 | 16 | 144 | 14.2 |
| 32×1 | 256×8 | 256 | 8 | 264 | 7.8 ✗ |

The `1×32` and `32×1` lane grids are not possible because the warp tile would be 8×256 or 256×8, and its 256-wide dimension exceeds the 128×128 CTA tile (`WN > BN` or `WM > BM`).

![The six lane grids drawn against the 128 by 128 CTA tile. Every arrangement issues the same 512 operand requests per k step, but the warp tile it implies changes how many of those addresses are distinct: 1 by 32 and 32 by 1 need a 256-wide warp tile and do not fit in the CTA at all, 2 by 16 and 16 by 2 pull 144 distinct floats, and the squarest feasible grids, 4 by 8 and 8 by 4, pull only 96](./warp-shape.png)

Among the remaining grids, `4×8` and `8×4` result in the fewest unique floats, allowing data to be broadcast while serving an `LDS` request from shared memory. Both also divide the CTA evenly: a 32×64 warp tile gives a 4×2 arrangement of the 8 warps, and a 64×32 warp tile gives a 2×4 arrangement.

The warp tile is the union of all 32 thread tiles assigned to a warp. Its dimensions are therefore:

```
(WM, WN) = (TM · LM, TN · LN)
```

### Reducing Overhead Slots

Why stop at a thread tile size of 8×8? With register prefetching, an m×n tile needs at least m·n + 2(m + n) registers: m·n accumulators, plus two sets of m + n operands. On top of that come ~20–30 registers for addresses, loop counters and indices, and the hardware limit is 255 registers per thread. An 8×16 tile would need 176 registers before this overhead, still under the limit. A bigger tile has a higher arithmetic intensity, meaning we perform more FMAs per `LDS` instruction. Fewer issue slots are then used up by `LDS`, which improves the issue-slot efficiency.

| Tile | FFMAs / k | LDS / k | Ceiling η | Registers (acc + 2× operands) |
|---|---|---|---|---|
| 4×4 | 16 | 8 | 67% | 16 + 16 = 32 |
| 8×8 | 64 | 16 | 80% | 64 + 32 = 96 |
| 8×16 | 128 | 24 | 84% | 128 + 48 = 176 |
| 16×16 | 256 | 32 | 89% | 256 + 64 = 320 > 255 ✗ |

![Registers per thread grow with the tile area, from 32 at 4 by 4 to 320 at 16 by 16, which overshoots the 255 register limit and spills to local memory, while the issue-slot efficiency ceiling it buys flattens from 67% to 80% to 84%](./tile-size-tradeoff.png)

Moving from an 8×8 tile to an 8×16 tile raises the issue-slot efficiency ceiling by only 4%, while consuming 80 more registers per thread. A 16×16 tile doesn't fit at all: 320 registers is over the limit, so the compiler would spill registers into local memory, which is per-thread storage that lives in global memory. Every access to a spilled value then needs an extra instruction:

- `STL` (store local) when the value is evicted
- `LDL` (load local) when it is needed again

These instructions add overhead slots, and waiting for the `LDL`s to return adds idle slots. That's the very problem we are trying to avoid, so choosing 16×16 would be counterproductive.

To see why 8×16 might also be problematic, let's assume a thread block (CTA) computes a 128×128 block of the 4096×4096 output matrix. The number of threads needed per block is:

threads per CTA = outputs per CTA / outputs per thread

For an 8×8 tile that is 16,384 / 64 = 256 threads, and for an 8×16 tile it is 16,384 / 128 = 128 threads. Including the bookkeeping registers, an 8×8 tile needs about 128 registers per thread and an 8×16 tile about 200. An H100 SM has 65,536 registers, so:

| Configuration | Threads/CTA | Regs/thread | Regs/CTA | CTAs per SM (65,536 regs) | Warps/SM | Warps/SMSP |
|---|---|---|---|---|---|---|
| 8×8, CTA 128×128 | 256 | ~128 | 32,768 | 2 | 16 | 4 |
| 8×16, CTA 128×128 | 128 | ~200 | 25,600 | 2 | 8 | 2 |

While 8×16 slightly raises the efficiency ceiling, it does so at the cost of halving the occupancy. Each SMSP's warp scheduler would only have 2 warps to choose from. If a warp stalls, for example on a global load or at a barrier, the scheduler is more likely to have no other work to pick up, and the idle slots go up.

![With 4 warps per SMSP a stalled warp is always covered by another one, so every issue slot is filled; with only 2 warps per SMSP the same stall pattern leaves cycles where both warps are stalled and the issue slot goes idle](./warps-per-smsp.png)

So there are clear trade-offs in choosing the tile size. 8×8 is not a rule, and libraries often autotune tile sizes for each GPU and problem shape. For the rest of this blog, we will assume a thread tile size of 8x8.

Until now, we have assumed that the data required by the FMAs is already available in shared memory. In the next section we will see how this is done.

> Every tiling level lowers the bandwidth demand on the memory level below it by reusing each fetched value across more FMAs. Register tiling and warp tiling both protect shared-memory bandwidth, from two sides: register tiling reduces how many values each thread requests per FMA, while warp tiling arranges the lanes so that serving those requests costs fewer wavefronts (via broadcast and conflict-free access). Next, we will look at block tiling, which reuses each global-memory load across the whole CTA through shared memory, cutting L2/HBM traffic.

## Block Tiling

All the warps in an SM load their inputs from shared memory. At each `k` step, a single warp loads `WM` entries of the `k`-th column of `A` and `WN` entries of the `k`-th row of `B`. That is `WM + WN` floats per warp loaded from shared memory, and the SM hosts more than a single warp to hide latency. These values need to be staged into shared memory from global memory by the CTA before its warps can access them.

Let's say a CTA is responsible for computing a `BM × BN` tile of the output matrix `C`. This means that at each `k` step, the CTA loads `BM` entries of column `k` of `A` and `BN` entries of row `k` of `B` from global memory into shared memory.

All the threads in this CTA then perform the outer product, loading `BM + BN` floats and computing `BM · BN` FMAs. Since every FMA is `2 FLOPs` and every float is `4 bytes`, this results in an AI of

$$
\text{AI} = \frac{2 \cdot BM \cdot BN}{4 \cdot (BM + BN)} = \frac{BM \cdot BN}{2 \cdot (BM + BN)} \ \text{FLOPs per byte}
$$

Since shared memory is limited, `BM + BN` has to stay within a budget. Under this constraint, the AI is maximized when `BM = BN = B`, resulting in an AI of `B / 4` FLOPs per byte.

> The threads in a CTA can choose to load `BM` entries of `A` and `BN` entries of `B` across multiple `k` steps, say `BK`. However, this does not affect the AI, since the FLOPs performed would be `BK` times higher and the floats loaded would also be `BK` times higher, canceling out in the AI equation. What this does achieve is fewer trips to GMEM, at the cost of a higher shared memory footprint. SMEM per CTA would then be `BK · (BM + BN) · 4` bytes, and double that if double buffering is involved.

### Block tile dimension

#### Lower Limit for B

The lower limit for `B` is set by the condition that GMEM does not become the bottleneck. To be conservative, we assume that L2 does not help, and every byte loaded by a CTA comes from HBM.

The documented boost clock of the H100 SXM is 1.98 GHz, resulting in a peak throughput of `67 TFLOPs/sec`. HBM3 supports a bandwidth of `3.35 TB/sec`.

> However, the sustained clock rate is about 1.75 GHz to keep power consumption in check under dense FP32 workloads, resulting in a peak throughput of about `59 TFLOPs/sec`. This lowers the required AI to about `17.6 FLOPs per byte`. On the other hand, the achievable HBM bandwidth is also lower than `3.35 TB/sec`, which pushes it back up. We will use `20 FLOPs per byte` as a slightly conservative target.

To prevent HBM transfers from being the bottleneck, we need a minimum AI of

$$
\frac{67 \ \text{TFLOPs/sec}}{3.35 \ \text{TB/sec}} = 20 \ \text{FLOPs per byte}
$$

$$
\implies \frac{B}{4} \geq 20 \implies B \geq 80
$$

For a square block tile, `B` must be at least `80`.

#### Upper Limit for B

Big tile sizes require more threads, and every thread requires registers for its accumulators, inputs and other bookkeeping.

As we saw earlier, a thread with an `8×8` thread tile requires about `128` registers and produces `64` outputs. A CTA with a `B×B` block tile produces `B²` outputs. Therefore,

$$
\text{threads per CTA} = \frac{B^2}{64}
$$

$$
\text{registers per CTA} = \frac{B^2}{64} \times 128 = 2B^2
$$

An H100 SM has `65,536` registers, and the whole CTA must fit on a single SM. Therefore,

$$
2B^2 \leq 65{,}536 \implies B \leq \sqrt{32{,}768} \approx 181
$$

So for square block tiles, the final inequality becomes

$$
80 \leq B \leq 181
$$

`B` also needs to divide evenly into warp tiles of `32×64`, and powers of two keep the index math to shifts and masks. The only power of two in this range is `128`.

#### Candidates

Here is how the different block tile sizes compare against the `20 FLOPs per byte` target:

| CTA tile  | FLOPs/byte | vs. 20 | Verdict                                         |
|-----------|-----------:|-------:|-------------------------------------------------|
| 32×32     | 8.0        | 0.40×  | HBM-bound                                       |
| 64×64     | 16.0       | 0.80×  | HBM-bound without L2 help                       |
| 64×128    | 21.3       | 1.07×  | nominally clears, no margin                     |
| 96×128    | 27.4       | 1.37×  | clears; awkward warp decomposition              |
| **128×128** | **32.0** | **1.60×** | **clears, with headroom**                    |
| 256×128   | 42.7       | 2.13×  | more margin, at a price (only 1 CTA per SM)     |
| 256×256   | 64.0       | 3.20×  | cannot launch (exceeds the register file)       |

Keep in mind that this assumes every byte comes from HBM. In practice, CTAs that share rows of `A` or columns of `B` get some of their data from L2, which lowers the HBM traffic further.

![Arithmetic intensity grows as B over 4 with the square block tile side. Below B equals 80 the tile sits under the 20 FLOPs per byte line and the cores wait on global memory; above B equals 181 the CTA needs more than the 65,536 registers of an SM and cannot be resident; 128 is the only power of two inside that window, landing at 32 FLOPs per byte. The ladder of candidate CTA tiles alongside shows 32 by 32 and 64 by 64 falling short of the target, 64 by 128 clearing it with no margin, and 128 by 128 and 256 by 128 clearing it with headroom](./block-tile-bounds.png)

#### Is 128×128 the best we can do?

The square tile argument assumed that shared memory was the budget. So let's check how much shared memory a `128×128` block tile actually uses. With `BK = 8` and double buffering:

$$
2 \times 8 \times (128 + 128) \times 4 \ \text{bytes} = 16 \ \text{KB per CTA}
$$

The register file allows `2` such CTAs per SM, so together they use `32 KB` out of the `228 KB` of shared memory available on an H100 SM. Even with `BK = 32`, each CTA uses only `64 KB`, or `128 KB` for two. Most of the shared memory is left unused.

The budget that actually limits us is the register file. The register file limits the *area* of the tile, `BM · BN` (the accumulators), and not `BM + BN`. For a fixed area, a square tile still has the highest AI, but nothing forces us to keep the area fixed. Let's look at bigger tiles: `256×128` and `256×256`.

##### 256×256

Plugging `BM = BN = 256` into the formulas above,

$$
\text{threads per CTA} = \frac{256 \times 256}{64} = 1024
$$

$$
\text{registers per CTA} = 1024 \times 128 = 131{,}072
$$

This is twice the size of the SM's register file.

##### 256×128

Plugging `BM = 256` and `BN = 128` into the formulas,

$$
\text{threads per CTA} = \frac{256 \times 128}{64} = 512
$$

$$
\text{registers per CTA} = 512 \times 128 = 65{,}536
$$

This uses exactly the whole register file. The CTA fits, but there is no room for a second CTA on the same SM. Its 16 warps are arranged as `8×2` warp tiles of `32×64`, and it uses only `2 × 8 × (256 + 128) × 4 bytes = 24 KB` of shared memory.

| Configuration | Threads/CTA | Regs/CTA | CTAs/SM | Warps/SM | Warps/SMSP | SMEM/SM | AI (FLOPs/byte) | HBM traffic (4096³) |
|---------------|------------:|---------:|--------:|---------:|-----------:|--------:|----------------:|--------------------:|
| 128×128       | 256         | 32,768   | 2       | 16       | 4          | 32 KB   | 32.0            | 4.29 GB             |
| 256×128       | 512         | 65,536   | 1       | 16       | 4          | 24 KB   | 42.7            | 3.22 GB             |
| 256×256       | 1024        | 131,072  | 0 ✗     | —        | —          | —       | 64.0            | —                   |

Notice that both configurations have the same number of warps per SM, and therefore the same `4` warps per SMSP for the warp scheduler to choose from. The difference is that with `128×128`, those warps belong to two independent CTAs, while with `256×128` they all belong to one.

#### One Big CTA or Two Small Ones?

The warps in a CTA are tied together by `__syncthreads`. This is where having two CTAs per SM helps:

- **Barriers:** With two `128×128` CTAs, a `__syncthreads` stalls only the 8 warps of its own CTA. While they wait for the slowest warp, or for a global load that was not fully hidden, the 8 warps of the other CTA can keep issuing FFMAs. With a single `256×128` CTA, all 16 warps wait at the same barrier at every `k` tile, and the slowest warp holds up the whole SM.
- **Prologue and epilogue:** Every CTA starts by loading its first tile, with no compute to overlap it with, and ends by writing its output tile to `C`. With two CTAs per SM, one CTA is usually computing while the other is starting or finishing. With one CTA per SM, the SM sits idle during every prologue and epilogue.
- **Small problems:** Bigger tiles mean fewer CTAs. For a `1024×1024×1024` matrix multiply, `128×128` results in 64 CTAs, while `256×128` results in only 32 CTAs for 132 SMs.

![Schematic SM timelines for the two configurations. With two 128 by 128 CTAs, the barriers of CTA 0 and CTA 1 are independent and fall at different times, so whenever one CTA is waiting at a __syncthreads the other still has FFMAs to issue and the SM loses only its prologue. With one 256 by 128 CTA, all 16 warps wait at the same barrier, so every barrier, along with the prologue and the epilogue, leaves the SM with nothing to issue](./one-cta-vs-two.png)

So when would `256×128` be worth giving up the second CTA? Only when the `20 FLOPs per byte` target no longer holds, which can happen in two ways:

- **The GEMM gets less HBM bandwidth than we assumed.** The target assumes the kernel has all `3.35 TB/sec` to itself. If other kernels run at the same time, the GEMM's share can drop below the `2.1 TB/sec` that `128×128` needs. `256×128` needs only `1.6 TB/sec`

- **The compute rate is much higher.** On the same H100, FP16 tensor cores deliver about `989 TFLOPs/sec`, which raises the target to about `300 FLOPs per byte`. No block tile reaches that from HBM alone, so tensor-core kernels rely on L2 reuse and use the biggest tiles that fit, typically `128×256` or `256×128`.

Neither applies to our FP32 GEMM, so for the rest of this blog we will assume a block tile size of `128×128`.

> Block tiling lowers the bandwidth demand on global memory by reusing each value loaded into shared memory across the whole CTA. Shared memory decides how deep (`BK`) a block tile can be, but the register file decides how big (`BM × BN`) it can be, and on an H100 the register file runs out first. Within this budget, `128×128` already clears the HBM target with room to spare, so we spend the remaining budget on a second CTA per SM to hide latency, rather than on a bigger tile.

### Padding

### Swizzling