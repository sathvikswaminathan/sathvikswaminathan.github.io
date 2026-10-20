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
description: Why fast GPU kernels tile at three levels instead of one, and how each tier of the hierarchy multiplies arithmetic intensity.
---

Write a matrix multiply the obvious way and it will be correct, it will issue exactly the right number of floating-point operations, and it will run roughly an order of magnitude below what the hardware can do. Nothing is wrong with the arithmetic. The problem is that the kernel spends almost all of its time waiting for operands to arrive.

The fix is tiling, and the thing that trips people up is that it isn't one transformation applied once. Fast GEMM kernels tile at three nested levels, each one targeting a different tier of the memory hierarchy. This post is about why three, and what each level actually buys you.

## Table of contents

## The problem is a ratio

Start with the naive kernel: one thread per output element of `C = A × B`, where `A` is M×K and `B` is K×N.

```c
// one thread, one output element
float acc = 0.0f;
for (int k = 0; k < K; ++k)
    acc += A[row * K + k] * B[k * N + col];
C[row * N + col] = acc;
```

Count what that thread does. It performs `2K` floating-point operations — a multiply and an add per iteration. It reads `2K` values from memory, one element of `A` and one of `B` per iteration. In fp32 that's `8K` bytes.

The ratio of those two numbers is the kernel's **arithmetic intensity**:

```text
AI = (2K FLOP) / (8K bytes) = 0.25 FLOP/byte
```

The `K` cancels. Making the matrices bigger does not help — this kernel performs a quarter of a floating-point operation for every byte it moves, forever.

Now compare that against the machine. A GPU's *ridge point* is the ratio of its peak compute rate to its peak memory bandwidth, and it tells you the arithmetic intensity a kernel must reach before it stops being memory-bound. For a modern accelerator doing fp32 vector math — call it tens of TFLOP/s against a couple of TB/s of HBM — the ridge point sits somewhere around 10 FLOP/byte. Switch to the tensor/matrix cores and the compute number jumps by more than an order of magnitude while bandwidth stays roughly put, pushing the ridge point into the hundreds.

Against a requirement of 10, a kernel delivering 0.25 is leaving about 97% of the machine idle. Against a tensor-core requirement of several hundred, it's worse than that.

So the whole game is raising one ratio. Every tiling decision below is in service of that.

## What the hierarchy actually offers

The reason the naive kernel is slow is that it sources every operand from the wrong place. A GPU gives you four tiers, and they differ by orders of magnitude in both bandwidth and capacity:

| Tier | Capacity | Relative bandwidth | Scope |
| --- | --- | --- | --- |
| HBM / device memory | tens of GB | 1× | whole device |
| L2 | tens of MB | ~5–10× | whole device, shared |
| Shared memory / LDS | tens to ~200 KB per SM/CU | ~50–100× | one threadblock |
| Registers | ~256 KB per SM/CU | ~500×+ | one thread |

Two things matter about this table. The first is the obvious one: bandwidth climbs steeply as you move down. The second is less obvious but more important — **capacity shrinks just as steeply**, and the scope of sharing narrows with it. Registers are the fastest storage on the chip and also the most private; a value in a register is useful to exactly one thread.

That tension is what forces a hierarchy of tiles rather than a single one. You cannot just "put the working set in registers." You stage it: a block-sized chunk into shared memory, a slice of that into registers, and you arrange the loop nest so each chunk gets reused many times before it's evicted.

## Level 1: the threadblock tile

Stop assigning one thread per output element. Assign one *threadblock* to a T×T tile of `C`, and march that tile along the K dimension in steps of `BK`:

```c
__shared__ float As[BM][BK], Bs[BK][BN];

for (int k0 = 0; k0 < K; k0 += BK) {
    // cooperatively stage one slab of A and B into shared memory
    load_tile(As, A, k0);
    load_tile(Bs, B, k0);
    __syncthreads();

    // every thread in the block now reads from shared, not HBM
    for (int k = 0; k < BK; ++k)
        /* ... accumulate ... */;
    __syncthreads();
}
```

Redo the ratio for a single k-step. The block loads `BM × BK` elements of `A` and `BK × BN` of `B`. With a square tile `BM = BN = T`, that's `2 × T × BK` elements, or `8 × T × BK` bytes in fp32. The arithmetic it performs on them is a `T×T` by `BK` outer-product accumulation: `2 × T² × BK` FLOP.

```text
       2 * T^2 * BK FLOP        T
AI = ----------------------  =  -  FLOP/byte
        8 * T * BK bytes        4
```

`BK` cancels out, which is a useful thing to know — the depth of the K-step does not affect arithmetic intensity, only the tile edge `T` does. And the relationship is linear: **doubling the tile edge doubles arithmetic intensity.**

A 128×128 threadblock tile gives 32 FLOP/byte. That clears a 10 FLOP/byte ridge point comfortably. The naive kernel's 0.25 became 32 because each element loaded from HBM is now used `T` times instead of once.

For 2-byte types the denominator halves and you get `T/2`, which is part of why reduced precision is such a large win — it buys bandwidth and arithmetic intensity simultaneously.

The limit on `T` is capacity. Shared memory per block has to hold both tiles:

```text
bytes = (BM * BK + BK * BN) * sizeof(dtype)
```

At `BM = BN = 128`, `BK = 16`, fp32: `(128·16 + 16·128) × 4 = 16` KB per buffer. That fits, with room to double-buffer — which we'll need shortly.

## Level 2: the warp tile

The threadblock tile solved HBM traffic. It did not solve shared memory traffic, and shared memory is not free — it is banked, it has finite bandwidth, and on most architectures a warp-wide access costs real cycles and can serialize on bank conflicts.

So you subdivide again. The block's T×T tile is partitioned among the warps in the block, each warp owning a sub-tile. This level is less about a new arithmetic-intensity formula and more about three practical things:

- **Bank-conflict-free access patterns.** A warp's threads should hit distinct banks on each shared-memory read. This is what drives the padded and swizzled shared-memory layouts you see in real kernels — `As[BK][BM + 4]`, XOR-swizzled offsets, and so on. The padding exists purely to break the stride that would otherwise map many lanes onto one bank.
- **Matching the matrix-core shape.** If you're issuing MMA instructions, the hardware dictates fragment geometry — a fixed M×N×K shape per instruction, with operands distributed across the lanes of the warp in a specific pattern. The warp tile has to be an integer multiple of that shape, or you waste slots.
- **Operand reuse across the warp.** Lanes within a warp share fragments, so a value fetched from shared memory feeds several lanes' worth of math.

The warp tile is the layer where portability tends to break down, because the fragment shapes and the banking rules are architecture-specific in a way the other two levels aren't.

## Level 3: the register tile

The innermost level is where the arithmetic intensity argument repeats itself, one tier down.

Each thread owns an `r × r` block of accumulators held in registers. Per k-step, it loads `r` values of `A` and `r` of `B` from shared memory, then does an `r × r` outer product into its accumulators:

```text
  2 * r^2 FLOP
----------------- = r FLOP per shared-memory value
 2 * r shared loads
```

Same shape as the threadblock formula, same linear payoff, applied to a different tier. With `r = 8`, each value pulled out of shared memory feeds 8 FLOP instead of 1.

The cost is register pressure. An `8×8` thread tile needs 64 accumulator registers, plus 8 + 8 for the fragments, plus addressing and loop state — call it ~85 registers per thread. Against a per-thread ceiling of 255 architectural registers that's fine, but registers per SM are a fixed pool, so a higher per-thread count means fewer resident warps. Push `r` too far and the compiler spills to local memory, which is backed by device memory, and you have accidentally reintroduced the original problem at the innermost loop.

This is the characteristic tuning tension of GPU GEMM: **bigger tiles raise arithmetic intensity but lower occupancy**, and occupancy is what hides the latency the tiles didn't eliminate. There is no analytic optimum. It's why CUTLASS, Triton, and friends all end up doing some form of autotuning over tile shapes.

## The loop nest, assembled

All three levels together:

```
for k0 in range(0, K, BK):              # threadblock: HBM -> shared
    stage A[:, k0:k0+BK], B[k0:k0+BK, :] into shared memory
    barrier

    for k1 in range(0, BK, WK):         # warp: shared -> fragments
        load warp fragments from shared

        for k2 in range(0, WK):         # thread: fragments -> registers
            accumulate r x r outer product in registers

    barrier

write accumulators back to C
```

Three nested loops over `K`, each one feeding a different tier. Each tier multiplies reuse by roughly its tile edge, and the product of those factors is what carries you from 0.25 FLOP/byte to something the compute units can actually keep up with.

## Keeping the pipe full

Tiling alone still leaves the kernel stalling, because the structure above is strictly serial: load a tile, wait at the barrier, compute on it, wait again, load the next. While the loads are in flight the math units do nothing.

The standard fix is **double buffering**. Allocate two shared-memory buffers. While the math runs on buffer 0, the loads for the next k-step are already in flight into buffer 1, and the two swap each iteration. This is why the capacity budget above mattered: 16 KB per buffer means 32 KB double-buffered, which still fits.

Hardware helps here. Modern GPUs provide asynchronous global-to-shared copy instructions that bypass the register file entirely — `cp.async` on recent NVIDIA parts, direct global-to-LDS loads on AMD. Without them the "copy" is really a load into registers followed by a store to shared, which burns registers and forces the thread to wait on the load. With them, the copy is a fire-and-forget operation the thread can issue and then go do math while it completes. Deepen the pipeline to three or four stages and you can cover most of HBM latency.

## Above the threadblock: L2 and swizzling

There's a tier the three levels skip: L2 is shared across the whole device, so the *order* in which threadblocks are scheduled determines whether they find their operands already in L2.

With default linear ordering, consecutively-scheduled blocks walk along one row of the output. Collectively, a wave of them touches one narrow band of `A` and nearly all of `B` — a large footprint with poor locality.

Threadblock swizzling remaps the block index so that concurrently-running blocks form a compact rectangle in the output instead of a long row. A square-ish group of blocks touches a much smaller slice of both operands, and those slices stay resident in L2 across the wave.

```c
// group blocks into column-strips of width GROUP so that
// concurrent blocks form a rectangle, not a row
int group_id    = blockIdx.x / (GROUP * grid_n);
int first_m     = group_id * GROUP;
int group_size  = min(grid_m - first_m, GROUP);
int tile_m      = first_m + (blockIdx.x % group_size);
int tile_n      = (blockIdx.x % (GROUP * grid_n)) / group_size;
```

No change in arithmetic, no change in what gets computed — purely a renumbering. It routinely buys double-digit percentage gains on large GEMMs, because it converts HBM traffic into L2 hits.

## When the tiles don't fit the problem

The hierarchy assumes `M` and `N` are large enough to produce many threadblock tiles. Skinny problems break that assumption, and they're common — a batch-size-1 decode step in a transformer is effectively a matrix-vector product.

If `M × N` divided by the tile size yields fewer tiles than the GPU has SMs, most of the machine sits idle no matter how good the tiling is. The usual answer is **split-K**: partition the reduction dimension across multiple threadblocks, let each compute a partial sum over its slice of `K`, and combine the partials afterward — either with atomics or a separate reduction kernel.

This trades extra memory traffic for the partials against getting the whole device busy. It's the right trade exactly when you're launch-bound rather than bandwidth-bound, and the wrong one otherwise.

## The thing worth remembering

Strip away the architecture-specific detail and one idea generates the whole structure:

> A tile of edge length `t` staged in a given tier of the hierarchy multiplies operand reuse out of that tier by roughly `t`.

Apply it against HBM and you get the threadblock tile. Apply it against shared memory and you get the register tile. The warp tile sits between them to make the shared-memory access pattern realizable on banked hardware. Threadblock swizzling applies the same logic to L2 but through scheduling order rather than through a data tile, because you don't address L2 explicitly.

The levels exist because the tiers exist. Every tier of storage that is faster and smaller than the one above it wants its own tile, sized to its capacity and shaped to its access rules — and the reuse factors multiply.

That's also why this generalizes past GEMM. Convolutions, attention, stencils, and FFTs all get the same treatment, because the underlying constraint has nothing to do with matrix multiply. It's just a hierarchy with a steep bandwidth gradient and a steep capacity gradient pointing the opposite way.
