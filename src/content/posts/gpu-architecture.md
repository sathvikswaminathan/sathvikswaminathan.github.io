---
author: Sathvik Swaminathan
pubDatetime: 2026-10-20T05:20:00Z
title: GPU Architecture
featured: false
tags:
  - gpu
  - computer-architecture
  - simt
  - performance
description: How GPUs trade latency for throughput, and why the SIMT execution model, the register file, and the memory system all follow from that one decision.
---

A CPU core and a GPU core are both pipelines that fetch instructions and execute them, and the family resemblance stops at about that level of description. Almost everything else about a GPU — the width of the register file, the absence of large caches, the fact that branches cost you throughput rather than a mispredict penalty — follows from a single design commitment made early and carried all the way through.

That commitment is: **optimize for throughput, and pay for it with latency.**

This post walks the structure from that premise down to the parts you actually have to reason about when a kernel is slow.

## Table of contents

## Two answers to the same problem

Every processor has to deal with the fact that memory is far away. A load from DRAM costs hundreds of cycles, and during those cycles the pipeline has nothing to do. There are two ways to respond.

The CPU's answer is to make the stall rare and short. Spend transistors on large multi-level caches so the data is usually nearby. Spend more on branch prediction, out-of-order execution, and speculative execution so that when a stall does happen, there is other useful work already in flight. This is *latency avoidance*, and it is extraordinarily expensive per unit of throughput — on a modern CPU core, the structures that exist purely to keep one instruction stream moving fast dwarf the arithmetic units they feed.

The GPU's answer is the opposite. Accept that the stall will happen. Accept that it will be long. Then make sure there is always another thread ready to run, so the execution units never idle. This is *latency hiding*, and it is cheap per unit of throughput — the cost is a large register file and a thread scheduler, not a reorder buffer and a branch predictor.

The consequence is that a GPU is mostly arithmetic units and register storage, while a CPU is mostly control logic and cache. Both are the right answer to their respective workloads. The GPU's answer only works if you can supply it with thousands of independent threads, which is precisely the bet it makes.

## SIMT: threads that aren't quite threads

To make latency hiding affordable, GPUs amortize instruction fetch and decode across many lanes. Threads are grouped into fixed-size bundles that execute together in lockstep — a **warp** of 32 threads on NVIDIA hardware, a **wavefront** on AMD (64 on GCN and CDNA, 32 on RDNA).

The warp, not the thread, is the real unit of scheduling. One instruction is fetched, decoded, and issued once; it executes across all lanes simultaneously, each on its own data. You get the die-area efficiency of SIMD while writing code that looks scalar.

That abstraction is comfortable right up to the moment the lanes disagree.

### Divergence

```c
if (threadIdx.x % 2 == 0)
    heavy_path_A();      // half the lanes want this
else
    heavy_path_B();      // the other half want this
```

There is one program counter for the warp. The hardware cannot issue two different instructions in one cycle, so it executes both paths *sequentially*, masking off the inactive lanes in each. Both halves pay the full cost of both branches. A warp that diverges 32 ways runs at 1/32 of peak.

A few things follow:

- Divergence only matters **within** a warp. If every thread in a warp takes the same branch, the cost is zero — the untaken path is skipped entirely. Branching on `blockIdx.x`, or on `threadIdx.x / 32`, is free.
- The expensive case is a data-dependent branch whose outcome varies lane to lane. Sorting or binning data so that warps become uniform is a real and commonly-worthwhile optimization.
- On recent NVIDIA architectures (Volta onward) each thread has its own PC and call stack — *independent thread scheduling*. This makes some previously-illegal synchronization patterns safe, but it does not make divergence free. Reconvergence is explicit, which is why `__syncwarp()` exists and why the `_sync` variants of warp intrinsics took over from the implicit-reconvergence ones.

## Inside an SM

The **SM** (Streaming Multiprocessor) on NVIDIA, or **CU** (Compute Unit) on AMD, is the unit that actually gets replicated — tens to north of a hundred per chip. Each one contains:

- **Warp schedulers** — typically four per SM, each owning a slice of the resident warps. Every cycle a scheduler picks one *eligible* warp and issues an instruction from it.
- **A very large register file** — commonly 64K 32-bit registers, i.e. 256 KB per SM. This is larger than the SM's L1 cache, which is the opposite of the CPU ratio and is the single most telling structural fact about a GPU.
- **Execution units** — FP32 lanes, a smaller number of FP64 lanes on datacenter parts, integer units, special-function units for transcendentals, load/store units, and matrix/tensor cores.
- **Shared memory / LDS** — software-managed scratchpad, plus L1. On many NVIDIA parts these are carved out of the same physical SRAM, and the split is configurable.

The register file size is the key to the whole design. A CPU keeps one thread's architectural state live and context-switches through memory, which costs microseconds. A GPU keeps **every resident thread's full register state live simultaneously**. Nothing is saved or restored. Switching warps costs zero cycles — the scheduler just selects a different set of registers to read.

That is what makes "hide the latency with other threads" actually work. The entire strategy is bought with SRAM.

## Occupancy, and what it actually means

**Occupancy** is the ratio of warps resident on an SM to the architectural maximum. Because warp switching is free, resident warps are the pool the scheduler draws from when the current warp stalls. More warps, more chances to find one that's ready.

Three resources cap it, and the binding one is whichever runs out first:

| Resource | Effect |
| --- | --- |
| Registers per thread | more registers per thread → fewer resident threads |
| Shared memory per block | more shared memory per block → fewer resident blocks |
| Hard limits | max warps and max blocks per SM |

If each thread needs 64 registers and the SM has 65,536, at most 1,024 threads (32 warps) can be resident. Ask for 128 registers and you halve that.

The part people get wrong: **occupancy is a means, not an end.** It is not a score to maximize. What you actually need is enough *memory-level parallelism* to cover the latency — enough independent loads in flight. Occupancy is one way to get that; instruction-level parallelism within a single thread is another.

This is Volkov's well-known result: a kernel at low occupancy but with several independent loads per thread can outperform a high-occupancy kernel with one dependent load each. It's also exactly the tension from [the tiling post](/posts/anatomy-of-a-fast-gemm-on-cuda-cores/) — big register tiles raise arithmetic intensity and lower occupancy, and the optimum is usually not at either extreme.

Raising occupancy from very low to moderate usually helps a lot. Pushing from moderate to maximal, at the cost of smaller tiles and more spilling, usually hurts.

## The memory system

### Coalescing

The load/store unit services a warp's memory instruction by splitting it into transactions against fixed-size, aligned cache lines. The number of transactions it takes is what you pay.

If the 32 lanes of a warp access 32 consecutive 4-byte words, that's 128 contiguous bytes — a small number of transactions, every delivered byte used. This is a **coalesced** access.

If the lanes access addresses scattered across memory, the hardware issues up to one transaction per lane, each fetching a full line to deliver 4 useful bytes. Same instruction count, same loop structure, an order of magnitude more traffic and latency.

This is why data layout dominates GPU performance so often, and why structure-of-arrays beats array-of-structures. With an AoS layout, consecutive lanes reading `particle[i].x` stride by the size of the whole struct; with SoA they read a contiguous `x[]` array.

### Shared memory and banks

Shared memory is banked — typically 32 banks of 4-byte width. Lanes hitting distinct banks are serviced in parallel. Lanes hitting *different addresses in the same bank* serialize. (Lanes reading the *same* address are fine — that's a broadcast, handled in one go.)

The classic trap is a stride that aliases:

```c
__shared__ float tile[32][32];
float v = tile[threadIdx.x][0];   // stride 32 floats -> every lane hits bank 0
                                  // 32-way conflict, fully serialized
```

The standard fix is padding the minor dimension — `tile[32][33]` — so the stride becomes coprime with the bank count and the accesses spread out. Swizzled XOR layouts do the same thing without wasting the storage.

### The rest of the hierarchy

L1 and shared memory sit per-SM. L2 is shared device-wide and is the coherence point between SMs. Beyond that is HBM, hundreds of GB/s to several TB/s, which sounds enormous until you divide it by the compute throughput and get the ridge point — a handful of FLOP per byte on vector units, hundreds with matrix cores.

The number worth internalizing: a GPU can perform arithmetic *far* faster than it can be fed. Most kernels are memory-bound, and most GPU optimization is therefore data-movement optimization rather than instruction-count optimization.

## Matrix cores

Dense linear algebra was a large enough fraction of the workload to justify dedicated hardware. Tensor cores (NVIDIA) and matrix cores (AMD) execute a small matrix multiply-accumulate — `D = A × B + C` on fixed tile shapes — as a single warp-scoped instruction.

Three things matter about them in practice:

1. **The operands are warp-distributed.** A fragment is held collectively across the lanes of the warp in a layout the hardware dictates. You don't address it elementwise; you use the provided fragment API, or a library, or you match the layout exactly.
2. **They widen the gap.** Matrix cores raise peak compute by more than an order of magnitude without raising memory bandwidth. The ridge point moves accordingly, so a kernel that was comfortably compute-bound on vector units can be firmly memory-bound on matrix cores. This is the entire reason the tiling hierarchy has to be as deep as it is.
3. **They're mixed-precision by design.** Inputs in a narrow format, accumulation in a wider one. That's what makes the numerics tolerable, and the narrow inputs also halve or quarter the bandwidth cost of feeding them.

## Mapping the programming model

The software abstractions line up with hardware, though not one-to-one:

| Software | Hardware | Notes |
| --- | --- | --- |
| Thread | Lane | one element of a warp |
| Warp / wavefront | Scheduling unit | 32 (NVIDIA) or 32/64 (AMD) |
| Thread block | Resident on one SM | shares LDS and can `__syncthreads()` |
| Grid | Whole device | blocks distributed across SMs |

The guarantees are deliberately weak. A block is guaranteed to be resident on a single SM, which is what makes shared memory and block-wide barriers possible. Blocks, by contrast, are guaranteed almost nothing about each other — no ordering, no concurrency, no safe global barrier in the base model. That freedom is what lets the same binary scale across a laptop part and a datacenter part by varying only the SM count.

Newer architectures add intermediate tiers — thread block clusters on Hopper, letting a group of blocks on neighbouring SMs share memory — but the principle holds: the fewer cross-block guarantees the model makes, the more freely the hardware can schedule.

## What this buys you in practice

When a kernel underperforms, the architecture suggests the questions in rough order of expected payoff:

1. **Is it memory-bound?** Usually yes. Compute arithmetic intensity, compare against the ridge point, and find out before optimizing anything.
2. **Are accesses coalesced?** A layout change often beats every other optimization combined.
3. **Is there enough parallelism in flight?** Not "is occupancy high" but "are there enough independent loads outstanding to cover the latency."
4. **Are warps diverging?** Only matters if the branch varies within a warp.
5. **Are shared memory accesses conflict-free?** Check the stride against the bank count.
6. **Is anything spilling?** Check the register count; spills go to device memory and are silently catastrophic.

All six are downstream of the same original decision. The GPU bet that you would bring it enough independent work to hide the latency it refused to pay to avoid. When a kernel is slow, it is almost always because the bet didn't pay off — not enough parallelism exposed, or the data wasn't arranged so the memory system could deliver it.
