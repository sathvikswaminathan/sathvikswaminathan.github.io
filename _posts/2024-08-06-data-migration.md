---
layout: post
title: "Memory Bandwidth & Migration Performance"
---

In today's rapidly evolving landscape of system architecture for efficient AI / LLM inference and the growing adoption of CXL-attached memory devices, high-performant data migration is in more demand than ever. The rapid growth of LLMs is atrributed to using larger weights and longer context windows, which have led to signficiant memory bandwidth and capacity demands. CPUs, GPUs, and accelerators have limited on chip high bandwidth memory and data migration helps offer the illusion of a much larger high bandwidth memory capacity to the application. So much so, that NVIDIA invested in NVLink, a high-speed interconnect, to migrate data between NVIDIA GPUs and CPUs, and Intel invested in the Data Streaming Accelerator, DSA, a data migration accelerator amongst other things. 

<figure style="display: flex; justify-content: center; gap: 20px; align-items: center;">
  <img src="/assets/images/pcie_nvlink_bandwidth_growth.svg">
  <!-- <img src="/assets/images/DSA_perf.png" style="width:50%;" alt="DSA performance"> -->
</figure>

<!-- <figure style="display: flex; justify-content: center; gap: 20px; align-items: center;">
  <img src="/assets/images/pcie_nvlink_bandwidth_growth.svg">
</figure>


<figure style="display: flex; justify-content: center; gap: 20px; align-items: center;">
  <img src="/assets/images/DSA_perf.png">
</figure> -->

Below are a few scenarios where performance of data migration becomes critical:

- Tiered Memory Systems (HBM, CXL-attached memory)
- CPU-GPU hybrid systems
- GPU clusters

The overarching goal of data migration is to ensure data is available in the appropriate memory module in time, preventing compute resources from idling while waiting for the data.

In this post, I will review memory bandwidth and data migration performance on the Intel Sapphire Rapids (SPR) architecture, because that is the only system I have access to. 

The SPR system is a dual-socket system, with each socket supporting 56 physical cores, 64 GB on-chip High Bandwidth Memory and 1 TB of DDR DRAM. Overall, the system is also equipped with 8 DSAs.
<figure>
  <img src="/assets/images/SPR%20architecture.png" alt="SPR architecture">
</figure>

I used the Intel MLC tool to compute the maximum bandwidth across each of these memory nodes

<figure>
  <img src="/assets/images/SPR_BW_mlc.png">
</figure>

As expected, with the cores on the same socket, the bandwidth to HBM is much higher than that of DRAM. However the bandwidth between remote DRAM and HBM nodes take a significant hit because inter-socket communciation on SPR systems is handled by the Intel Ultra Path Interconnect (UPI) protocol which offers only 64 GB/sec in each direction, making it the bottleneck in inter-socket communications. For this reason, inter socket communications are generally discouraged for bandwidth-sensitive applications. 

The numbers presented above when the CPU keeps bombarding memory requests with minimal delay between each to keep saturating the memory subsystem and extracting the maximum bandwidth possible. However, real workloads often fetch data, process the data and then issue a memory request to fetch data for the next computation. What would the memory bandwidth look like when there is a longer delay between these requests?

<figure>
  <img src="/assets/images/local_hbm_dram_bandwidth_vs_delay.png">
</figure>

The memory bandwidth goes down with increasing delay between each request. The memory bandwidth that can be extracted by a workload does not just depend on the memory subsystem but also the acess pattern of the workload and the available compute power to compute the workload's operations. For example, the LLAMA-3.1 8B model with a batch size of 2 and a context length of 8K running on all 56 cores in a single socket, could only extract a peak bandwidth of 306 GB/sec from the local HBM.

In tiered memory systems like SPR, it becomes crucial to migrate data from the slow tier (DRAM) to the fast tier (HBM) to ensure that DRAM simply acts as a capacity tier and most of the memory requests are serviced by HBM, allowing bandwidth-sensitive applications like LLM inference workloads to achieve the best performance possible. 

The peak migration bandwidth between HBM and DRAM is limited by the smaller of the two peak bandwidths because migration speed cannot exceed the bandwidth of the slowest memory in the path.

The peak migration bandwidth is given by  

$$
\min \left( \text{Peak Bandwidth}_{\text{HBM}}, \ \text{Peak Bandwidth}_{\text{DRAM}} \right)
$$

This would be around $$200 GB/sec$$ on my system.

Let's look at how `memcpy` performs on this system

<figure>
  <img src="/assets/images/memcpy_bw_plot.png">
</figure>

The above plot illustrates the migration bandwidth acheived when `memcpy` tries to move data from DRAM to HBM for different data sizes split across different number of cores in parallel. As the data size grows, it makes sense to parallelize the `memcpy` call across cores to saturate the memory bandwidth. Of course, using all the cores to just perform data migration does not make sense as it results in a high tax. 

<figure>
  <img src="/assets/images/dsa_bw_plot.png">
</figure>

This is where Intel DSA enters the picture. 

The CPU cores simply submit job descriptors to Intel DSA to migrate data between two different memory locations and DSA handles the rest. DSA is also equipped with better interconnects to migrate data, resulting in higher memory migration bandwidth while freeing up the CPU to run the actual workload.

However, simply copying the data is not enough. When data gets copied from DRAM to HBM, the `memcpy` or the DSA copy call assigns a new buffer in the destination, resulting in a new virtual address. The application can either be modified to access the new virtual address or the kernel can take it upon itself to swap the virtual pointer to point to the new physical pages in HBM. This is exactly what the system call `move_pages` does. `move_pages` internally uses `memcpy` to copy data, but ongoing work aims to enable it to utilize DSA.

<figure>
  <img src="/assets/images/move_pages_bw.png">
</figure>

As shown, the migration bandwidth achieved by `move_pages` is much lower than the copy calls, as it performs additional steps of swapping pointers, performing TLB shootdowns to invalidate stale TLB mappings, etc. 

It boils down to a tradeoff of either manually changing the application code to access a new virtual address after every migration or having the convenience to use a generic migration call with a performance penalty.

In the era of LLM inference, where every layer of the transformer model executes in less than a millisecond, efficient data migration is the need of the hour.