---
layout: post
title: "Data Migration"
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

In this post, I will review the data migration performance on the Intel Sapphire Rapids (SPR) architecture, because that is the only system I have access to. 

The SPR system is a dual-socket system, with each socket supporting 56 physical cores, 64 GB on-chip High Bandwidth Memory and 1 TB of DDR DRAM.
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

In tiered memory systems like SPR, it becomes crucial to migrate data from the slow tier (DRAM) to the fast tier (HBM) to ensure that DRAM simply acts as a capcity tier and most of the memory requests are serviced by HBM, allowing bandwidth-sensitive applications like LLM inference workloads to achieve the best performance possible. 

The peak migration bandwidth between HBM and DRAM is limited by the smaller of the two peak bandwidths because migration speed cannot exceed the bandwidth of the slowest memory in the path.

The peak migration bandwidth is given by  

$$
\min \left( \text{Peak Bandwidth}_{\text{HBM}}, \ \text{Peak Bandwidth}_{\text{DRAM}} \right)
$$



