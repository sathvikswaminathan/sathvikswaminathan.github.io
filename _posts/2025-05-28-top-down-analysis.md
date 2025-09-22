---
layout: post
title: "Detect Performance Bottlenecks with Top-Down Analysis"
---
Figures and metrics in this post are based on concepts from the paper:

> Ahmad Yasin, "A Top-Down Method for Performance Analysis and Counters Architecture," *2014 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)*, 2014, pp. 35-44.  
> [https://doi.org/10.1109/ISPASS.2014.6844459](https://doi.org/10.1109/ISPASS.2014.6844459)


Top-Down Analysis is a methodology to identify performance bottlenecks. It starts by getting a 10,000-foot view of which subsystem the application spends the most time in. Throughout the application's lifetime, it spends time in one of the following subsystems:

- CPU Frontend (instruction decoding)

- CPU Backend (cores, memory)

- Retiring instructions (useful work)

- Bad speculation (correcting wrong speculation results)

Once the dominant category has been identified, we focus on it by drilling further to pinpoint the performance bottleneck.

The Top-Down Analysis hierarchy is depicted below:

<figure>
  <img src="/assets/images/tma-heir.png" alt="SPR architecture">
</figure>

Linux `perf` leverages hardware performance counters to facilitate the top-down analysis methodology.

In this post, we will focus on identifying bottlenecks for a simple LLM inference application using the LLAMA-3.1 8B model with an input length of 512, output length of 256, and a batch size of 2 on an Intel SPR system.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")  

batch_prompts = [
    "Bangalore's road infrastructure is wonderful",
    "Sumo deadlift is not cheating"
]

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

input_ids = tokenizer(
    batch_prompts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
).input_ids

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=False  
)

for i in range(outputs.shape[0]):
    gen_text = tokenizer.decode(outputs[i][input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"Sample {i+1}: {gen_text}")
```

As a side note, `LLAMA-3.1 8B` needs to be trained on better data as it believes Bangalore's road infrastructure is indeed wonderful :)

---

Run:

``` perf stat python llm_ex.py  ```

It reports the following Topdown stats:

```
| Metric            | Percentage | Description              |
|-------------------|-----------:|--------------------------|
| TopdownL1         |      88.8% | tma_backend_bound        |
|                   |       1.2% | tma_bad_speculation      |
|                   |       4.3% | tma_frontend_bound       |
|                   |       5.7% | tma_retiring             |
```

The report shows that, in Topdown L1, 88.8% of the pipeline slots are stalled waiting for resources from the backend. This could be due to waiting for the execution units to free up, data to be fetched from memory, etc. As a result, very few slots are actually being used to retire the instructions.

We now drill further into the `backend-bound` category, and look at the TopdownL2 stats for `core-bound` and `memory-bound`

```
| Metric            | Percentage | Description              |
|-------------------|-----------:|--------------------------|
| TopdownL2         |       0.5% | tma_branch_mispredicts   |
|                   |      13.3% | tma_core_bound           |
|                   |       1.7% | tma_fetch_bandwidth      |
|                   |       2.6% | tma_fetch_latency        |
|                   |       1.3% | tma_heavy_operations     |
|                   |       4.4% | tma_light_operations     |
|                   |       0.7% | tma_machine_clears       |
|                   |      75.5% | tma_memory_bound         |
```

As shown, the 75.5% of the slots stall waiting for data from the memory subsystem (caches, DRAM, etc.) and only 13.3% of the slots stall due to limited compute resources, making the inference workload `memory-bound`.

Now, we can drill down into the `memory-bound` group by running:

```
perf stat -M tma_memory_bound_group python llm_ex.py

| Metric                 | Percentage |
|------------------------|------------|
| tma_dram_bound         | 66.5 %     |
| tma_l3_bound           | 5.0 %      |
| tma_l2_bound           | 0.2 %      |
| tma_l1_bound           | 2.1 %      |
| tma_store_bound        | 0.4 %      |

```

Drilling down further into `tma_dram_bound`:

```
perf stat -M tma_dram_bound_group python llm_ex.py 

| Metric           | Percentage |
|------------------|------------|
| tma_mem_bandwidth | 72.6 %     |
| tma_mem_latency   | 4.3 %      |
```

`tma_mem_bandwidth` is the longest, meaning the inference workload primarily stalls waiting for data to be returned from DRAM due to insufficent bandwidth.

`perf record` can be used to identify where in the workload the performance bottleneck is. Since our workload is DRAM bound, we can check for where in the workload are most of DRAM requests being issued from:

```
perf record -e  mem_load_l3_miss_retired.local_dram  python llm_ex.py; perf report

| Overhead | Command | Shared Object          | Symbol                                                      |
|----------|---------|-----------------------|-------------------------------------------------------------|
| 99.73%   | python  | libmkl_avx512.so.2    | [.] sgemm_t_n23_b0                                          |
| 0.15%    | python  | libtorch_cpu.so       | [.] void c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<a |
| 0.02%    | python  | [kernel.kallsyms]     | [k] asm_exc_page_fault                                      |
| 0.02%    | python  | libmkl_avx512.so.2    | [.] mkl_blas_avx512_sgemm_kernel_nocopy_TN_b1               |
| 0.01%    | python  | libtorch_cpu.so       | [.] void c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<a |
| 0.01%    | python  | libtorch_cpu.so       | [.] void at::internal::invoke_parallel<at::parallel_for<at::native::AVX2::reduced_float_copy_kernel(at::TensorIterat |
| 0.01%    | python  | libmkl_avx512.so.2    | [.] mkl_blas_avx512_sgemm_kernel_nocopy_TN_b0               |
| 0.00%    | python  | [kernel.kallsyms]     | [k] asm_sysvec_apic_timer_interrupt                         |
| 0.00%    | python  | [kernel.kallsyms]     | [k] update_curr                                             |
```

The function `sgemm_t_n23_b0 ` from the `libmkl_avx512.so.2 ` library issues most of the DRAM requests. `perf` lets you zoom into the assembly of these functions

```
| Percent | Instruction                                                                                     |
|---------|------------------------------------------------------------------------------------------------|
| 0.20%   | lea           (%r11,%rbp,1),%r13                                                              |
| 0.15%   | lea           (%r14,%r13,1),%r12                                                              |
| 0.16%   | add           $0xffffffffffffff40,%r12                                                        |
| 0.18%   | vmovups       (%r14,%r12,1),%zmm15                                                            |
| 8.42%   | vmovups       -0xc0(%r15,%r11,1),%zmm17                                                       |
| 0.57%   | vmovups       -0xc0(%rbp,%r11,1),%zmm16                                                       |
| 5.18%   | vmovups       -0xc0(%rax,%r11,1),%zmm18                                                       |
| 0.46%   | add           %r14,%r12                                                                       |
| 0.10%   | vmovups       -0xc0(%r14,%r13,1),%zmm19                                                       |
| 6.92%   | vmovups       (%r14,%r12,1),%zmm20                                                            |
| 5.79%   | add           %r14,%r12                                                                       |
```

The `vmovups` instructions load data from memory into SIMD registers and consume a lot of time.


Now that the performance bottleneck has been identified, differnet approaches can be experimented to lowering the loading time for the `vmovups` instructions such as calling the `__builtin_prefetch` function at the appropriate time to prefetch the relevant data into the cache.

