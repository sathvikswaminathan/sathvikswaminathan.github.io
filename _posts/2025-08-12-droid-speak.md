---
layout: post
title: "Prefix sharing in LLM Agents (DroidSpeak)" 
---

*All the figures in this post have been used from the DroidSpeak paper

LLM Agents are all the buzz these days. 

LLM agents orchestrate multiple LLMs, each fine-tuned on different datasets to serve distinct tasks and collaborate on solving complex use cases. These models generally share the same foundational architecture (ex: LLAMA-3.1 8B), although system designs that allow different LLMs to originate from different foundational models are not uncommon.

An example of this would be a user interacting with a chatbot to generate code based on certain requirements. The chatbot invokes the coding agent, an LLM model fine-tuned on coding datatsets, to generate the code and later invokes the validation agent, an LLM model fine-tuned on code validation datasets, to generate user tests based on the requirements. 

<figure style="display: flex; justify-content: center; gap: 20px; align-items: center;">
  <img src="/assets/images/LLM agents ex.png" style="width: 50%;">
</figure>


Before getting into more details, here is a quick overview of how prefix-sharing works for a single LLM model.

## Prefix Sharing for a single LLM model

The memory requirements for running LLM inference mainly stems from two data structures:

- Weights (fixed; ~15 GB for LLAMA-3.1 8B, ~140 GB for LLAMA-3.1 70B)
- KV cache (dynamic; grows with batch size and context length)

At higher batch sizes, the memory requirements go beyond the VRAM capacity of a single H100 GPU, which supports 80 GB of high bandwidth memory. For this reason, a single model is paralleized across multiple GPUs, either using model parallelism or data parallelism, to keep up with the memory requirements.

During the prefill phase of one request, the whole input prompt is processed in parallel to generate the KV cache. A hash is computed for this context and stored. Later, when another GPU node processes a different request, it checks if the hash matches any previously stored contexts and identifies which GPU node holds the corresponding context. It then fetches the KV cache from this GPU and reuses it, elimating the KV cache generation step. This approach significantly helps reduce the time-to-first-token (TTFT) and works well because the weights used across different requests are the same, allowing the context to be processed by the second request exactly as if it were recomputed.

## Prefix Sharing in LLM Agents

LLM Agents deploy a bunch of fine-tuned LLMs on a cluster of GPU nodes. As a result of fine-tuning, these LLMs hold completely different weights, making prefix sharing challenging. As a concrete example, imagine a `llama-3.1-8b-instruct` model sharing the prefix context with a `llama-3.1-storm-8b` model. The prefix context would *not* be processed by the `llama-3.1-storm-8b` model the same way as if it were recomputed, as the prefill phase would have generated a completely different representation.


<figure style="display: flex; justify-content: center; gap: 20px; align-items: center;">
  <img src="/assets/images/naive prefix sahring llm agents.png" style="width: 100%;">
</figure>


The paper, [DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving](https://arxiv.org/abs/2411.02820), addresses this issue.


## DroidSpeak

 As shown in the above figure, naively using the whole KV cache results in a huge accuracy loss. The authors of the paper have studied the sensitivity of the accuracy loss to specific layers of reusing the KV and recomputing the KV cache for the rest of the layers. They show the accuracy comparison (F1 score) for `llama-3-8B-sft-lora-ultrachat` model reusing the KV cache from the `fingpt-llama-3-8B` model against no KV cache reuse at all.

<figure style="display: flex; justify-content: center; gap: 20px; align-items: center;">
  <img src="/assets/images/critical layers F1.png" style="width: 100%;">
</figure>

The accuracy drop is only sensitive to a few specific layers that reuse the KV cache and stays minimal for the reamining layers. These layers are referred to as *critical layers* in the paper, accounting only for $11%$ of the total layers. While the critical layers for every dataset, the authors have shown that the variance in accuracy score only varies for the critical layers while remaining roughly the same across non-critical layers across datasets.


### Recomputing KV cache for critical layers

While it is easy enough to reuse the KV cache for non-critical layers, recomputing the KV cache for critical layers remains challenging since the intiial embedding vector for the critical layer needs to be computed from prior layers to compute the KV for the critical layer, which defeats the whole purpose of prefix caching.

The authors propose using the embedding vector for the critical layers from the sender model. Even this introduces an accuracy loss since the embedding vector is computed by the sender model. This problem gets worse if there are non-contigous critical layers, as multiple embedding vectors need to be sent by the sender model. To get around this, the authors pick *all* the layers between critical layers and recompute the KV cache. 

For example, if the critical layers have been identified as layers 16–18, 20, and 25–27, as opposed to transmitting the embedding vector for layers 16, 20, and 25, only the embedding vector for layer 16 is used and the KV cache is recomputed for layers 16-27.


<figure style="display: flex; justify-content: center; gap: 20px; align-items: center;">
  <img src="/assets/images/contigous compute error.png" style="width: 70%;">
</figure>

As an additional optimization, the embedidng vectors for the critical layer is sent along with the KV cache for the non-critical layers in parallel to overlap recompute of the KV cache with the KV cache transfer, impoving the overall throughput.


<figure style="display: flex; justify-content: center; gap: 20px; align-items: center;">
  <img src="/assets/images/droidspeak_results.png" style="width: 100%;">
</figure>

The above results are for models based on the `mistral` architecture. DroidSpeak achieves nearly the same accuracy as full prefill computation, while also reducing time to first token (TTFT).