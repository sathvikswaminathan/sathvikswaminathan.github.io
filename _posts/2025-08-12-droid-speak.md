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
  <img src="/assets/images/naive prefix sahring llm agents.png" style="width: 80%;">
</figure>


The paper, [DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving](https://arxiv.org/abs/2411.02820), addresses this issue.





