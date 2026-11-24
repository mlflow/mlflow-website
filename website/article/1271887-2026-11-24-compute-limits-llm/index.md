---
title: "Compute Limits for LLMs: Memory, Throughput and Capacity"
description: "Learn which compute limits cap LLM training and inference, how to size GPU memory and capacity, and how to trace real usage so estimates match production."
slug: compute-limits-llm
tags: [compute limits llm]
date: 2026-11-24
image: https://media.babylovegrowth.ai/blog-images/organization-30814/1795372264348_GPU-servers-installed-in-a-data-center.jpeg
---

![GPU servers installed in a data center](https://media.babylovegrowth.ai/blog-images/organization-30814/1795372264348_GPU-servers-installed-in-a-data-center.jpeg)

An LLM hits a compute limit in one of three places: arithmetic (FLOPs), memory (capacity and bandwidth), or communication between chips. Which one binds depends on the phase of the workload, the model, the context length and the batch size. The estimates you get from calculators are a ceiling. Your production traces tell you where you actually sit. We built MLflow to give teams that measurement: open-source tracing and evaluation that show latency, tokens and cost per request, so capacity decisions rest on your own traffic.

## TL;DR

- Training is bounded by compute and memory. Inference is two phases: prefill is compute-bound, decode is memory-bandwidth-bound.
- VRAM needed is roughly weights plus KV cache plus runtime overhead. Context length and concurrency grow the KV cache fast.
- Size GPUs from peak tokens per second at your latency target, with headroom, not from peak benchmark throughput.
- Real systems land below roofline estimates, so measure with your own prompt and output lengths.
- MLflow tracing links each request's latency and token use to evaluation results.

## Table of Contents

- [What a compute limit actually is](#what-a-compute-limit-actually-is)
- [Why inference has two different limits](#why-inference-has-two-different-limits)
- [Sizing GPU memory](#sizing-gpu-memory)
- [Estimating how many GPUs you need](#estimating-how-many-gpus-you-need)
- [Why real performance falls below the estimate](#why-real-performance-falls-below-the-estimate)
- [Putting measurement into your workflow](#putting-measurement-into-your-workflow)
- [Where MLflow fits](#where-mlflow-fits)
- [Our perspective](#our-perspective)
- [FAQ](#faq)
- [Sources](#sources)

## What a compute limit actually is

Four resources constrain an LLM: FLOPs (math throughput), memory capacity, memory bandwidth, and interconnect bandwidth between chips.

For training a dense model with N parameters on D tokens, a common rule of thumb is [C ≈ 6ND FLOPs](https://longformexplainers.com/llm-pretraining/06-compute-and-memory), about 2ND for the forward pass and 4ND for the backward pass. The rule omits attention cost that grows with context length, so it drifts for long sequences. A [verification across six model sizes](https://www.neuralialabs.com/en/learn/ai-from-scratch/llm-pretraining-scaling-laws-cost) found measured FLOPs 3% to 17% above 6ND once embeddings were counted.

Training memory is heavier than weights alone. It must hold [inputs, outputs, weights, gradients, optimizer states and activations](https://arxiv.org/html/2506.15006v3), scaling linearly with batch size and quadratically with sequence length. Mixed-precision Adam needs [16 bytes per parameter](https://www.chipfieldguide.com/architectures/workloads/) for weights, gradients and optimizer states, so a 1.5B model needs at least 24 GB before anything else.

Across chips, Google's TPU documentation describes the goal of [strong scaling](https://docs.cloud.google.com/tpu/docs/scaling-model-on-tpus): adding chips and getting proportional throughput. When communication takes longer than computation, the model becomes communication-bound and stops scaling well.

![What a compute limit actually is](https://media.babylovegrowth.ai/blog-images/organization-30814/1795372333520_What-a-compute-limit-actually-is.jpeg)

## Why inference has two different limits

Inference splits into prefill (processing the prompt) and decode (generating tokens one at a time).

- **Prefill** is compute-bound. [Llama-2-7B with a 2,048-token prompt](https://www.chipfieldguide.com/architectures/workloads/) runs at about 1,024 ops per byte, and this phase sets time to first token (TTFT).
- **Decode** at small batch runs at about 1 FLOP per byte, deeply memory-bound. Each token requires reading the weights and KV cache from memory.

That is why [batch-one decode is bandwidth-bound](https://slavadubrov.com/blog/2026/03/13/llm-engineering-guide/) while larger batches share weight reads and raise throughput, until latency, memory, compute or communication becomes the cap. A chip rated for 500 trillion ops per second can [deliver a tenth of that](https://jax-ml.github.io/scaling-book/) if it spends its time moving parameters.

At the extreme, an [NVIDIA Research limit study](https://arxiv.org/html/2507.14397v1) found 128-chip HBM3 systems can reach 600+ user tokens per second on large models, but none reach 1,000 at large context, and synchronization latency needs to be around 1 microsecond to keep memory bandwidth effective.

## Sizing GPU memory

A workable formula from [FastGPU's 2026 guide](https://fastgpu.co/blog/llm-vram-guide): total VRAM ≈ weights + KV cache + activations and runtime overhead.

### Weights

Weights equal parameters times bytes per parameter: about 2 GB per billion parameters at 16-bit, 1 GB at 8-bit, and roughly 0.5 to 0.6 GB at 4-bit. A [7B model](https://slavadubrov.com/blog/2026/03/13/llm-engineering-guide/) drops from about 14 GB at FP16/BF16 to about 7 GB at FP8/INT8 and 3.5 GB at INT4. For mixture-of-experts models, total parameters set memory, not active ones: Qwen3-30B-A3B needs 61.1 GB at 16-bit despite activating 3.3B parameters.

![7B model weight memory at three precisions](https://media.babylovegrowth.ai/blog-images/organization-30814/1795372308868_7B-model-weight-memory-at-three-precisions.jpeg)

### KV cache

The KV cache is 2 × layers × KV heads × head dimension × bytes per value × tokens × sequences. It grows linearly with context and with concurrency. Llama 3.1 8B at 128K tokens uses [16 GiB at batch 1](https://slavadubrov.com/blog/2026/03/13/llm-engineering-guide/), and Llama 3.1 70B at 128K needs about 40 GiB, roughly half an H100's 80 GB.

### A worked 24 GiB example

A [memory budget card](https://llmsresearch.com/cards/gpu-memory-budget) works through an 8B BF16 model on a 24 GiB card:

| Component                           | Size                    |
| ----------------------------------- | ----------------------- |
| Weights (8B, BF16)                  | ~14.9 GiB               |
| KV cache (8,192 tokens, 4 requests) | 4 GiB                   |
| Runtime overhead                    | ~2 GiB                  |
| Total                               | ~20.9 GiB, 3.1 GiB free |

At 8 concurrent requests the cache doubles to 8 GiB and the total of about 24.9 GiB no longer fits. The model did not change. Concurrency was the limit.

At the large end, serving Llama3-405B or DeepSeekV3 at 64K+ context needs [300 to 600 GB for one user](https://arxiv.org/html/2507.14397v1), and a 32-user, 128K-context deployment reaches 1.4 TB.

## Estimating how many GPUs you need

The [capacity-planning formulas](https://slavadubrov.com/blog/2026/03/13/llm-engineering-guide/) are simple:

1. Required GPUs = (peak tokens per second × safety multiplier) / per-GPU tokens per second at your target SLO. A 1.3× multiplier gives 30% headroom.
2. Max concurrent sequences = (GPU memory − weights − overhead) / per-sequence KV cache.
3. Use throughput measured at the target latency, not peak throughput.
4. Benchmark with your real prompt and output length distribution at your required TTFT and TPOT (time per output token) thresholds.

For reference, MLPerf's interactive Llama 2 70B scenario uses P99 TTFT of 450 ms or less and P99 TPOT of 40 ms or less. Your own SLOs may differ, but they must be written down before the math means anything.

Step 4 is where most estimates fail, because it needs real data. Teams that trace every request see the true distribution of input tokens, output tokens and latency instead of guessing. We cover that workflow in [MLflow for Engineers: trace LLM requests and link cost to eval](https://mlflow.org/articles/llm-request-tracing).

## Why real performance falls below the estimate

Calculators give best-case numbers. A roofline is [a limit real kernels never reach](https://ishwarj.com/writings/llm-inference-6-serving-in-production/). The JAX scaling book shows [bounds of max(T_math, T_comms) and T_math + T_comms](https://jax-ml.github.io/scaling-book/roofline/), which differ by at most 2× with perfect overlap, and says accuracy comes from profiling the specific model and system.

Three practical consequences:

- Engine version, kernels and traffic mix all move your numbers. Re-measure after upgrades.
- Keep spare replicas so losing one does not break the SLO.
- Do not autoscale on GPU utilization alone: it stays high in both healthy load and overload. Combine it with [queue depth, KV cache utilization and goodput](https://slavadubrov.com/blog/2026/03/13/llm-engineering-guide/), and tune thresholds from load tests.

Inference efficiency is also low compared with training. [Sardana et al.](https://arxiv.org/pdf/2401.00448v3.pdf) report model FLOPs utilization as low as about 1% for inference versus typically 40 to 60% in training. That gap is why serving cost deserves its own planning.

## Putting measurement into your workflow

A repeatable routine for an AI ops team:

1. Trace every request with token counts and latency, split by TTFT and total time.
2. Compute percentiles (P50, P95, P99) by route, model and prompt version.
3. Feed the observed length distribution into the capacity formulas above.
4. Run evaluations on the same traffic so a cheaper quantization or shorter context is judged on quality as well as cost.
5. Alert on queue depth and KV cache pressure, not only utilization.

MLflow tracing records agentic reasoning step by step, so you can see which calls in an agent loop consume tokens and time. Our [LLM-as-a-Judge evaluation](https://mlflow.org/llm-as-a-judge) runs automated quality checks on those same traces. For further reading, browse our posts on [LLM performance monitoring](https://mlflow.org/articles/tags/llm-performance-monitoring) and [efficient LLM memory usage](https://mlflow.org/articles/tags/efficient-llm-memory-usage).

## Where MLflow fits

MLflow is an open-source platform for GenAI and LLM lifecycle management: observability with deep tracing, automated LLM-based evaluation, prompt management and versioning, a cross-provider AI Gateway and agent server deployment. It works through OpenTelemetry and integrates with major AI and ML frameworks. You can start with the [Agent and LLM engineering overview](https://mlflow.org/genai) or the [AI platform guide](https://mlflow.org/ai-platform).

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

## Our perspective

> Most compute-limit conversations stop at the calculator. We think that is the wrong end of the problem. The formulas are sound, and every team should know them, but the inputs are the hard part: real prompt lengths, real output lengths, real concurrency, real agent loops that fan out into many model calls.
>
> That is why we treat tracing as the foundation. Once each request carries its tokens, latency and cost, capacity planning becomes arithmetic on your own data. Evaluation then closes the loop, so a quantization or routing change is accepted because quality held, not because the benchmark looked good.
>
> We also believe this should be open. MLflow ships all of these features in open source under Linux Foundation governance, with no enterprise feature paywalls, so any team can measure before it buys hardware.
>
> _Kevin_

## FAQ

### How much GPU memory do I need to run an LLM?

Add weights, KV cache and overhead. Weights take about 2 GB per billion parameters at 16-bit and about half that at 8-bit. The KV cache then grows with context length and the number of concurrent requests, so size for your peak concurrency, not a single user.

### Is LLM inference compute-bound or memory-bound?

Both, in different phases. Prefill is compute-bound and sets time to first token. Decode at small batch is memory-bandwidth-bound. Batching improves decode efficiency until KV cache traffic dominates.

### Why is my real throughput lower than the calculator says?

Calculators model best-case limits. Real kernels, engine versions, traffic mix and communication overhead all reduce throughput. Profile your own model on your own system with your actual prompt and output lengths.

### How much headroom should I plan for?

A common starting point is a 1.3× multiplier on peak tokens per second, which gives 30% headroom, plus spare replicas so a single failure does not break your latency target. Tune from load tests on your own traffic.

## Sources

- [Compute & memory, Long Form Explainers](https://longformexplainers.com/llm-pretraining/06-compute-and-memory)
- [Pretraining an LLM: Data, Compute, Scaling Laws and Cost](https://www.neuralialabs.com/en/learn/ai-from-scratch/llm-pretraining-scaling-laws-cost)
- [Scaling Intelligence: Designing Data Centers for Next-Gen LLMs](https://arxiv.org/html/2506.15006v3)
- [AI workloads: matrix math, attention and the roofline model](https://www.chipfieldguide.com/architectures/workloads/)
- [Scale a model on TPUs](https://docs.cloud.google.com/tpu/docs/scaling-model-on-tpus)
- [LLM Engineering Guide: 45 Concepts for Production Systems](https://slavadubrov.com/blog/2026/03/13/llm-engineering-guide/)
- [How To Scale Your Model](https://jax-ml.github.io/scaling-book/)
- [Efficient LLM Inference: Bandwidth, Compute, Synchronization](https://arxiv.org/html/2507.14397v1)
- [How much VRAM do you need to run an LLM?](https://fastgpu.co/blog/llm-vram-guide)
- [LLM GPU Memory: Weights, KV Cache and Runtime Overhead](https://llmsresearch.com/cards/gpu-memory-budget)
- [All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/)
- [Serving LLMs in Production](https://ishwarj.com/writings/llm-inference-6-serving-in-production/)
- [Beyond Chinchilla-Optimal](https://arxiv.org/pdf/2401.00448v3.pdf)
