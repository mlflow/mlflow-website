---
title: "Engineers: Tie p99 SLOs to Token Budgets with MLflow for LLM Latency"
description: "Ops first playbook for engineers: set p99-driven SLOs, profile token budgets, and use MLflow tracing to enforce and validate LLM latency targets."
slug: latency-budgets-llm
tags:
  [
    latency optimization LLM,
    how to manage latency in LLM,
    LLM delay considerations,
    AI latency management,
    latency trade-offs,
    latency budgets llm,
    budgeting for LLM latency,
    effective LLM throughput,
    reducing LLM latency,
    LLM response times,
    latency analysis for AI,
    LLM performance metrics,
  ]
date: 2026-10-01
image: https://media.babylovegrowth.ai/blog-images/organization-30814/1790825630850_Engineers-reviewing-LLM-latency-measurements.jpeg
---

![Engineers reviewing LLM latency measurements](https://media.babylovegrowth.ai/blog-images/organization-30814/1790825630850_Engineers-reviewing-LLM-latency-measurements.jpeg)

A latency budget for an LLM system is the end-to-end time objective you allocate across every measurable hop a request takes, from client to model and back. The first action is always the same: measure end-to-end percentiles (p50, p95, p99) and map the real request path using shared timestamps. Once that baseline exists, tools like MLflow can help operationalize the budget.

---

> **TL;DR:**
>
> - A latency budget should be broken down into individual hops like network, retrieval, and decode to identify and address specific bottlenecks.
> - Consistent instrumentation across services using shared timestamps and trace IDs is essential for enforcing and tracking latency budgets effectively.
> - Profiling token budgets through representative tests helps optimize prefill and decode allocations, preventing waste and ensuring SLAs are met.
> - Caching, model routing, batching strategies, and streaming responses are among the most effective tactics for reducing latency in production environments.
> - Proper handling of variability and multi-tenant effects requires explicit tail latency budgeting, including retries and fallback mechanisms, to maintain end-to-end SLAs.

---

## Table of Contents

- [Core latency metrics you need to measure](#core-latency-metrics-you-need-to-measure)
- [Mapping the componentized latency budget](#mapping-the-componentized-latency-budget)
- [How to instrument latency budgets so they are enforceable](#how-to-instrument-latency-budgets-so-they-are-enforceable)
- [Setting token budgets and allocating percentile targets](#setting-token-budgets-and-allocating-percentile-targets)
- [Optimization tactics you can apply now](#optimization-tactics-you-can-apply-now)
- [Testing and verification under realistic load](#testing-and-verification-under-realistic-load)
- [Operationalizing latency budgets with MLflow](#operationalizing-latency-budgets-with-mlflow)
- [Hardware and infrastructure choices shape the budget](#hardware-and-infrastructure-choices-shape-the-budget)
- [Handling variability in network and compute latency](#handling-variability-in-network-and-compute-latency)
- [Model size, complexity, and the latency trade-off](#model-size-complexity-and-the-latency-trade-off)
- [Case studies in latency budget breakdowns](#case-studies-in-latency-budget-breakdowns)
- [Latency budgets in multi-tenant and cloud environments](#latency-budgets-in-multi-tenant-and-cloud-environments)
- [Batch size, concurrency, and the throughput balance](#batch-size-concurrency-and-the-throughput-balance)
- [What I would fix first](#what-i-would-fix-first)
- [Getting started with MLflow for latency budgets](#getting-started-with-mlflow-for-latency-budgets)
- [Key standards and benchmarks to follow](#key-standards-and-benchmarks-to-follow)
- [Sources](#sources)
- [FAQ](#faq)

## Core latency metrics you need to measure

Before you can allocate a budget, you need a shared vocabulary for what you are measuring. Time to first token (TTFT) covers the delay between sending a request and receiving the first token back, and it typically bundles queuing, prefill, and network transit. Inter-token latency (ITL), sometimes called time between tokens (TBT), measures the gap between successive tokens during generation. Generation time is the full decode phase duration, and end-to-end (e2e) latency is the sum of TTFT plus generation time. Throughput is usually tracked as tokens per second, either per request or aggregated across concurrent requests.

Benchmarking tools do not agree on conventions. [NVIDIA's genAI-perf metrics documentation](https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/metrics.html) defines TTFT as inclusive of queuing, prefill, and network latency, and notes that longer prompts increase TTFT because of KV-cache construction. Other suites, like LLMPerf, draw the line between TTFT and ITL differently. That inconsistency matters because your SLOs inherit whichever convention you pick.

For interactive chat workloads, track p95 and p99 TTFT closely since users notice stalls immediately. For background or batch workloads, median (p50) e2e latency and throughput matter more than tail behavior.

- TTFT: queuing plus prefill plus network transit to the first token.
- ITL/TBT: the gap between consecutive generated tokens.
- Generation time: full decode duration from first to last token.
- E2e latency: TTFT plus generation time, the number users actually feel.

**Benchmarks define TTFT differently across tools.** NVIDIA's metrics reference bundles queuing and prefill into TTFT, which means a budget built on one tool's numbers may not translate cleanly to another.

## Mapping the componentized latency budget

A single e2e latency number hides where time actually goes. Break the request path into hops and give each one its own budget, because a tail spike in any single hop can blow the entire SLO even when the average looks fine.

- Client render: time spent building the request and rendering the response in the UI.
- Network: round trips between client, gateway, and backend, often the most variable hop.
- Gateway/API layer: authentication, routing, and rate limiting before the request reaches a model.
- Retrieval/RAG: vector search, document fetch, and reranking before the prompt is assembled.
- Queueing/batching: time waiting for a serving slot or batch window.
- Prefill: processing the full input prompt into KV-cache state.
- Decode: autoregressive token generation, one step at a time.
- Post-processing: formatting, moderation checks, or function-call parsing.

Network and client render hops tend to be network-bound and highly variable. Gateway and retrieval hops are usually I/O-bound, dependent on downstream services you may not control directly. Prefill and decode are compute-bound and tied to model size, batch composition, and hardware.

Tail behavior compounds across hops rather than adding neatly. Prefill and decode interactions make this worse: chunked prefill competing with ongoing decode steps on the same batch can stall token generation mid-stream, producing p99 surprises that never show up in average-case testing.

![Tail latency compounding across LLM serving stages](https://media.babylovegrowth.ai/blog-images/organization-30814/1790825726177_Tail-latency-compounding-across-LLM-serving-stages.jpeg)

## How to instrument latency budgets so they are enforceable

A budget only matters if it is measured consistently enough to catch regressions. That starts with tracing conventions that hold across every service in the path.

1. Propagate a shared timestamp or trace ID from the client through the gateway, retrieval layer, and model server so every hop can be stitched into one timeline.
2. Record per-span timings with consistent event names (request_start, retrieval_done, prefill_done, first_token, last_token) so dashboards stay comparable across releases.
3. Store full request-level distributions, not just averages, and tag each by workload class, prompt length, and deployment location so tail latency from one segment does not get averaged away by another.
4. Run inference profilers and periodic batch profiling sweeps to find where token budgets and chunk sizes start degrading decode speed, rather than guessing from production incidents.

**Pro Tip:** _Tag every trace with prompt length bucket as well as workload class. A 200-token support query and a 4,000-token document summary should never share the same p99 target._

## Setting token budgets and allocating percentile targets

Token budgets decide how much of a prompt gets processed per inference step, and the choice trades off latency against throughput in ways that are easy to get backward. Smaller token budgets reduce inter-token latency because less work happens per step, but they increase overhead from chunked prefill and can waste GPU cycles when chunk sizes do not divide evenly across hardware tiles. The [OSDI paper on LLM serving](https://www.usenix.org/system/files/osdi24-agrawal.pdf) documents cases where a non-divisible chunk size caused roughly 32% slower prefill processing due to tile-quantization effects, which makes profiling a requirement rather than a nice-to-have.

The practical approach is a one-time profiling sweep: run representative batches across a range of token budgets and chunk sizes, then find the largest budget that still meets your TBT SLO without tipping GPU utilization into waste. Iteration-level batching systems make this easier to test because they let you adjust prefill and decode prioritization independently.

- Start from your target end-to-end p99, then subtract network, retrieval, and gateway overhead to find the compute budget.
- Carve the remaining budget into prefill and decode allocations with explicit headroom, never a tight fit.
- Validate the allocation under realistic concurrent load, since isolated single-request tests hide batching contention.

**A 32% prefill slowdown can come from chunk size alone.** The OSDI/USENIX paper on LLM serving found that non-divisible chunk sizes interacting with tile quantization produced that penalty in testing, independent of model size or hardware generation.

## Optimization tactics you can apply now

Once you know where the budget is being spent, the fix depends entirely on which hop is the bottleneck. Matching the tactic to the component avoids wasted engineering effort.

- Semantic or prompt caching reduces repeated prefill cost; [Google's Prompt Cache research](https://research.google/pubs/prompt-cache-modular-attention-reuse-for-low-latency-inference/) reports TTFT improvements ranging from 8 times faster on GPU to 60 times faster on CPU in prototype testing, particularly for repeated system prompts and document-based question answering.
- Model routing to smaller or fine-tuned models cuts decode time directly when a task does not need a frontier-scale model.
- Batching strategy matters: decode-prioritizing schedulers protect latency for in-flight generations, while prefill-prioritizing schedulers favor throughput at the cost of occasional generation stalls.
- Streaming responses token by token reduces perceived latency even when total generation time stays the same, since users see progress immediately.
- Speculative decoding, where a smaller proposal model drafts tokens validated by the full model, has produced 2 to 3 times decoding speedups in Google Research's experiments without changing output quality.
- [OpenAI's latency optimization guide](https://developers.openai.com/api/docs/guides/latency-optimization) groups tactics into seven principles: process tokens faster, generate fewer tokens, use fewer input tokens, make fewer requests, parallelize calls, reduce perceived wait through streaming, and skip the LLM entirely when a classical method works.

## Testing and verification under realistic load

Benchmarks that only report averages will miss the failures that matter. Collect full per-request distributions for TTFT, ITL, and e2e latency, along with the failure fraction, and treat p99 under concurrent load as the number that decides whether the budget holds.

1. Run ramp tests that gradually increase concurrency to find where queueing starts dominating latency.
2. Run spike tests that simulate sudden traffic bursts, since tail latency under spikes often looks nothing like tail latency under steady load.
3. Run chaos tests that simulate degraded network conditions or saturated queues to see which component budget breaks first.
4. Separate network, model compute, and retrieval contributions by instrumenting each independently, using NVIDIA's AI-Perf metrics reference as a guide for network-adjusted metrics that isolate model compute from transit time.

## Operationalizing latency budgets with MLflow

Turning a budget from a spreadsheet into an enforced system standard is where most teams stall. Tracing tools capture per-span timings and shared timestamps across services, which gives you the cross-service view the earlier sections describe without building that plumbing from scratch.

- Per-span tracing surfaces prefill, decode, retrieval, and gateway timings on one timeline.
- Experiment tracking and versioning let you rerun token-budget sweeps and compare results reproducibly over time.
- Prompt tracking keeps the prompt templates tied to the latency numbers they produced, so a regression can be traced back to a specific prompt change.

## Hardware and infrastructure choices shape the budget

The hardware under your model sets the ceiling on what any budget allocation can achieve. GPU generation, memory bandwidth, and interconnect speed all affect prefill and decode throughput independently, which means the same token budget behaves differently on different accelerators. Pipeline parallelism and tensor parallelism, used to split large models across multiple devices, introduce their own communication overhead between devices, and that overhead interacts with chunk size choices in ways the OSDI paper on LLM serving ties directly to tile-quantization effects during profiling.

Infrastructure placement matters as much as the accelerator itself. A model server colocated with its retrieval index avoids a network hop that a geographically separated deployment cannot. Autoscaling policies determine how quickly new capacity comes online when queueing starts to dominate latency, and a slow scale-up response shows up first in p99 TTFT during traffic spikes rather than in the median.

None of this means more hardware is always the answer. A latency budget blown by queueing delay often reflects a scheduler or routing misconfiguration rather than insufficient compute, and adding GPUs to a routing problem tends to mask the symptom temporarily while leaving the underlying allocation wrong. The right sequence is to profile first, confirm which hop is actually compute-bound, and only then evaluate whether a hardware change moves the needle. Teams that skip the profiling step frequently find that a scheduler change or a smaller model for a specific route delivers more latency improvement than a hardware upgrade would have.

![Hardware and infrastructure choices shape the budget — overview diagram](https://media.babylovegrowth.ai/blog-images/organization-30814/1790825825725_Hardware-and-infrastructure-choices-shape-the-budget-overview-diagram.jpeg)

## Handling variability in network and compute latency

Network and compute latency rarely behave like a fixed number with small noise around it. Network latency fluctuates with routing changes, congestion, and the physical distance between client and server, and that variability tends to show up as intermittent tail spikes rather than a steady shift in the median. Compute latency varies with batch composition: a request that lands in a batch next to several long-prompt requests will see slower prefill than the same request processed alone, even though nothing about the request itself changed.

The practical response is to stop treating variability as noise to average away and start treating it as a distribution to budget for explicitly. Allocate headroom in each component budget rather than setting tight targets based on median behavior, since a budget with no slack guarantees that any ordinary variance becomes an SLO violation. Tagging traces by deployment location and time of day, as covered earlier in instrumentation, helps separate genuine regressions from expected variability tied to traffic patterns.

Retries and timeouts need their own budget line. A request that times out and retries consumes two hops' worth of latency while only counting as one logical request to the end user, and failing to account for that in your SLO math will make your dashboards look better than the experience actually is. Circuit breakers and fallback routes, triggered when a downstream retrieval service or model endpoint starts showing elevated p95 latency, protect the overall budget from a single degraded dependency cascading into every request.

## Model size, complexity, and the latency trade-off

Larger models generally produce better output quality at the cost of slower prefill and decode, and that trade-off sits at the center of most latency budget decisions. A model with more parameters needs more compute per token, which stretches both TTFT and generation time even before accounting for prompt length or batch contention. Fine-tuned smaller models can close much of the quality gap for narrow tasks while running meaningfully faster, which is why model routing, covered earlier among optimization tactics, is often the highest-leverage lever available.

Model complexity extends beyond parameter count. Architectures with longer context windows carry higher KV-cache construction costs during prefill, which is part of why NVIDIA's genAI-perf documentation notes that longer prompts increase TTFT independent of output length. Mixture-of-experts architectures can reduce average compute per token relative to a dense model of similar total size, but routing overhead between experts adds its own latency contribution that needs separate measurement rather than being folded into a generic "model latency" number.

The budgeting implication is that model selection is not a one-time architectural decision made in isolation from the latency budget. Every component budget set earlier in the process assumes a specific model's prefill and decode characteristics, and swapping models without reprofiling token budgets and chunk sizes risks invalidating the allocation even when the model itself performs better on quality benchmarks.

## Case studies in latency budget breakdowns

Teams that have published token-budget research give a concrete look at where the gains actually come from. The OSDI serving paper documents systems adopting iteration-level batching, including designs similar to Orca and vLLM, specifically to improve throughput while controlling the generation stalls that come from naive prefill-decode scheduling. Those systems show measurable throughput gains from batching smarter, but the paper is equally clear that scheduling choice alone does not eliminate tail latency risk without accompanying token-budget profiling.

On the caching side, Google's Prompt Cache research demonstrates attention-state reuse delivering TTFT improvements from 8 times faster on GPU to 60 times faster on CPU in prototype testing, concentrated in workloads with repeated prompt segments such as document-based question answering. That gap between GPU and CPU improvement is itself a useful budgeting signal: caching strategy benefit depends heavily on where the compute bottleneck already sits.

Speculative decoding offers a third data point. Google Research's speculative execution work measured 2 to 3 times decoding speedups on T5-XXL using a cheap proposal model validated by the full model, with identical output quality preserved. The consistent thread across these results is that no single tactic closes a latency budget gap on its own. Caching helps TTFT, speculative decoding helps decode speed, and batching strategy helps throughput under concurrency, and a real production budget usually needs more than one of them stacked together.

## Latency budgets in multi-tenant and cloud environments

Shared infrastructure changes the math behind every component budget. In a multi-tenant deployment, one tenant's burst of long-context requests can degrade prefill latency for every other tenant sharing the same GPU pool, because batch composition is no longer under any single team's control. Noisy-neighbor effects of this kind rarely show up in single-tenant testing, which is why load tests that simulate realistic multi-tenant traffic mixes matter more than isolated benchmark runs.

Cloud deployments add their own variability on top of this. Autoscaling introduces cold-start latency when new instances spin up to absorb demand, and that cold-start penalty shows up as a TTFT spike that a steady-state benchmark will never catch. Region selection affects network latency directly, and a tenant served from a distant region will see a different baseline budget than one served locally, even when the model and infrastructure are otherwise identical.

Fair allocation policies, such as per-tenant rate limits or priority queues, are often the only practical way to keep one tenant's traffic pattern from consuming another's latency budget. Tagging traces by tenant, as an extension of the workload-class tagging covered in the instrumentation section, lets you attribute a regression to a specific tenant's traffic rather than chasing a system-wide cause that does not exist.

## Batch size, concurrency, and the throughput balance

Batch size sits directly at the tension between latency and throughput, and getting it wrong in either direction costs you something real. Larger batches improve GPU utilization and raise aggregate tokens-per-second throughput, but they also increase the time any individual request within the batch waits for its share of compute, which stretches both TTFT and inter-token latency for that request. Smaller batches protect individual request latency but leave GPU cycles underused, which raises the per-request cost of serving.

Concurrency compounds this. As concurrent request volume rises, queueing delay before a request even enters a batch becomes the dominant latency contributor, often outweighing prefill and decode time combined once load gets high enough. This is precisely where the tail multiplicativity discussed earlier in the component mapping becomes visible in production: a system that looks fine at moderate concurrency can show a sharp p99 cliff once queueing starts dominating.

Iteration-level batching, where the scheduler reassembles the batch at every decode step rather than waiting for a full batch to complete, is one practical answer because it lets new requests join an in-progress batch instead of waiting for the next batch window. The trade-off is scheduler complexity and the generation stalls that can occur when a decode-prioritizing and prefill-prioritizing policy compete for the same batch slot, which is exactly the scenario the token-budget profiling covered earlier is meant to catch before it reaches production.

## What I would fix first

Thirty days: instrument shared timestamps and percentiles. Sixty: profile token budgets. Ninety: allocate and validate under load. Watch for invisible prefill time and tails blamed on the wrong hop.

> _— Kevin_

## Getting started with MLflow for latency budgets

Instrumenting a latency budget by hand across every service in a request path is tedious work, and it is exactly the kind of tracing and experiment management that MLflow is built to standardize. The platform provides open-source tracing and evaluation capabilities with compatibility for OpenTelemetry, enabling integration with existing infrastructure.

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

- Instrument your highest-traffic request path first, since that is where a latency regression does the most damage.
- Run a token-budget profiling sweep and log each run as a tracked MLflow experiment so results stay comparable across model or chunk-size changes.
- Use prompt tracking to tie a specific prompt template to the latency numbers it produced, closing the loop between prompt changes and budget regressions.

Start by pointing [MLflow](https://mlflow.org) at one service, get the per-span tracing running, and use the [MLflow AI Platform](https://mlflow.org/classical-ml) to carry that instrumentation across the rest of your model lifecycle from there.

## Key standards and benchmarks to follow

- [NVIDIA genAI-perf metrics documentation](https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/metrics.html) for TTFT, ITL, and e2e latency definitions.
- [OSDI/USENIX paper on LLM serving](https://www.usenix.org/system/files/osdi24-agrawal.pdf) for token-budget and chunked-prefill research.
- [OpenAI's latency optimization guide](https://developers.openai.com/api/docs/guides/latency-optimization) for practical production tactics.
- [NIST CSRC SP 800-207a](https://csrc.nist.gov/pubs/sp/800/207/a/final) for architecture guidance on distributed systems.

## Sources

- [OSDI paper on LLM serving (token budgets and scheduling)](https://www.usenix.org/system/files/osdi24-agrawal.pdf)
- [NVIDIA genAI‑perf LLM benchmarking metrics](https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/metrics.html)
- [OpenAI: Latency optimization guide](https://developers.openai.com/api/docs/guides/latency-optimization)
- [Prompt Cache: Modular attention reuse for low-latency inference (Google Research)](https://research.google/pubs/prompt-cache-modular-attention-reuse-for-low-latency-inference/)

## FAQ

### What is latency in an LLM system?

Latency in an LLM system is the time between sending a request and receiving a usable response, usually broken into time to first token and total generation time. It spans network transit, queueing, prefill, and decode, and each of those hops contributes its own share to the total.

### Which LLM has the lowest latency?

There is no single model that holds the lowest latency across every workload, because latency depends on model size, hardware, token budget, and infrastructure placement as much as the model itself. Smaller or fine-tuned models generally decode faster than larger general-purpose models, which is why model routing based on task difficulty is a common optimization tactic.

### How do you reduce LLM call latency?

The most effective tactics include semantic caching to avoid repeated prefill work, streaming responses to reduce perceived wait, and routing simpler requests to smaller models to cut decode time. OpenAI's latency optimization guide groups these into seven principles, including making fewer requests and avoiding an LLM call entirely when a classical method suffices.

### What is the difference between latency and throughput in an LLM?

Latency measures how long a single request takes, typically tracked as time to first token and end-to-end response time. Throughput measures how many tokens or requests the system processes per second across all concurrent traffic, and the two often trade off against each other through batch size and scheduling choices.

### How do you set latency SLOs for an LLM system?

Start from a target end-to-end p99 latency, then subtract network, retrieval, and gateway overhead to find the remaining compute budget for prefill and decode. Allocate that compute budget with explicit headroom rather than a tight fit, then validate the full allocation under realistic concurrent load before treating it as final.

## Recommended

- [Stop 429s, 15% Token Drift: Gateway LLM Rate Limits for Engineers](https://mlflow.org/articles/rate-limiting-llm)
- [AI Gateway](https://mlflow.org/genai/ai-gateway)
- [Reproducible LLM Evaluation for Engineers: 4 Components and MLflow](https://mlflow.org/articles/llm-evaluation-harness)
