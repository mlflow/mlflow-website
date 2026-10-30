---
title: "AI Gateway Patterns: Architecture, Routing and Batching"
description: "Seven AI gateway patterns for production LLM apps: request flow, routing, fallbacks, budgets, observability and batching, with how MLflow's AI Gateway fits."
slug: ai-gateway-patterns
tags:
  [
    llm batching strategies,
    ai gateway architecture,
    ai gateway patterns,
    dynamic batching llm,
  ]
date: 2026-10-27
image: https://media.babylovegrowth.ai/blog-images/organization-30814/1793284705315_Engineer-reviewing-an-AI-gateway-route-configuration.jpeg
---

![Engineer reviewing an AI gateway route configuration](https://media.babylovegrowth.ai/blog-images/organization-30814/1793284705315_Engineer-reviewing-an-AI-gateway-route-configuration.jpeg)

An AI gateway is a single layer between your application code and your model providers. It owns authentication, model resolution, routing, rate limits, cost tracking and telemetry, so none of that logic leaks into each service. The patterns below cover how a request flows through that layer, how to deploy it, how to route and protect traffic, and where batching belongs. Gateway-level request handling and backend-level batching are different jobs, and mixing them up is the most common design mistake.

## TL;DR

- Put one gateway in front of every provider and expose logical endpoint names, not raw model IDs, so you can change models without redeploying apps.
- Split a thin, stateless data plane (request handling) from a control plane (keys, budgets, audit logs).
- Use traffic splitting, fallbacks and circuit breaking for resilience, and count tokens at the gateway for cost accounting.
- Batching mostly lives in the inference server behind the gateway. The gateway's job is to shape load and protect it with queue limits and rate limits.
- MLflow ships an open-source AI Gateway with these capabilities, plus tracing and evaluation in the same platform.

## Table of Contents

- [What an AI gateway does, and how it differs from an API gateway](#what-an-ai-gateway-does-and-how-it-differs-from-an-api-gateway)
- [Pattern 1: The centralized model access layer](#pattern-1-the-centralized-model-access-layer)
- [Pattern 2: Two planes, control and data](#pattern-2-two-planes-control-and-data)
- [Pattern 3: Routing, traffic splitting and fallbacks](#pattern-3-routing-traffic-splitting-and-fallbacks)
- [Pattern 4: Governance, budgets and guardrails](#pattern-4-governance-budgets-and-guardrails)
- [Pattern 5: Observability and cost accounting at the gateway](#pattern-5-observability-and-cost-accounting-at-the-gateway)
- [Pattern 6: Load shaping for the inference layer](#pattern-6-load-shaping-for-the-inference-layer)
- [Pattern 7: Batching, and where it belongs](#pattern-7-batching-and-where-it-belongs)
- [Perspective: why we build the gateway into the lifecycle](#perspective-why-we-build-the-gateway-into-the-lifecycle)
- [Try the patterns with MLflow](#try-the-patterns-with-mlflow)
- [A rollout checklist](#a-rollout-checklist)
- [FAQ](#faq)
- [Sources](#sources)

## What an AI gateway does, and how it differs from an API gateway

A conventional API gateway routes HTTP traffic, authenticates callers and applies rate limits. An AI gateway keeps those jobs and adds ones specific to model traffic: understanding tokens, mapping a logical model name to a provider, inspecting prompts and responses, and tracking spend per team or key. Many production environments run both, with the API gateway handling general service traffic and the AI gateway handling model calls.

A typical request moves through five stages:

1. **Caller authentication.** The gateway checks the key or identity of the calling application or team.
2. **Model resolution.** The caller asks for a logical name such as `support-chat`. The gateway maps it to a concrete provider and model.
3. **Request controls.** Rate limits, budgets and guardrails (for example PII scrubbing) run before anything leaves your network.
4. **Provider dispatch.** The gateway forwards the request, translating formats if the provider differs from the client's API style.
5. **Response handling.** The gateway streams the answer back, then records tokens, latency and cost.

The key design property is that the gateway holds provider credentials, not the applications. Developers receive a gateway-scoped credential, and rotating a provider key is a one-place change.

## Pattern 1: The centralized model access layer

The foundational pattern is a single access layer that every application uses. The payoff comes from what you no longer duplicate: retry code, provider SDK upgrades, key distribution, usage logging.

In MLflow's AI Gateway, you define an _endpoint_ with a name, and applications query that name. According to the [MLflow documentation on querying endpoints](https://mlflow.org/docs/latest/genai/governance/ai-gateway/endpoints/query-endpoints), you can use the MLflow unified API, an OpenAI-compatible chat completions interface, or passthrough APIs for OpenAI, Anthropic and Google Gemini. The OpenAI-compatible route is a drop-in replacement for the OpenAI chat completions API, with the endpoint name used as the model parameter.

```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:5000/gateway/mlflow/v1',
    api_key='unused-gateway-handles-credentials',
)

response = client.chat.completions.create(
    model='support-chat',  # logical endpoint name, not a provider model ID
    messages=[{'role': 'user', 'content': 'Summarize this ticket.'}],
)
print(response.choices[0].message.content)
```

Because the application only knows `support-chat`, swapping the underlying model is a gateway configuration change. Existing client code, including code written against the OpenAI SDK, keeps working.

## Pattern 2: Two planes, control and data

The most reliable deployment shape separates two concerns:

- **Data plane:** a thin, stateless proxy that authenticates, resolves, forwards and streams. Because it is stateless, you scale it horizontally behind a load balancer.
- **Control plane:** the place where you manage virtual keys, budgets, endpoint definitions, prompt versions and audit logs.

Keeping the data plane stateless means a crashed instance costs you one in-flight request, not configuration. Shared state such as rate-limit counters belongs in a shared store that all gateway instances read, so limits hold across replicas rather than per pod.

We cover the two-tier idea in more depth in our AI gateway architecture guide for technical teams, including Kubernetes and cloud deployment.

### Where to place the gateway

Placement affects latency. A gateway that sits next to the application, in the same region or cluster, adds little overhead. A gateway in a different region adds a network round trip to every call, and for streaming chat that shows up in time to first token. Place the gateway in the same region as your heaviest callers, and use regional replicas when callers are spread out.

## Pattern 3: Routing, traffic splitting and fallbacks

Routing is where a gateway pays for itself during an incident or a model migration. Three routing behaviors cover most needs:

| Behavior             | What it does                                                | When to use it                                    |
| -------------------- | ----------------------------------------------------------- | ------------------------------------------------- |
| Logical-name routing | Maps an endpoint name to a provider and model               | Every deployment, as the default                  |
| Traffic splitting    | Sends a percentage of requests to each of several models    | Canary a new model, A/B test cost against quality |
| Fallbacks            | Retries on another model or provider when the primary fails | Provider outages, quota exhaustion                |

MLflow's documentation notes that its native invocations interface handles routing features like traffic splitting and fallbacks. A practical rollout looks like this: send 5% of an endpoint's traffic to the candidate model, compare evaluation scores and cost, then raise the share in steps. Because the split is a gateway setting, there is no application deploy in the loop.

Pair splitting with evaluation. A cheaper model is only cheaper if quality holds. Run an LLM-as-a-Judge evaluation on traced requests from both arms, then decide on data rather than on a handful of spot checks.

### Circuit breaking and health signals

Fallbacks need a trigger. Microsoft's architecture guidance on [gateways in front of multiple model backends](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/azure-openai-gateway-multi-backend) says that, unless another reliable signal exists, health checks should treat HTTP 429, 500 and 503 responses as the cue to circuit-break future requests to that backend. A sensible implementation:

- Count consecutive failures per backend.
- Open the circuit after a threshold, and route to the fallback.
- Probe the failed backend periodically with a small request before closing the circuit.
- Honor any retry-after hint from a 429 instead of retrying immediately.

Keep fallback chains short. A chain of four models hides a real outage behind rising latency and unpredictable output quality.

## Pattern 4: Governance, budgets and guardrails

A gateway is the one place where policy can be enforced for every caller. The capabilities teams usually ask for:

- **Virtual keys per team or application**, so you can attribute spend and revoke access without touching provider accounts.
- **Token-based rate limits and budgets**, which map to cost better than request counts do, since one long-context call can cost as much as dozens of short ones.
- **Input and output policies**, such as PII scrubbing before a prompt leaves your boundary.
- **Audit logs** recording who called which endpoint, with what outcome.

Set budgets at two levels: a soft alert at a fraction of the limit that notifies the owning team, and a hard cap that rejects requests. Start with alerts only for the first weeks of a new application, then tighten once you know its normal usage.

Prompt management belongs in the same governance story. When prompts are versioned and referenced by name, you can review a change, roll it back and tie a regression to a specific version rather than to a vague deploy.

## Pattern 5: Observability and cost accounting at the gateway

The gateway sees every request, which makes it the natural place to record the basics: endpoint, caller, provider, model, input and output tokens, latency, status code and cost. Treat these records as your source of truth for spend, since provider invoices arrive late and rarely break down by internal team.

Gateway telemetry only tells half the story, though. It shows what was routed and how many tokens moved. It does not show why an agent chose a tool, or which retrieval step returned irrelevant context. For that you need tracing inside the application.

MLflow Tracing captures the reasoning steps of an agent as nested spans, so you can follow a slow or wrong answer from the user's question through each LLM call and tool call. Because MLflow builds on OpenTelemetry, traces fit into the telemetry tooling you already run. The practical setup is two views joined by a shared request or trace ID:

- **Gateway view:** routing decisions, token counts, spend, rate-limit rejections.
- **Application view:** agent steps, tool calls, retrieval results, evaluation scores.

When a cost spike appears in the first view, the second tells you which prompt or agent loop caused it.

## Pattern 6: Load shaping for the inference layer

If you host models yourself, the gateway should protect GPU capacity. Inference servers have limited concurrency, and a flood of requests turns into long queues and timeouts for everyone. Gateway-level protections:

- Per-caller and per-endpoint rate limits, so one noisy batch job cannot starve interactive users.
- Request size limits on prompts and maximum output tokens.
- Priority classes, so interactive traffic is served ahead of background jobs.
- Rejecting excess load early with a clear error, rather than letting it reach the GPU tier.

Shedding load at the gateway is cheaper than discovering overload at the model server.

## Pattern 7: Batching, and where it belongs

Readers often ask about dynamic batching for LLMs and LLM batching strategies in the same breath as gateways. The two are related but separate. Batching is performed by the inference server that runs the model. The gateway influences it indirectly through traffic shape, priorities and limits.

### Dynamic batching

Dynamic batching combines individual inference requests into a larger batch, which often runs more efficiently than handling each request alone, according to the [NVIDIA Triton Inference Server optimization guide](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2590/user-guide/docs/user_guide/optimization.html). The server waits briefly for requests to accumulate, then executes them together. The trade-off is built in: a longer wait window improves hardware utilization but adds queueing delay for the first request in each batch.

![Dynamic batching wait window and latency tradeoff](https://media.babylovegrowth.ai/blog-images/organization-30814/1793284736194_Dynamic-batching-wait-window-and-latency-tradeoff.jpeg)

The same guide gives a rule of thumb for maximum throughput: set request concurrency to 2 × maximum batch size × number of model instances. In practice, this tells you how many simultaneous requests your load generator or gateway should allow to keep the batcher full. It is a starting point for throughput testing, not a target for latency-sensitive services.

### Static, dynamic and continuous batching

| Strategy            | How it works                                                      | Best fit                                                |
| ------------------- | ----------------------------------------------------------------- | ------------------------------------------------------- |
| Static batching     | Fixed batch assembled up front                                    | Offline jobs with uniform inputs                        |
| Dynamic batching    | Requests are grouped within a short wait window                   | Stateless models, mixed traffic                         |
| Continuous batching | Requests join or leave the running batch between generation steps | Autoregressive LLM serving with variable output lengths |

LLM output lengths vary widely, so a batch held until its longest sequence finishes wastes capacity. Continuous batching addresses this by letting finished sequences leave and new ones join at iteration boundaries, which keeps hardware busier. Most modern LLM serving engines implement some form of it, so for self-hosted models you usually enable and tune it rather than build it.

### Choosing a batching approach

Start from the workload, then set limits at both the server and the gateway:

1. **Interactive chat or agents:** prioritize latency. Use short wait windows and strict queue limits, and set a time-to-first-token target at the 95th percentile.
2. **Offline jobs such as bulk summarization or evaluation runs:** batch aggressively and accept longer waits. Route these through a separate endpoint with its own rate limit.
3. **Mixed traffic:** separate endpoints, and ideally separate server pools, so a bulk job cannot inflate interactive latency.

Measure before tuning. Record p50 and p95 latency, time to first token and throughput per endpoint, change one setting at a time, and re-run the same load test. If you call hosted providers rather than your own models, provider-side batching is out of your hands. Your levers are concurrency limits and, for non-urgent work, provider batch APIs where offered.

## Perspective: why we build the gateway into the lifecycle

We see teams adopt a gateway to solve a routing problem and then discover they have a visibility problem. Once model calls are centralized, the next question is always whether a change made things better. That is why we think a gateway should not be an isolated proxy.

![Perspective: why we build the gateway into the lifecycle](https://media.babylovegrowth.ai/blog-images/organization-30814/1793284755716_Perspective-why-we-build-the-gateway-into-the-lifecycle.jpeg)

In MLflow, the AI Gateway sits alongside tracing, LLM-as-a-Judge evaluation and prompt management. A traffic split sends requests to a candidate model, traces show what each arm did, evaluation scores tell you whether quality held, and prompt versions let you roll back a regression. Every one of those features is open source under Linux Foundation governance, with no enterprise paywall on core capabilities. We believe the governance layer for agents should be something teams can inspect, extend and run anywhere, and OpenTelemetry compatibility means it fits the stack you already have.

Our advice: pick the gateway pattern that matches your biggest risk today (cost, outages or compliance), but choose tooling that will also carry you through evaluation and agent debugging.

> _— Kevin_

## Try the patterns with MLflow

If you want to put these patterns into practice, [MLflow](https://mlflow.org) gives you the AI Gateway, tracing, evaluation and prompt management in one open-source platform. Start with a single endpoint, point one application at it, and add splitting, budgets and tracing as you go.

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

## A rollout checklist

1. Inventory every place your applications call a model provider directly.
2. Create one gateway endpoint per logical use case and migrate callers one at a time.
3. Move provider credentials into the gateway and issue scoped keys to teams.
4. Turn on token and latency recording, and join it with application traces by request ID.
5. Add fallbacks with a short chain, and test them by blocking the primary in staging.
6. Set budgets, alert first and cap later.
7. Canary model changes with traffic splitting, judged by evaluation scores.
8. For self-hosted models, set queue limits and separate interactive from batch endpoints.

## FAQ

### What is the difference between an AI gateway and an API gateway?

An API gateway handles general HTTP traffic with authentication, routing and rate limits. An AI gateway adds model-specific features such as token tracking, cost management, prompt and response inspection and provider routing. Many environments run both, each for its own traffic.

### Does the gateway do the batching of LLM requests?

Usually not. Batching is done by the inference server running the model. The gateway shapes the load that reaches it through rate limits, priorities, request size limits and separate endpoints for interactive and batch traffic.

### How many fallback models should an endpoint have?

One or two is usually enough. Longer chains hide real outages behind higher latency and inconsistent answer quality. Trigger fallbacks on clear failure signals such as HTTP 429, 500 and 503 responses.

### Can I use the OpenAI SDK with MLflow's AI Gateway?

Yes. The gateway exposes an OpenAI-compatible chat completions route, and you pass the gateway endpoint name as the model parameter. Point the SDK's base URL at your gateway and your existing client code keeps working.

## Sources

- [MLflow: Query Endpoints](https://mlflow.org/docs/latest/genai/governance/ai-gateway/endpoints/query-endpoints)
- [Microsoft Azure Architecture Center: gateway in front of multiple model backends](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/azure-openai-gateway-multi-backend)
- [NVIDIA Triton Inference Server: Optimization](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2590/user-guide/docs/user_guide/optimization.html)
