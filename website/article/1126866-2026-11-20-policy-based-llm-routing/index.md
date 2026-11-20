---
title: "Policy-Based LLM Routing: How to Design, Test and Run It"
description: "Policy-based LLM routing explained for production teams: how to write policies, pick a strategy, handle fallbacks, and trace and evaluate every routing decision."
slug: policy-based-llm-routing
tags: [policy based model selection, policy-based llm routing]
date: 2026-11-20
image: https://media.babylovegrowth.ai/blog-images/organization-30814/1795050069313_Engineer-reviewing-a-model-routing-policy.jpeg
---

![Engineer reviewing a model-routing policy](https://media.babylovegrowth.ai/blog-images/organization-30814/1795050069313_Engineer-reviewing-a-model-routing-policy.jpeg)

Policy-based LLM routing means your application states what task it needs done, and a declarative policy decides which model serves each request. Policies key off request attributes such as feature, user tier, data sensitivity, latency target and budget. The payoff is that changing a model choice becomes a config change instead of a deploy, and you can always answer which rule sent a given request where. To run it in production you need three things: a policy layer in a central gateway, tracing that records every decision, and evaluation that tells you whether a policy still earns its place. MLflow gives you the gateway, tracing and evaluation pieces in one open-source platform.

## TL;DR

- A routing policy maps request attributes to a model through declarative rules, so app code asks for a task and the policy picks the model.
- Apply hard constraints first (data residency, output format, latency ceilings), then score the remaining models on quality, cost and reliability.
- Combine a cheap rule with a cascade fallback. Pure semantic or classifier routing needs a tuned threshold and labeled data.
- Log the policy version, the decision and its reason on every request, and re-run evaluations whenever a provider changes a model version.
- Put routing in a central gateway with tracing and LLM-as-a-judge evaluation behind it, so policies are governed and measurable.

## Table of Contents

- [What policy-based routing is, and how it differs from static model selection](#what-policy-based-routing-is-and-how-it-differs-from-static-model-selection)
- [The four strategies a policy can use](#the-four-strategies-a-policy-can-use)
- [How to define a routing policy, step by step](#how-to-define-a-routing-policy-step-by-step)
- [Where routing should run, and what happens when a call fails](#where-routing-should-run-and-what-happens-when-a-call-fails)
- [Governance and observability controls](#governance-and-observability-controls)
- [Evaluating a routing policy](#evaluating-a-routing-policy)
- [Doing this with MLflow](#doing-this-with-mlflow)
- [Our perspective](#our-perspective)
- [FAQ](#faq)
- [Sources](#sources)

## What policy-based routing is, and how it differs from static model selection

In a static setup, a model name is hard-coded where the call is made. Swapping it means a code change, a review and a release. Every team that copies the pattern ends up with a different model string in a different repo.

Policy-driven routing moves that decision out of application code. One description of the approach is that, given a request with known attributes (feature, user, risk, budget), a [declarative policy chooses the model](https://tech.hoomanely.com/policy-driven-model-routing-selecting-the-right-llm-per-request/) rather than ad-hoc code. A policy might say that free-tier bulk summarization uses a small, cheap model up to a token cap, while enterprise incident reports always use a premium long-context model with safety filters on. Three consequences follow:

1. Changing routing rules is a config change, not a deploy.
2. Application code states the task, and policies decide the model.
3. You can answer "what policy fired?" in one place.

The academic framing is useful for the engineering trade-offs. A [systematic survey of LLM routing](https://aclanthology.org/2026.findings-emnlp.2072.pdf) treats model choice as a per-query decision that balances expected quality against inference cost. It points out that the LLM setting differs from classical algorithm selection in four ways: costs vary by orders of magnitude, evaluating every candidate per query is prohibitive, model strengths are emergent, and the candidate pool churns, so rules have to cope with models they have never seen. That last point is the strongest argument for policies over hard-coded names: the model list will change, and your policy layer should absorb it.

## The four strategies a policy can use

A policy is the contract. The strategy is how it reaches a decision. Four approaches cover most real systems, as [one practitioner overview](https://aiarch.dev/llm-router) lays out:

| Strategy               | How it decides                                                  | Strength                      | Watch out for                          |
| ---------------------- | --------------------------------------------------------------- | ----------------------------- | -------------------------------------- |
| Static rules           | Request type, prompt length, user tier                          | Predictable, easy to audit    | Brittle as traffic changes             |
| Classifier or semantic | Small classifier or embedding match on prompt meaning           | Fast, handles varied phrasing | Needs labeled data or an utterance set |
| Cascade or fallback    | Try a cheap model, escalate on low confidence or provider error | Saves cost on easy traffic    | Escalated requests pay for two calls   |
| Cost-aware threshold   | Route on a cost-quality threshold tuned on preference data      | Explicit trade-off knob       | Needs preference data and re-tuning    |

The survey draws a clean line between two terms you will see used loosely. [Routing is a single decision](https://arxiv.org/abs/2603.04445) that maps a query to one model, while cascading is sequential escalation from smaller to larger models when quality is insufficient. Difficulty can be judged by heuristics, learned classifiers or an LLM-as-a-judge, and production systems often combine routing and cascading.

### Rules first, then learned routing

Start with rules. They are cheap to reason about and trivial to explain to a compliance reviewer. Add learned routing only where rules cannot separate the traffic.

If you do go semantic, remember that the similarity threshold carries the weight. [Semantic routing](https://redis.io/blog/llm-router-architecture-best-practices/) encodes queries and candidate routes as embeddings and picks the closest match by cosine similarity. Without a threshold, a router can confidently misroute ambiguous traffic. Give low-similarity requests a default route instead of forcing a match.

A newer variation lets you write the policies themselves in natural language. VentureBeat [reported on Arch-Router](https://venturebeat.com/technology/new-1-5b-router-model-achieves-93-accuracy-without-costly-retraining), a 1.5B-parameter model that reached 93% routing accuracy without retraining. Users define policies through a two-level domain and action taxonomy, and a mapping links each policy to a model, so models can be swapped by editing policies alone. It shows how far the policy idea can go: the policy is the interface and the model is an implementation detail.

![Rules first, then learned routing](https://media.babylovegrowth.ai/blog-images/organization-30814/1795050179737_Rules-first-then-learned-routing.jpeg)

## How to define a routing policy, step by step

The most common failure is jumping straight to comparing models. Work in this order instead.

### 1. Write the workflow contract

Before comparing anything, define representative inputs, required outputs and capabilities, latency and volume targets, quality expectations and the failures you cannot accept. [Model-selection guidance](https://zephior.com/insights/how-to-select-an-ai-model-for-an-application) recommends judging the assembled application behavior, including retrieval, tools, validation and human review, rather than an abstract model score. For an agent, that means the contract covers the whole loop, not a single completion.

### 2. Classify requests by stable attributes

Use attributes that do not shift week to week: task type, complexity, output format, data sensitivity and business tier. [Policy-engineering guidance](https://blog.inference-labs.com/posts/ai-model-selection-policy-engineering) uses exactly these as the classification step. Resist classifying on volatile signals, such as this week's observed latency, in the policy itself. Those belong in health checks and circuit breakers.

### 3. Apply hard eligibility constraints

Separate non-negotiables from preferences. Hard constraints remove models that violate regulated-data eligibility, JSON output requirements or a tenant's latency target. The same guidance suggests excluding candidates that fail data-handling, residency, licensing, deployment, context-window or minimum-service requirements, and treating everything else as tradeable.

A concrete example:

```yaml
policy: support-ticket-triage
version: 7
match:
  feature: ticket_triage
  data_class: [internal, pii]
constraints:
  require_json_mode: true
  allowed_regions: [us]
  max_p95_latency_ms: 2500
prefer:
  order: [small-fast, mid-tier]
  escalate_to: premium
  escalate_when: judge_score < 0.7
```

This is illustrative pseudo-config, not an MLflow schema. The shape is what matters: match, constrain, prefer, escalate, plus an explicit version.

### 4. Score what remains

Among eligible models, score capability fit, latency, expected cost and reliability. The "Route Anything" study gives a clean way to think about the result: a router's value is the best expected outcome net of priced cost, and a router sits on a [ladder from the best fixed model to a per-input oracle](https://arxiv.org/pdf/2611.05275.pdf). If your router does not beat the best single model net of cost, you have added complexity for nothing.

### 5. Version the policy and its rationale

A policy without a recorded reason becomes folklore within a quarter. Store the version, the owner and the reason for each rule next to the rule itself.

## Where routing should run, and what happens when a call fails

Put routing in one shared layer, not in each service. A central gateway gives you a single place to enforce policies, hold provider credentials, apply budgets and observe traffic. Scattered client-side routing recreates the static-selection problem with extra steps.

The scale argument is documented. PayPal's gateway paper describes a [policy-driven, multi-vendor gateway](https://arxiv.org/abs/2610.35509v1) where governance is declarative policy evaluated by Open Policy Agent, with role-scoped authoring, approval workflows, shadow-mode rollout and versioned policy bundles. It serves more than 12,000 developers and 100+ models, at roughly 200 billion tokens per day, adding a median of 4 ms and a p99 of 11 ms of overhead. Most teams are nowhere near that volume, but the design carries over: policies are reviewed, versioned and rolled out gradually, like code.

### Fallbacks need rules too

A fallback is a routing decision, so it needs a policy. OpenAI's [failover guidance](https://developers.openai.com/blog/designing-failover-across-openai-model-providers) says to limit retries to safe, retryable failures and stay inside the request or workflow deadline. It advises against automatically switching routes for malformed requests, authorization failures, refusals, or exhausted quota and spend limits, and warns against blindly replaying partial outputs or uncertain tool actions. The secondary route should be approved, healthy, tested and funded.

For agents this matters most. If a tool call may have already executed, a blind retry on another model can duplicate a side effect. Encode that in the policy: which error classes may fail over, which must surface, and which steps are idempotent.

![Fallbacks need rules too](https://media.babylovegrowth.ai/blog-images/organization-30814/1795050148127_Fallbacks-need-rules-too.jpeg)

## Governance and observability controls

Routing now decides which model, provider and policy path handles a request, so it is a governance surface as well as a cost lever. A [production control-plane checklist](https://icmd.app/article/stop-shipping-ai-features-start-shipping-model-control-planes-1781609669192) names the components worth having:

- Per-request routing by task, cost, latency, region and policy, with fallbacks and circuit breakers.
- Policy enforcement for PII and PHI, plus tool permissions.
- Evaluation gates: regression suites, adversarial tests and golden sets.
- Traces linking the user action to the model call to the tool execution.
- Cost governance through per-tenant budgets, throttles, caching and usage-tied alerts.

For each request, log the policy version, the chosen model, the reason, latency and cost. Without those fields you cannot debug a bad answer or prove a rule was followed.

## Evaluating a routing policy

A policy is a hypothesis about your traffic. Test it like one.

1. **Build a representative set.** Include difficult cases, malformed inputs, high-consequence requests and the languages your users actually write in, as the [selection guidance](https://zephior.com/insights/how-to-select-an-ai-model-for-an-application) recommends.
2. **Replay offline.** Run historical traffic through the new policy and compare quality, cost and latency against the current one.
3. **Canary online.** Send a guarded slice of live traffic to the new policy, with rollback conditions defined before you start.
4. **Re-run on provider changes.** Silent model drift can make old evaluations stale, so [rerun evaluations when provider versions change](https://blog.inference-labs.com/posts/ai-model-selection-policy-engineering).

Quality scoring at this volume usually means automated judges. A worked example of the arithmetic: if a cheap model handles 70% of traffic at a judge score you accept, and the other 30% cascades to a premium model, your blended cost is the cheap rate for every request plus the premium rate for 30% of them. The escalated third pays twice. Compare that blend, not the cheap model's price, against sending everything to the premium model, and check that the quality scores hold on the escalated slice.

## Doing this with MLflow

MLflow is an open-source platform for managing the lifecycle of GenAI and LLM applications. Its pieces line up with the layers above.

- **Gateway.** The [MLflow AI Gateway](https://mlflow.org/ai-gateway) is a centralized point for secure prompt management and cross-provider governance, which is where a policy layer belongs.
- **Tracing.** [LLM and agent observability](https://mlflow.org/genai/observability) records deep traces of agentic reasoning, so you can see which model handled each step and what it cost in time and tokens. Tracing is built on OpenTelemetry-compatible standards, which keeps it compatible with the frameworks you already use.
- **Evaluation.** [Automated evaluation](https://mlflow.org/genai/evaluations), including [LLM-as-a-judge](https://mlflow.org/llm-as-a-judge) scoring, gives you the replay and canary measurements a policy change needs.

If you want the staged view of how these fit together, our guide to [LLM routing in production](https://mlflow.org/articles/llm-routing) maps four stages to MLflow features, and the [LLM routing strategies](https://mlflow.org/articles/tags/llm-routing-strategies) tag collects related reading.

> Ready to try it? MLflow is free and open source, with all features available and no enterprise feature paywalls. Linux Foundation governance keeps it vendor-neutral. Start with the [MLflow platform](https://mlflow.org), add tracing to one agent, and put your first policy decision on the trace.

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

> We built this section as a short starting point: trace first, then route, then evaluate.

## Our perspective

> Most routing projects fail at measurement, not at routing. Teams write clever rules, ship them, and then cannot say whether the rules helped, because nothing recorded which rule fired or how the answer scored.
>
> We think of routing as the middle of a loop: trace every request, route by policy, evaluate the outcome, and feed the result back into the policy. That is why we build tracing, evaluation, prompt management and the gateway in one platform rather than as separate tools. A policy change you cannot trace and score is a guess.
>
> We also think the loop should be open. Policies govern which provider sees your data, so the layer that enforces them should be inspectable, run on open standards such as OpenTelemetry, and sit under neutral Linux Foundation governance. Start with simple rules, record every decision, and let the evaluations tell you when a smarter router is worth building.
>
> _Kevin_

## FAQ

### What is the difference between policy-based routing and model selection?

Model selection is choosing a model once, usually at design time. Policy-based routing chooses per request at runtime, using declarative rules on attributes like task, tier, sensitivity and budget. The rules live outside application code, so you can change them without a deploy.

### Should I start with a classifier or with rules?

Start with rules. They are predictable and easy to audit, and they cover most of the routing value for typical traffic. Add a semantic or classifier layer when rules cannot separate your request types, and tune its similarity threshold on real traffic with a safe default route.

### How do I know a routing policy is working?

Compare it against the best single model, net of cost, using an evaluation set drawn from real traffic. Run offline replay first, then a small online canary with rollback conditions. Track quality scores, latency and cost per policy version.

### When should a request fail over to another model?

Only for safe, retryable failures such as provider outages or timeouts, and only within the workflow deadline. Do not fail over automatically on malformed requests, authorization errors, refusals or exhausted budgets. Be careful with agent steps where a tool action may already have run.

## Sources

- [Policy-Driven Model Routing: The Right LLM Per Request](https://tech.hoomanely.com/policy-driven-model-routing-selecting-the-right-llm-per-request/)
- [A Systematic Survey of Large Language Model Routing](https://aclanthology.org/2026.findings-emnlp.2072.pdf)
- [LLM routing: sending each request to the right model](https://aiarch.dev/llm-router)
- [Dynamic Model Routing and Cascading for Efficient LLM Inference](https://arxiv.org/abs/2603.04445)
- [New 1.5B router model achieves 93% accuracy without costly retraining](https://venturebeat.com/technology/new-1-5b-router-model-achieves-93-accuracy-without-costly-retraining)
- [LLM router architecture: best practices for 2026](https://redis.io/blog/llm-router-architecture-best-practices/)
- [AI Model Selection Policy Engineering](https://blog.inference-labs.com/posts/ai-model-selection-policy-engineering)
- [How to select an AI model for a real application](https://zephior.com/insights/how-to-select-an-ai-model-for-an-application)
- [Route Anything: A Study of Model and Policy Routing](https://arxiv.org/pdf/2611.05275.pdf)
- [Governing Enterprise AI Traffic at Scale](https://arxiv.org/abs/2610.35509v1)
- [Designing failover across OpenAI model providers](https://developers.openai.com/blog/designing-failover-across-openai-model-providers)
- [Stop Shipping AI Features. Start Shipping Model-Control Planes](https://icmd.app/article/stop-shipping-ai-features-start-shipping-model-control-planes-1781609669192)

## Recommended

- [LLM Routing in Production: Four Stages, Mapped to MLflow](https://mlflow.org/articles/llm-routing)
