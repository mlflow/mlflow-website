---
title: "LLM Routing in Production: Four Stages, Mapped to MLflow"
description: "Production LLM routing playbook: start with cache and rules, add semantic and predictive routing, and use MLflow for evaluation, tracing, and governance."
slug: llm-routing
tags:
  [
    cross provider llm routing,
    vertex ai llm routing,
    best llm routing libraries,
    best multi-llm platforms,
    multi cloud llm routing,
    llm routing,
    how to implement llm routing,
    llm routing techniques,
    llm network optimization,
    llm routing strategies,
    model routing llm,
    best practices for llm routing,
    llm routing algorithms,
  ]
date: 2026-09-07
image: https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1788766626084_Production-infrastructure-supporting-LLM-request-routing.jpeg
---

![Production infrastructure supporting LLM request routing](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1788766626084_Production-infrastructure-supporting-LLM-request-routing.jpeg)

LLM routing is the middleware layer that decides which model handles each incoming request, matching query complexity to model capability so you're not paying frontier-model prices for a task a 7B model can handle. The immediate engineering payoff is cost, latency, and reliability gained without touching your application logic. Start with rule-based routing plus a semantic cache, then graduate to semantic and predictive routing only once you have the traffic volume and labeled data to justify it.

---

> **TL;DR:**
>
> - Most teams should start with rule-based routing and semantic caching before progressing to complex predictive models to avoid unnecessary implementation risks.
> - Combining multiple routing layers, such as cache checks, semantic embeddings, and cascades, offers the most cost-effective and reliable production setup.
> - Proper infrastructure components like a gateway, a semantic cache, a vector store, fallback logic, and telemetry are essential for scalable, maintainable LLM routing.
> - Routing accuracy alone does not guarantee efficiency; latency and debuggability are equally critical factors in a production environment.
> - Continuous monitoring of key metrics such as fallback rate, cache hit rate, and cost is necessary to prevent silent drift and optimize routing policies over time.

---

## Table of Contents

- [What Is LLM Routing and Where Does It Fit in Your Stack?](#what-is-llm-routing-and-where-does-it-fit-in-your-stack)
- [Which LLM Routing Strategy Should You Use?](#which-llm-routing-strategy-should-you-use)
- [What Components Make Up a Production Routing Architecture?](#what-components-make-up-a-production-routing-architecture)
- [How Do You Orchestrate Multiple Model Calls?](#how-do-you-orchestrate-multiple-model-calls)
- [How Do You Evaluate and Benchmark a Routing Policy?](#how-do-you-evaluate-and-benchmark-a-routing-policy)
- [When Should You Adopt Routing, and How Do You Scale It?](#when-should-you-adopt-routing-and-how-do-you-scale-it)
- [How MLflow Supports This Routing Architecture](#how-mlflow-supports-this-routing-architecture)
- [What Do Engineers Consistently Get Wrong About Routing?](#what-do-engineers-consistently-get-wrong-about-routing)
- [Get Started with MLflow for Production LLM Routing](#get-started-with-mlflow-for-production-llm-routing)
- [Sources](#sources)

## What Is LLM Routing and Where Does It Fit in Your Stack?

Think of a router as the traffic cop sitting between your application and every model provider you use. It intercepts each request, inspects it, and decides where it goes: GPT class model for nuanced reasoning, a smaller open-weight model for classification, a local model for anything touching regulated data. Your application code never talks to a provider SDK directly. It talks to the router, and the router talks to the world.

That distinction matters because routing decisions happen at two different points in the request lifecycle. **Pre-generation routing** decides before any tokens get generated, based on features of the incoming prompt: length, detected intent, keyword matches, or an embedding similarity score against known query clusters. **Post-generation routing**, often called cascading, lets a cheap model attempt the response first, then evaluates the output quality and escalates to a stronger model only if the answer fails a confidence or correctness check. Production systems frequently combine both approaches into [multi-stage routing pipelines](https://www.arxiv.org/pdf/2603.04445), using cheap pre-filters to weed out easy cases and cascades to catch the ones that slip through.

Routing rarely operates in isolation. It sits next to, and often depends on, a few adjacent systems:

- **Semantic caching**, which should intercept a request before the router ever sees it. A cache hit means zero routing decisions and zero model calls.
- **Retrieval-augmented generation**, where the router might choose a model based partly on how much context the retrieval step pulled back, since bigger context windows change the cost calculus.
- **Agent orchestration**, where a single user request triggers multiple internal LLM calls (planning, tool selection, summarization) and each one may route to a different model depending on the sub-task.
- **Telemetry pipelines**, which capture the routing decision itself as a first-class event, not just the model's output, so you can audit why a given model got picked.

The practical implication: if you're building an agent that calls an LLM five times per user turn, you're not making one routing decision, you're making five, each with its own cost and latency profile. A router that only optimizes the first call and ignores the other four is optimizing the wrong [20%](https://arome.substack.com/p/the-model-router-blueprint-building) of your spend. This is also where cross-provider LLM routing earns its keep. Multi-cloud LLM routing setups that span, say, Anthropic, OpenAI, and a self-hosted Llama deployment need a router that abstracts provider-specific quirks (rate limits, token counting, context window sizes) into one consistent interface, so swapping a backend model doesn't mean rewriting application code.

## Which LLM Routing Strategy Should You Use?

Four routing strategies cover almost every production system in use today, and each one trades off implementation complexity against routing accuracy. Picking the wrong one for your traffic pattern is the single most common mistake teams make, usually because they reach for the most sophisticated option before they've earned the right to need it.

1. **Rule-based routing.** You write explicit conditional logic: requests under 200 tokens with no code blocks go to a small model; anything containing "SELECT" or "def " gets routed to a code-specialized model; requests flagged with a customer tier of "enterprise" always get the frontier model regardless of content. The appeal here isn't sophistication, it's debuggability. When a routing decision goes wrong, you can read the rule that fired and know exactly why in under a minute. There's no embedding drift to chase, no model to retrain. The downside is coverage: rules only handle patterns you anticipated, and real user queries are messier than any rule set you write on day one.

2. **Semantic/embedding routing.** Instead of matching keywords, you embed the incoming query and compare it against a library of labeled reference embeddings, routing based on which cluster the query lands closest to. A support-ticket router might have embedded clusters for "billing," "technical bug," and "feature request," each pointing to a different model tuned or prompted for that domain. This handles paraphrasing and varied phrasing that rules miss entirely, but it introduces two new problems: you need to pick a similarity threshold (too loose and unrelated queries get routed together; too tight and edge cases fall through with no match), and you need drift monitoring because the distribution of real queries shifts over time and your reference embeddings can go stale without warning.

3. **Predictive/classifier routing.** Here you train a lightweight classifier on labeled preference data, essentially pairs of (query, which model produced the better answer), so the router learns to predict model fit rather than relying on hand-picked thresholds. This is the highest-accuracy option when you have enough data, but it's also the most fragile to stand up. You need a meaningful volume of labeled examples, a stable enough query distribution that the classifier doesn't go stale within weeks, and a tolerance for the added inference latency of running a classifier on every request. Most teams underestimate how much labeled data this actually requires and end up with a classifier that's confidently wrong on the traffic that matters most.

4. **Cascading (post-generation) routing.** A cheap model answers first. A cheap classifier, or the same model scoring its own confidence, decides whether that answer is good enough. If not, the request escalates to a stronger model. This pattern is popular precisely because it sidesteps the hardest problem in pre-generation routing, predicting difficulty before you've seen the answer, and instead reacts to actual output quality. Research on cascading and routing shows this approach meaningfully reduces average cost while preserving quality on the harder subset of queries, since only the queries that actually fail the cheap model incur the expensive model's cost. The trade-off is added latency on escalated requests, since you're now paying for two generations instead of one.

**Pro Tip:** _Don't build a predictive classifier until you've logged at least a few weeks of production traffic through rule-based routing first. That log is your labeled dataset, and skipping straight to "learned routing" without it means training on synthetic guesses instead of real query patterns._

Most mature systems don't pick one strategy and stop. They layer rules for obvious cases, semantic routing for the long tail of paraphrased intent, and a cascade as a safety net for anything that slips past both. That layering, not any single technique, is what survey research on dynamic multi-LLM routing finds most production stacks actually converge on.

## What Components Make Up a Production Routing Architecture?

A router is only as good as the infrastructure around it. Five components show up in nearly every production deployment worth calling mature, and skipping any one of them tends to surface as a latency spike or a cost overrun within the first month of real traffic.

**The gateway** is the single entry point every request passes through before hitting any model provider. This is where you pin model snapshots (so a provider's silent model update doesn't change your application's behavior overnight), enforce per-tenant quotas and budgets, and apply routing policy centrally instead of scattering routing logic across a dozen microservices. Operational guidance on production LLM routing treats snapshot pinning and structured per-route telemetry as non-negotiable baseline practices, not advanced optimizations.

**The semantic cache** should sit in front of the router, not behind it. A cache hit means the request never reaches the routing logic at all, which is both faster and cheaper than routing to even the smallest model. [Architectural guidance from Redis](https://redis.io/blog/llm-router-architecture-best-practices/) is explicit on this point: check the cache first, route only on a miss.

- Colocate the cache and vector lookups on low-latency, in-process or same-datacenter stores. An external network hop for a similarity check can erase the latency savings routing was supposed to deliver.
- Set cache TTLs based on how often the underlying knowledge changes, not on a default value copied from another project.
- Track cache hit rate as a first-class metric, since a declining hit rate is often the earliest signal that query patterns are shifting.

**The vector store** backing semantic routing and retrieval needs to answer similarity queries in single-digit to low double-digit milliseconds, because that latency stacks on top of whatever the model call itself takes. If your vector store lives behind a slow API instead of a colocated index, you've built a system where the routing decision costs more time than it saves.

**Fallback and circuit breaker logic** protect you from provider outages and rate-limit walls, which happen more often than most teams plan for. A circuit breaker trips after a threshold of failed or timed-out calls to a given provider, temporarily routing all traffic away from it and retrying on a backoff schedule instead of hammering a struggling endpoint.

**Telemetry** ties the whole thing together. At minimum, emit per-route records covering:

- Which model handled the request and which strategy selected it
- End-to-end latency and time-to-first-token
- Cost per request, attributed to the specific route
- Cache hit or miss status
- Whether the request was escalated in a cascade, and why

One production benchmark makes the stakes concrete: Redis's architecture guidance notes that when cache and vector lookups run on colocated, low-latency infrastructure, routing overhead stays low enough that the savings from avoiding an expensive model call clearly outweigh the routing tax. Push those lookups behind a slow network call, and that math flips.

One caveat belongs here regardless of which architecture you pick: sensitive data subject to regulations like [HIPAA](https://www.hhs.gov/hipaa/for-professionals/privacy/index.html) may need to route to on-prem or local models rather than any public cloud provider, no matter how well that provider scores on your quality metrics. Build that constraint into your gateway's policy layer from day one rather than retrofitting it later.

## How Do You Orchestrate Multiple Model Calls?

Routing a single request to a single model is the easy case. Real production systems, especially agentic ones, chain multiple calls together, and the orchestration pattern you pick shapes your latency, cost, and debugging experience just as much as the routing strategy does.

**Sequential/cascade pipelines** run steps one after another, where each step's output feeds the next. This is the natural fit for cascading routing: cheap model attempts, confidence check, escalate if needed. The key discipline here is defining an explicit contract for each step, covering expected inputs, outputs, failure modes, and latency SLOs, and persisting intermediate state so you can replay from any stage during an incident instead of rerunning the entire pipeline from scratch. Production guidance on LLM routing treats this kind of step-level contract as essential for making multi-step routing pipelines debuggable under pressure.

**Parallel model calls** send the same or related prompts to multiple models simultaneously and reconcile the results, either by voting, by scoring each output against a rubric, or by picking whichever response returns first if latency matters more than consensus. This costs more per request by definition, since you're paying for multiple generations, but it buys you resilience against any single model producing a bad or hallucinated answer.

**Conditional routing chains** branch based on intermediate results rather than following a fixed sequence. A customer support agent might route to a retrieval step first, then branch to either a simple templated response or a full LLM generation depending on whether retrieval found a strong match. These chains need heavier observability than linear pipelines, because the branch taken on any given request isn't obvious from the outside without structured logging.

For orchestration involving agents and tool calling specifically, a few patterns show up consistently:

- Each tool call inside an agent loop is its own routing decision, and treating the whole agent turn as one opaque call hides where cost and latency actually accumulate.
- Planning steps (deciding what to do) often warrant a stronger model than execution steps (formatting a response from retrieved data), so routing within a single agent turn should vary by sub-task.
- Failure in one step of a chain shouldn't silently propagate; define what "partial success" means for your specific agent and route accordingly, whether that means retrying, falling back to a simpler action, or surfacing the failure to the user.

The common thread across all three patterns is that orchestration and routing are not separable concerns. A routing policy designed for single-shot requests will misbehave the moment you drop it into a five-step agent pipeline, because the cost and latency budget for step three depends on what happened in steps one and two.

## How Do You Evaluate and Benchmark a Routing Policy?

You can't tune what you don't measure, and routing policies decay quietly if nobody's watching the right numbers. Five metrics form the backbone of any useful eval suite: cost per request (broken out by route, not averaged across your whole system), p95 time-to-first-token, fallback rate (how often cascades escalate), cache hit rate, and a quality metric specific to your task, whether that's a rubric score, a classification accuracy, or human preference labels on a sample of outputs.

Building the eval suite itself follows a fairly consistent recipe:

1. Collect a representative sample of real production queries, not synthetic test cases, since synthetic data rarely captures the messiness of actual user phrasing.
2. Have each candidate model or route produce an answer for the same sample, then collect preference labels, either from human raters or from an LLM-as-a-Judge setup scoring against a rubric.
3. Compute your quality metric per route, alongside cost and latency, so you can see the actual trade-off curve rather than optimizing cost in isolation.
4. Run A/B tests on routing policy changes against a held-out slice of live traffic before rolling out to 100%, since offline eval sets rarely capture every edge case that live traffic will surface.
5. Set a recalibration cadence, weekly or monthly depending on traffic volume, to catch drift in query patterns before it silently degrades routing accuracy.

**Pro Tip:** _Treat your fallback rate as a leading indicator, not a lagging one. A rising fallback rate usually means your cheap-model tier is drifting out of sync with what users are actually asking, and it shows up in that metric weeks before it shows up in a quality complaint._

[Operational telemetry](https://www.metacto.com/blogs/llm-routing-production-guide) covering fallback rate, cache hit rate, and cost attribution per route tends to be the most actionable dataset teams have for spotting drift and rebalancing policy, more useful in practice than any offline benchmark run once a quarter. A tool like the Multi-LLM Audit from BabyLoveGrowth can help sanity-check how different models respond to the same prompt as a quick cross-check alongside your own eval suite.

## When Should You Adopt Routing, and How Do You Scale It?

Three signals tell you it's time to build a router instead of hardcoding a single model: your model spend has become large enough that a 20 to 30% reduction would matter to the business, you're exposed to meaningful risk from depending on a single provider, or you have latency requirements that a one-size-fits-all model can't meet across your full range of query complexity. If none of those apply yet, a router adds operational overhead without a payoff.

Once you've decided to build, the progression that avoids the most pain follows a consistent shape across the teams that get this right:

- **Stage one:** rule-based routing plus a semantic cache. This alone often captures the bulk of available savings, since Redis's architecture guidance treats this combination as the standard starting point before any learned component enters the picture.
- **Stage two:** semantic/embedding routing layered on top, to catch paraphrased or varied queries that rules miss.
- **Stage three:** predictive/classifier routing, only once you've accumulated labeled preference data and your query distribution has stabilized enough that a trained classifier won't go stale in weeks.
- **Stage four:** online learning or continuous recalibration, reserved for teams with the traffic volume and infrastructure to justify a routing policy that updates itself.

Engineering best-practice guidance consistently warns against skipping straight to stage three or four before stages one and two have had time to mature, since premature complexity tends to produce a router that's harder to debug than the cost problem it was meant to solve.

Ownership matters as much as the technical stages. Assign a clear SLO for routing latency overhead (a common target is keeping it under [10%](https://pmc.ncbi.nlm.nih.gov/articles/PMC9550181/) of total request latency), name an owner for the routing policy itself, and schedule a recurring review of telemetry rather than waiting for a cost spike to force the conversation. The most common pitfall isn't picking the wrong strategy, it's picking the right strategy and then never revisiting it as traffic evolves.

## How MLflow Supports This Routing Architecture

Everything in this guide, from the gateway layer to the eval suite, maps directly onto features built into [MLflow's GenAI platform](https://mlflow.org/genai). If you're implementing the maturity progression described above, MLflow gives you a centralized place to manage the pieces instead of stitching together separate tools for gateway config, evaluation, and observability.

The **AI Gateway** handles cross-provider governance and prompt management centrally, which matters directly for complex LLM routing and multi-cloud routing setups where you're managing snapshot pinning, per-tenant quotas, and policy enforcement across multiple backends. Rather than hardcoding provider logic into your application, the gateway can become the single point where routing policy lives.

For the evaluation work covered above, an **LLM-as-a-Judge** framework can automate the preference-labeling step in an eval suite, scoring outputs against a rubric instead of requiring a human rater for every sample. That's the same mechanism a cascade needs to decide whether a cheap model's answer is good enough to skip escalation. MLflow's [evaluation tooling](https://mlflow.org/genai/evaluations) is built around exactly this workflow.

Deep **tracing** captures agentic reasoning step by step, which is what makes the telemetry schema described earlier actually usable in practice, rather than a spec nobody implements. MLflow's [observability features](https://mlflow.org/genai/observability) log routing decisions, model calls, and intermediate agent steps as structured trace data you can query later during an incident.

A few signals suggest a platform is a strong fit for your team specifically:

- You need governance and auditability across multiple model providers, not just one.
- You're running agents with multi-step tool calling and need visibility into each internal call, not just the final output.
- You want evaluation and observability to share the same platform instead of maintaining separate tooling for each.

## What Do Engineers Consistently Get Wrong About Routing?

The biggest misconception I keep running into is that routing accuracy is the whole game. It isn't. A router that picks the right model most of the time but adds significant latency checking a remote vector store on every single request has built a system that's technically smarter and practically worse than the naive one-model setup it replaced. Latency budget is a routing constraint, not an afterthought.

The second thing conventional wisdom gets backwards: teams assume semantic routing is strictly better than rules because it's more sophisticated. In practice, rules win on debuggability by a wide margin, and debuggability is what saves you at 2 a.m. when a routing decision has gone sideways in production and you need to know why in the next five minutes, not after an afternoon of embedding forensics.

A pattern I've seen recur across postmortems: a team ships semantic routing, skips drift monitoring because the initial accuracy looked great, and three months later the reference embeddings no longer reflect what users are actually asking. Requests silently misroute to the wrong tier for weeks before anyone notices, usually because a cost report finally looks wrong. The fix was never a better model. It was a scheduled recalibration job that nobody had prioritized during the initial build.

Before shipping any router, three questions are worth answering honestly: Does your cache sit in front of routing, not behind it? Can you explain, in one sentence, why any given request got routed where it did? And do you have a telemetry dashboard you actually check weekly, not one you built and forgot?

> _— Kevin_

## Get Started with MLflow for Production LLM Routing

Reading about pinned snapshots and cascade contracts is one thing. Wiring them into a system that survives real traffic is another, and that gap is exactly where MLflow is built to help. This type of platform can give you one solution for the gateway, evaluation, and tracing layers this guide walks through, instead of stitching together separate tools for each piece of the routing stack.

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

The practical path most teams follow is pilot, integrate, monitor. Start by routing a single high-volume use case through MLflow's AI Gateway to prove out cost and latency gains on a contained slice of traffic. From there, integrate LLM-as-a-Judge evaluation into your eval suite so cascade escalation decisions are scored consistently rather than eyeballed. Once routing and evaluation are running, lean on tracing tools to monitor drift and catch the kind of silent misrouting described above before it shows up in a cost report. If prompt versioning across providers is part of your rollout, the [prompt engineering cookbook](https://mlflow.org/cookbook/prompt-engineering) walks through practical patterns for keeping prompts and routing policy in sync. Set up a pilot this week on your highest-volume route and measure the before-and-after on cost per request.

## Sources

For engineers who want to go past this guide, a few sources cover the technical ground in more depth. The arXiv survey on dynamic multi-LLM routing lays out the full taxonomy of routing paradigms in academic detail. The cascading and multi-LLM inference paper quantifies cost-quality trade-offs for escalation-based routing. Redis's router architecture writeup and AWS's multi-LLM routing strategies post both ground the theory in deployable patterns, while Metacto's production guide is the most operationally focused read of the group. Start with the survey for vocabulary, then move to the practical guides for implementation detail.

- [LLM router architecture: best practices (Redis blog)](https://redis.io/blog/llm-router-architecture-best-practices/)
- [Survey: Dynamic multi-LLM routing (arXiv)](https://www.arxiv.org/pdf/2603.04445)
- [LLM routing in production: a practical guide (Metacto)](https://www.metacto.com/blogs/llm-routing-production-guide)

## Recommended

- [Route Claude Code Through MLflow AI Gateway](https://mlflow.org/blog/gateway-claude-code)
- [Setting Up LLM Observability Pipelines in 2026](https://mlflow.org/articles/setting-up-llm-observability-pipelines-in-2026)
