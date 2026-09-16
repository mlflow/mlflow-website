---
title: "Stop 429s, 15% Token Drift: Gateway LLM Rate Limits for Engineers"
description: "Engineering first playbook for token aware gateway rate limiting. Enforce TPM and RPM, avoid 429s, detect 15% tokenizer drift, and partition quotas."
slug: rate-limiting-llm
tags:
  [
    per tenant rate limits,
    llm rate limit handling,
    rate limiting llm apis,
    scaling LLM applications,
    limiting large language models,
    rate limits in LLMs,
    LLM performance optimization,
    rate control for LLMs,
    how to limit LLM usage,
    best practices for LLM rate limiting,
    rate limiting llm,
    llm rate limits,
  ]
date: 2026-09-15
image: https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1789447378038_Engineer-monitoring-anonymous-API-gateway-equipment.jpeg
---

![Engineer monitoring anonymous API gateway equipment](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1789447378038_Engineer-monitoring-anonymous-API-gateway-equipment.jpeg)

Rate limiting an LLM application means controlling requests per minute (RPM), tokens per minute (TPM), and tokens per day (RPD) at once, not just counting API calls. Token-aware metering enforced at a gateway, not scattered across application code, is the design that survives production traffic. Start by capping token budgets per tenant, partitioning quotas by workload, and building graceful fallbacks before you touch retry logic. The rest of this guide covers the algorithms, monitoring, and configuration patterns that make that work in practice.

---

> **TL;DR:**
>
> - Token-based metering at the gateway is essential for accurate, scalable rate limiting, as request counts alone can misrepresent actual usage and costs.
> - TPM caps are usually the tightest constraint because token consumption varies significantly between small and large requests, often before RPM limits are reached.
> - Partitioning quotas by tenants or workload types prevents noisy neighbors from causing service degradation and enables fair resource distribution.
> - Enforcing limits at the gateway reduces effort, simplifies fallback routing, and minimizes synchronization issues compared to implementing logic within individual services.
> - Regularly comparing local token estimates with provider-reported usage and setting alerts for usage drift beyond 15 percent can prevent silent quota breaches before user impact occurs.

---

## Table of Contents

- [How LLM Rate Limits Actually Work](#how-llm-rate-limits-actually-work)
- [Why Counting Requests Alone Breaks Down for LLMs](#why-counting-requests-alone-breaks-down-for-llms)
- [Should You Enforce Limits at the Gateway or in Your App?](#should-you-enforce-limits-at-the-gateway-or-in-your-app)
- [Algorithms and Strategies for Rate Control](#algorithms-and-strategies-for-rate-control)
- [How Should You Handle 429s and Provider Overload?](#how-should-you-handle-429s-and-provider-overload)
- [What Metrics Reveal Quota Pressure Before Users Complain?](#what-metrics-reveal-quota-pressure-before-users-complain)
- [Implementation Patterns and MLflow AI Gateway Setup](#implementation-patterns-and-mlflow-ai-gateway-setup)
- [Rate Limiting Checklist and Default Settings to Start With](#rate-limiting-checklist-and-default-settings-to-start-with)
- [Rate Limiting Is a Distributed-Systems Problem, Not a Config Flag](#rate-limiting-is-a-distributed-systems-problem-not-a-config-flag)
- [Manage Rate Limits Without Building the Plumbing Yourself](#manage-rate-limits-without-building-the-plumbing-yourself)
- [Docs Worth Bookmarking Before You Build](#docs-worth-bookmarking-before-you-build)
- [Sources](#sources)
- [FAQ](#faq)

## How LLM Rate Limits Actually Work

Every major provider enforces limits across three or four dimensions at once, and they don't move in lockstep. RPM caps the number of API calls you can make per minute. TPM caps the number of tokens processed per minute, split between prompt (input) tokens and completion (output) tokens. RPD caps total requests per day, and some providers layer in spend-based windows that throttle you once you cross a rolling dollar threshold rather than a token count.

As of 2026, [OpenAI's rate limit documentation](https://developers.openai.com/api/docs/guides/rate-limits) confirms this multi-dimensional structure is the norm: limits scale with usage tier, apply per project or organization, and TPM is usually the first ceiling you hit, well before RPM becomes a factor. A single request with a 40,000-token document can burn through your minute-level token budget while your request count barely moves.

Google's Gemini API behaves similarly but with its own quirks. Some quotas [reset on Pacific time boundaries](https://ai.google.dev/gemini-api/docs/rate-limits.md.txt) rather than a rolling window, and certain tiers use spend-based short-term windows specifically to prevent runaway billing. When you get throttled, the response usually tells you what you need to know:

- A `429` status code signaling you've exceeded a quota.
- A `Retry-After` header telling you how long to wait.
- Usage details in the response body showing prompt and completion token counts.

**Token throughput, not call volume, is the metric that determines whether your application stays online during a traffic spike.**

## Why Counting Requests Alone Breaks Down for LLMs

A request-count limiter treats a three-word classification query the same as a 40,000-token legal document summary. That's the core failure. Traditional REST APIs have fairly uniform payload sizes, so counting requests works fine. LLM workloads don't behave that way, and the variance between your smallest and largest calls can span three orders of magnitude.

The consequences show up in three places. First, unexpected bills: a handful of oversized requests can consume an entire day's token budget while your dashboard still shows plenty of "requests remaining." Second, truncated outputs: when you're close to a token ceiling, providers may cut completions short mid generation, and you won't notice unless you're checking `finish_reason`. Third, noisy-neighbor exhaustion: one tenant running a batch summarization job can starve every other tenant sharing the same API key, even though each of them made far fewer calls.

Two mitigation patterns fix most of this. Pre-flight token estimation counts (or approximates) the input tokens before you send the request, so you can reject or queue it before it ever reaches the provider. Post-call reconciliation compares what you estimated against what the provider actually billed, since tokenizer differences between your local counter and the provider's real tokenizer routinely cause [drift you need to catch](https://www.systemshardening.com/articles/kubernetes/llm-rate-limiting/).

![Token estimation and billing reconciliation flow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1789447383516_Token-estimation-and-billing-reconciliation-flow.jpeg)

**Pro Tip:** _Don't trust your local tokenizer as gospel. Run a weekly sample comparison between your pre-flight estimate and the provider's actual `usage` block, and alert if the gap creeps past 15 percent._

## Should You Enforce Limits at the Gateway or in Your App?

This is the architectural decision that shapes everything downstream, and most teams get it wrong by defaulting to whatever's fastest to ship.

1. **Gateway-level enforcement** centralizes accounting across every service that calls an LLM, so you're not reimplementing token counters in five microservices. It also makes fallback routing trivial. Virtual keys and provider routing mean one exhausted provider doesn't take down the whole system, a pattern gateway vendors increasingly [build in by default](https://dev.to/pranay_batta/rate-limiting-in-llm-applications-why-you-need-it-and-how-to-build-it-5gf4).
2. **Application-level enforcement** keeps custom business logic close to the code that needs it. If one endpoint needs a completely different priority scheme than another, embedding that logic in the app avoids pushing edge cases into a shared gateway config.
3. **The operational trade-offs matter more than the theory.** A gateway adds a network hop and 10 to 50 milliseconds of latency for token estimation, plus the risk of tokenizer mismatch between the gateway's counter and the provider's real count. App-level logic skips that hop but duplicates effort across every service and makes cross-team quota visibility much harder to maintain.

Most production systems that scale past a handful of services land on gateway-centered enforcement, treating it as the single source of truth, with application code handling only workload-specific priority hints passed upstream.

## Algorithms and Strategies for Rate Control

Four algorithm patterns cover almost every real-world scenario, and picking the right combination matters more than picking a single "best" one.

**Token bucket** allots a bucket of tokens that refills at a fixed rate, letting you absorb short bursts without dropping requests instantly. It's the closest fit for TPM enforcement because it mirrors how token consumption actually behaves: bursty, then idle, then bursty again.

**Fixed window** counters reset entirely at clock boundaries (say, every 60 seconds), which is simple to implement but creates a thundering-herd problem right at the reset edge, when every queued request fires at once.

**Sliding window** counters smooth that edge by tracking a rolling timeframe instead of a hard reset, giving you more even throughput at the cost of slightly more computation.

**Quota partitioning** splits your total capacity across tenants, workloads, or API keys so no single consumer can exhaust shared capacity. This is the fix for the noisy-neighbor problem covered earlier, and it's arguably the highest-leverage change most teams haven't made yet.

The strongest production setups layer these together rather than picking one:

- Enforce RPM and TPM simultaneously, since either one can bind first depending on the request shape.
- Set a hard daily or monthly budget cap underneath both, so a sustained high rate within limits still can't blow past your spend ceiling.
- Partition quotas by tenant or workload class (interactive versus batch), and apply weighted fair queuing within each partition so low-priority jobs never permanently starve while high-priority ones stay responsive.

Treating rate limits as a distributed capacity problem rather than a per-call gate is what prevents [priority inversion and silent starvation](https://tianpan.co/blog/2026-04-17-llm-rate-limits-distributed-systems-starvation) once you have more than one consumer sharing a key.

**Pro Tip:** _Give batch jobs and interactive traffic separate API keys from day one. It's the single easiest change that prevents a nightly summarization job from starving your live chat feature._

## How Should You Handle 429s and Provider Overload?

Not every error means the same thing, and treating them identically is how naive retry logic turns a minor blip into a cascading outage.

1. **Tell quota errors apart from overload errors.** A `429` almost always means you've hit your own rate limit or quota. A `529` or generic `5xx` typically signals the provider itself is overloaded, which calls for different handling; hammering it with retries only makes things worse for everyone hitting that provider.
2. **Honor `Retry-After` when the provider sends it**, and fall back to exponential backoff with full jitter when it doesn't. Full jitter (a randomized delay rather than a fixed exponential curve) spreads retries out enough to avoid synchronized retry storms across your fleet, a pattern well documented in production retry guidance.
3. **Cap retry budgets by request deadline, not attempt count.** A user-facing chat request waiting eight seconds for a response shouldn't get the same five-retry budget as a background batch job with a ten-minute deadline.
4. **Open a circuit breaker after a threshold of consecutive failures**, then probe recovery with a single low-cost request before reopening the gate fully. This keeps a struggling provider from getting hammered by your entire retry queue while it recovers.

**Pro Tip:** _Log the error class (429 versus 5xx) as a separate metric dimension from the start. Lumping them together hides exactly which failure mode is driving your incident._

## What Metrics Reveal Quota Pressure Before Users Complain?

Rate limit problems are usually silent until they aren't. By the time users are filing tickets about truncated answers, you've been degrading for a while.

Track these signals continuously, not just during an incident review:

- TPM and RPM per tenant, broken out individually rather than aggregated across your whole fleet.
- The percentage of responses where `finish_reason == "length"`, which flags completions getting cut off before they finished.
- Provider-reported usage drift against your own gateway counts.
- 429 rate over time, alongside the `Retry-After` values you're actually receiving.

**A drift between your gateway's token accounting and the provider's real usage that exceeds roughly 15 percent is the threshold worth alerting on**, according to production reconciliation guidance built specifically around this problem.

Set alerts on a sudden spike in truncated responses, on any single tenant consuming an outsized share of total tokens within a one-hour window, and on persistent 429s hitting your interactive (user-facing) paths specifically, since those are the ones users actually feel. Run the reconciliation job periodically, comparing gateway counts against provider usage blocks, and treat sustained drift as a bug, not noise.

## Implementation Patterns and MLflow AI Gateway Setup

Token-aware gateway plugins follow a similar shape across implementations. A configuration typically defines a `limit`, a `time_window`, and a `limit_strategy` (total tokens, prompt tokens only, or completion tokens only), plus the key a rule applies to. APISIX's [ai-rate-limiting plugin](https://apisix.apache.org/docs/apisix/plugins/ai-rate-limiting/) is a clear real-world example: it supports `total_tokens`, `prompt_tokens`, and `completion_tokens` strategies, rule-based keys for per-tenant scoping, and response headers reporting remaining quota and reset time. Zuplo's approach takes a similar multi-counter idea further, letting a single request increment several named counters at once (requests, input tokens, output tokens) against per-plan limits.

For state storage, Redis is the default choice for most teams: fast, well-understood, and good enough for counters that live in a short window. DynamoDB becomes more attractive once you need cross-region consistency or longer-lived counters tied to monthly budgets. Whichever backend you pick, decide upfront whether you're pre-deducting tokens on admission (safer, but requires crediting back failed requests) or reconciling after the fact (simpler, but leaves a window where you can overshoot).

This is where [MLflow's AI Gateway](https://mlflow.org/genai/ai-gateway) fits the picture directly: centralized token accounting across providers, deep tracing of agentic reasoning so you can see exactly where token spend is going inside a multi-step agent workflow, and governance controls that apply consistently across every team hitting the gateway rather than living in five different app configs. Combined with MLflow's guidance on [tracking tokens](https://mlflow.org/articles/tags/how-to-track-tokens), it gives you the accounting layer that request-count limiting alone can never provide.

## Rate Limiting Checklist and Default Settings to Start With

Before shipping, confirm the basics are in place:

1. Enforce TPM and RPM together, never just one.
2. Set a hard budget cap (daily or monthly) beneath both.
3. Use separate API keys for interactive versus batch workloads.
4. Implement priority queuing so low-priority jobs can't starve high-priority ones.
5. Enable provider-usage reconciliation on a recurring schedule.

| Tier (example, tune to your traffic) | Monthly token budget                                                | TPM band                                              |
| ------------------------------------ | ------------------------------------------------------------------- | ----------------------------------------------------- |
| Free                                 | a moderate token budget typical of free plans                       | a typical token per minute band for small-scale use   |
| Pro                                  | a significantly higher token budget suitable for professional users | a higher token per minute band for medium-scale use   |
| Enterprise                           | a negotiated custom token budget                                    | a high token per minute band fitting enterprise needs |

These bands are illustrative starting points, not fixed rules; every workload profile is different. When a throttling incident hits, triage in order: confirm which dimension tripped (RPM, TPM, or spend), check `finish_reason` for truncation damage already done, isolate the offending tenant or workload, and only then decide whether to raise a limit or fix the calling pattern.

## Rate Limiting Is a Distributed-Systems Problem, Not a Config Flag

Most teams treat rate limiting as something you bolt on after an outage: add a counter, catch the 429, ship it. That's backwards. Quota is shared capacity, the same way CPU and memory are shared capacity in any multi-tenant system, and it needs the same discipline: partitioning, priority scheduling, and observability built in from day one, not stapled on after the first incident retro.

The teams that get burned repeatedly are the ones treating every rate limit fix as reactive. Building gateway-level token metering and dashboards before you need them costs a few days of engineering time up front. Rebuilding your entire request pipeline mid-incident, with a frustrated on-call engineer and an angry enterprise customer, costs a lot more. If there's one thing worth prioritizing this quarter, it's token metering and quota partitioning, not the next feature.

> _— Kevin_

## Manage Rate Limits Without Building the Plumbing Yourself

Most teams reinvent the same token-accounting layer described above, then spend months debugging tokenizer drift and reconciliation bugs that a purpose-built gateway already solved. MLflow's open-source [AI Gateway](https://mlflow.org/ai-gateway) gives you centralized token metering, cross-provider routing, and governance without writing a custom Redis counter from scratch, and because it's fully open-source under Linux Foundation governance, you get the entire feature set without an enterprise paywall gating the parts you actually need.

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

It's a strong fit if your team needs centralized quota governance across multiple services, deep tracing into agentic reasoning so you can see where token spend is actually going, and automated evaluation to catch quality regressions before they reach production. If you're running multi-model audits across providers, a tool like BabyLoveGrowth's multi-LLM audit pairs well with that observability layer. Start by exploring the [MLflow homepage](https://mlflow.org) or head straight to the AI Gateway product page to see how the token accounting and routing setup maps onto your own provider mix.

## Docs Worth Bookmarking Before You Build

- [OpenAI's rate limit docs](https://developers.openai.com/api/docs/guides/rate-limits): the canonical reference for RPM, TPM, and RPD structure.
- [APISIX's ai-rate-limiting plugin](https://apisix.apache.org/docs/apisix/plugins/ai-rate-limiting/): a working token-aware gateway config to model your own after.
- llm-rate-guard on GitHub: open-source patterns for failover, token buckets, and priority queuing.

## Sources

- [OpenAI: Rate limits (developer docs)](https://developers.openai.com/api/docs/guides/rate-limits)
- [APISIX: ai-rate-limiting plugin docs](https://apisix.apache.org/docs/apisix/plugins/ai-rate-limiting/)
- [LLM rate limiting in production: token budgets and reconciliation](https://www.systemshardening.com/articles/kubernetes/llm-rate-limiting/)

## FAQ

### What Is the LLM Rate Limit?

An LLM rate limit is a cap providers place on how much you can use their API within a time window, typically expressed as requests per minute, tokens per minute, and requests per day. OpenAI's documentation confirms TPM is usually the tightest constraint since token count varies wildly between requests.

### Does DeepSeek Have a Rate Limit?

Providers in this category generally do enforce rate limits similar in structure to other major LLM APIs, typically scoped by requests and tokens per time window, though exact figures and tiers vary by provider and change over time. Check the specific provider's current documentation rather than relying on a fixed number, since limits shift with usage tier and account status.

### Is Kafka a Rate Limiter?

No. Kafka is a distributed messaging and streaming platform, not a rate limiter. It can help buffer and smooth request bursts as part of a larger architecture, but you'd still need dedicated rate-limiting logic, whether in a gateway or application code, to enforce token and request quotas.

### What Is the Purpose of Rate Limiting?

Rate limiting protects shared infrastructure from being overwhelmed by a single consumer, controls costs by capping spend, and keeps service predictable for every tenant sharing the same capacity. For LLM applications specifically, it also prevents token-heavy requests from silently starving smaller ones sharing the same quota.

### Should I Enforce Rate Limits at the Gateway or in My Application?

Gateway-level enforcement is generally the better default once you have more than one service calling an LLM, since it centralizes accounting and simplifies fallback routing. Application-level enforcement still makes sense for highly custom business rules that don't belong in a shared config.

## Recommended

- [AI Gateway](https://mlflow.org/genai/ai-gateway)
- [How to Prevent Runaway Agent Costs with MLflow AI Gateway](https://mlflow.org/blog/agent-costs-mlflow-gateway)
- [Route Claude Code Through MLflow AI Gateway](https://mlflow.org/blog/gateway-claude-code)
- [Control LLM Spend with AI Gateway Budget Alerts and Limits](https://mlflow.org/blog/gateway-budget-alerts-limits)
