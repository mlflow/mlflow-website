---
title: "API Error Handling for LLM Apps: Retries, Timeouts, Fallbacks"
description: "How to handle LLM API errors in production: classify retryable and terminal failures, set backoff, jitter and timeouts, add fallbacks, and trace every attempt."
slug: api-error-handling-llm
tags: [api error handling llm]
date: 2026-11-16
image: https://media.babylovegrowth.ai/blog-images/organization-30814/1794668471813_Engineer-reviewing-an-API-failure-at-her-laptop.jpeg
---

![Engineer reviewing an API failure at her laptop](https://media.babylovegrowth.ai/blog-images/organization-30814/1794668471813_Engineer-reviewing-an-API-failure-at-her-laptop.jpeg)

Good LLM API error handling comes down to four decisions: which errors you retry, how long you wait between attempts, when you stop or switch to another model, and how you record every attempt so you can debug it later. Get those four right and a provider's rate limit or overload becomes a short delay instead of an outage. Because we build [MLflow](https://mlflow.org/), an open-source platform for GenAI and agent observability, evaluation and gateway governance, we see error handling as a tracing problem as much as a coding one: a retry you cannot see is a retry you cannot tune.

This guide is written for engineers who already call model APIs from production services or agents. It gives you the classification table, a working retry wrapper, timeout guidance by workload, fallback patterns and the signals to log.

## TL;DR

- Sort errors into two piles. Timeouts, 408, 429 and 5xx are usually transient and retryable. 400 and 401 are permanent until you change the request or credentials.
- Use exponential backoff with jitter, a hard cap on attempts, and honor `Retry-After` whenever the provider sends it.
- Set timeouts per workload: interactive chat, agent loops and batch jobs have very different latency envelopes.
- Retry in one layer only, add a fallback model for sustained failures, and degrade gracefully when everything is down.
- Log every attempt with error type, status code, latency and request ID so retry behavior can be debugged rather than guessed.

## Table of Contents

- [Why LLM calls fail differently from ordinary API calls](#why-llm-calls-fail-differently-from-ordinary-api-calls)
- [Step 1: Classify errors as retryable or terminal](#step-1-classify-errors-as-retryable-or-terminal)
- [Step 2: Retry with exponential backoff, jitter and a cap](#step-2-retry-with-exponential-backoff-jitter-and-a-cap)
- [Step 3: Set timeouts by workload](#step-3-set-timeouts-by-workload)
- [Step 4: Make retries safe with idempotency in mind](#step-4-make-retries-safe-with-idempotency-in-mind)
- [Step 5: Add fallbacks, cooldowns and graceful degradation](#step-5-add-fallbacks-cooldowns-and-graceful-degradation)
- [Step 6: Trace every attempt](#step-6-trace-every-attempt)
- [Step 7: Control demand before the provider does](#step-7-control-demand-before-the-provider-does)
- [A checklist you can paste into a design review](#a-checklist-you-can-paste-into-a-design-review)
- [Where MLflow fits](#where-mlflow-fits)
- [Our perspective](#our-perspective)
- [FAQ](#faq)
- [Sources](#sources)

## Why LLM calls fail differently from ordinary API calls

An LLM API is a remote service with variable latency, token-based quotas and shared capacity. A single request can take two seconds or two minutes. A burst of traffic can exhaust a tokens-per-minute budget even when your request count looks modest. And in an agent, one failed call in step six of a ten-step reasoning chain can waste everything that came before it.

Three properties shape the handling:

1. **Failures are frequent enough to be normal.** Overload, rate limits and dropped connections are part of the operating environment, so the happy path is not the only path.
2. **Latency is wide.** A timeout that suits a chat reply will kill a long agentic call.
3. **Retries cost money and quota.** Every retry spends tokens and counts against the same limit that may have caused the failure.

## Step 1: Classify errors as retryable or terminal

The first decision is whether trying again can possibly help. Google's retry guidance for its generative AI service treats [408, 429, 5xx responses, socket timeouts and TCP disconnects as generally retryable](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/retry-strategy), while 400 and 401 point to permanent problems that should not be retried without a change. Fireworks gives a similar split: [429 and 500, 502, 503, 504 retry with backoff, while 400, 401, 404 and 422 should be corrected rather than retried](https://docs.fireworks.ai/guides/reliability).

| Status        | Typical meaning                | Action                                                                                            |
| ------------- | ------------------------------ | ------------------------------------------------------------------------------------------------- |
| 400, 422      | Malformed or invalid request   | Do not retry. Fix the payload or prompt construction.                                             |
| 401           | Bad or missing credentials     | Do not retry. Fix the key or its permissions.                                                     |
| 404           | Wrong model name or endpoint   | Do not retry. Fix the configuration.                                                              |
| 408, timeout  | Request took too long          | Retry with backoff, then review your timeout.                                                     |
| 429           | Rate limit or quota            | Retry after the wait the provider specifies, else backoff with jitter. Check which limit you hit. |
| 500           | Unexpected provider error      | Retry briefly. If it persists, capture the request ID for support.                                |
| 502, 503, 504 | Gateway, overload or timeout   | Retry with backoff, following `Retry-After` when present.                                         |
| 529           | Temporary overload (Anthropic) | Retry with backoff.                                                                               |

Anthropic's error documentation describes [529 as temporary overload, 504 as a timeout and 500 as an unexpected internal error](https://platform.claude.com/docs/pt-BR/api/errors), and recommends passing the request ID to support if a 500 keeps happening.

### The 429 that can be two different things

A 429 is the one status code that needs a second look. Anthropic notes that it can come from ordinary rate limiting or from [monthly usage-tier or workspace spending limits](https://platform.claude.com/docs/pt-BR/api/errors), and that a spending-cap 429 carries no `Retry-After` and continues until access resumes. Retrying that second kind in a loop just burns time. OpenAI's guide identifies 429 as the response when [requests or tokens exceed your assigned limit](https://developers.openai.com/api/docs/guides/error-codes) and advises pacing requests and checking project or organization usage.

Practical rule: if a 429 persists across your full retry budget and the response has no `Retry-After`, treat it as a quota problem, alert a human, and route traffic to a fallback instead of waiting.

### Parse the error body, not just the status

Providers return structured errors. Anthropic documents a JSON body with a top-level error object holding a `type` and `message`, plus a [`request_id` meant for tracking and debugging](https://platform.claude.com/docs/pt-BR/api/errors). Your classifier should read the error type where it exists. Status codes alone cannot tell a transient overload from a quota cap.

## Step 2: Retry with exponential backoff, jitter and a cap

Retries have three failure modes of their own: retrying too fast, retrying in lockstep, and retrying forever.

OpenAI's guidance for rate limits is to wait at least as long as `Retry-After` specifies; if it is absent, use [exponential backoff with jitter and limited retries](https://developers.openai.com/api/docs/guides/error-codes). It also says to retry 500 and 503 errors after a brief wait, following `Retry-After` for 503. Google adds the reasoning behind each piece: [exponential delays, jitter to desynchronize clients, a maximum number of attempts, and logging of attempts, error types and response times](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/retry-strategy).

- **Exponential backoff** spaces attempts further apart (for example 1, 2, 4, 8 seconds) so a struggling service gets room to recover.
- **Jitter** adds randomness so a thousand clients that failed together do not all retry together.
- **A maximum attempt count** bounds latency and cost. Three to five attempts is a common starting range; tune it from your own traces.
- **`Retry-After`** overrides your own schedule whenever it is present.

Here is a compact wrapper that follows those rules:

```python
import random
import time

RETRYABLE = {408, 429, 500, 502, 503, 504, 529}

class LLMError(Exception):
    def __init__(self, status, retry_after=None, request_id=None):
        self.status = status
        self.retry_after = retry_after
        self.request_id = request_id

def call_with_retries(fn, max_attempts=4, base=1.0, cap=30.0):
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except LLMError as e:
            if e.status not in RETRYABLE or attempt == max_attempts:
                raise
            if e.retry_after is not None:
                delay = e.retry_after
            else:
                delay = min(cap, base * 2 ** (attempt - 1))
                delay = random.uniform(0, delay)  # full jitter
            time.sleep(delay)
```

Three details matter in production. Convert provider exceptions into one error type at the edge, so the retry logic does not depend on any one SDK. Cap the wait so a bad `Retry-After` cannot hold a user request for minutes. And record each attempt (see Step 6), because the wrapper above is silent by default.

![Retry attempts with jitter and a maximum wait](https://media.babylovegrowth.ai/blog-images/organization-30814/1794668498573_Retry-attempts-with-jitter-and-a-maximum-wait.jpeg)

### Do not stack retry layers

Many official SDKs already retry. Anthropic reports that its SDK [retries transient connection, rate-limit and 5xx errors twice by default with exponential backoff](https://platform.claude.com/docs/pt-BR/api/errors), respecting `retry-after`. If you wrap that in your own loop of four attempts, you can make twelve calls for one user request, each spending quota. Pick one layer to own retries, set the other to zero, and write down which one it is.

## Step 3: Set timeouts by workload

A single global timeout is a bug. Fireworks' guide recommends values that depend on the job: [30 to 60 seconds for interactive chat, 5 to 30 minutes for agentic calls, 10 to 30 minutes for large-model inference, and 60 seconds for asynchronous batch submission](https://docs.fireworks.ai/guides/reliability). Treat those as starting points from one provider's documentation, not universal constants, and tune from your own latency distribution.

| Workload                               | Starting timeout | Why                                                   |
| -------------------------------------- | ---------------- | ----------------------------------------------------- |
| Interactive chat                       | 30 to 60 seconds | A user is waiting; fail fast and retry or fall back.  |
| Agent step with tools and long context | Minutes          | Reasoning and large outputs take longer.              |
| Large-model inference                  | 10 to 30 minutes | Long generations are legitimate.                      |
| Batch submission                       | About 60 seconds | Submission is quick; the work happens asynchronously. |

Timeouts and retries interact. A 60-second timeout with four attempts can hold a request for four minutes plus backoff. Work out your worst case and compare it with what the caller can tolerate. Use separate connect and read timeouts where your HTTP client supports them, so a dead connection is detected in seconds even when generation itself is allowed to take longer.

When a provider returns a client-closed style response, the fix is usually on your side: check the logs and raise the client-side timeout rather than adding retries.

## Step 4: Make retries safe with idempotency in mind

Retrying is only harmless when repeating the request has no side effects. Google warns that [retrying non-idempotent operations can create duplicate resources](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/retry-strategy). It also notes that a plain text generation call is not strictly idempotent because output is stochastic, yet it is generally safe to retry on transient errors because it does not modify server-side state.

The risk moves to what surrounds the call, especially in agents:

- **Tool calls with side effects.** If an agent step sends an email, creates a ticket or charges a card, retrying the whole step repeats the action. Retry the model call, not the tool execution, and give side-effecting tools an idempotency key.
- **Multi-step workflows.** Checkpoint completed steps so that a failure at step six resumes at step six instead of replaying one to five.
- **Streaming responses.** If a stream drops halfway, decide whether you can resume or must restart, and never show a user two concatenated partial answers.

## Step 5: Add fallbacks, cooldowns and graceful degradation

Retries handle brief trouble. Sustained trouble needs a different move. Three tools cover it, each answering a different question:

1. **Retry the same model** when the failure looks momentary.
2. **Fall back to a different model or provider** when the primary keeps failing or the quota is exhausted. Keep the prompt contract compatible: same output schema, similar context window, tested quality on your evaluation set.
3. **Cool down a failing deployment** so it is temporarily skipped instead of being tried first on every request. This is the circuit breaker idea: open the circuit when error rate or latency crosses a threshold over a short window, and probe it again after a pause.

When every option fails, degrade on purpose. Return a cached answer, a shorter non-LLM response, or a clear message that the feature is temporarily limited. Queue non-urgent work for later instead of failing it. Decide this behavior ahead of time per feature, because the right answer for a support chatbot differs from the right answer for a nightly summarization job.

One caution applies to every fallback: a cheaper or smaller fallback model can change output quality. Run your evaluation suite against each fallback path before you depend on it, not after the first incident.

## Step 6: Trace every attempt

Google's advice is to [monitor and log retry attempts, error types and response times](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/retry-strategy) so you can debug retry strategies. For LLM apps that means one trace per user request with a child span per model call, and for each attempt:

- status code and provider error type
- attempt number and the delay before it
- model and provider used, including fallback hops
- latency, input and output token counts
- the provider's request ID, so you can hand it to support for a persistent 500
- the final outcome: success, fallback success, degraded response or failure

With that data, questions that are otherwise guesswork become queries. What share of requests need a second attempt? Which provider returns the most 529s at peak? Did raising the timeout reduce failures or just move them? Does the fallback model hurt answer quality?

MLflow Tracing captures the reasoning and tool calls of agents and works with OpenTelemetry, so retries and fallback hops can sit in the same trace as the prompt, the tool calls and the final answer. See [AI observability for LLMs and agents](https://mlflow.org/ai-observability) for how that fits together.

## Step 7: Control demand before the provider does

Most 429s are avoidable. If your own traffic is the cause, shape it:

- Track tokens per minute as well as requests per minute, because token quotas often bind first.
- Queue and smooth bursts from batch jobs so they do not crowd out interactive traffic.
- Apply per-tenant or per-feature budgets so one noisy workload cannot exhaust the shared quota.
- Cache repeated prompts where output reuse is acceptable.

Our own articles on this theme go deeper: see the posts on [rate limiting LLM apps](https://mlflow.org/articles/tags/rate-limiting-llm-apis), [best practices for LLM rate limiting](https://mlflow.org/articles/tags/best-practices-for-llm-rate-limiting) and [per-tenant rate limits](https://mlflow.org/articles/tags/per-tenant-rate-limits).

## A checklist you can paste into a design review

1. Every provider error is mapped to one internal type with a retryable flag.
2. 400, 401, 404 and 422 are never retried.
3. 429 handling distinguishes short rate limits from quota caps.
4. Backoff is exponential with jitter, capped, with a maximum attempt count.
5. `Retry-After` is honored, with a ceiling.
6. Exactly one layer owns retries.
7. Timeouts are set per workload and the worst-case total is known.
8. Side-effecting tool calls carry idempotency keys.
9. A fallback model exists and has been evaluated.
10. Degraded behavior is defined per feature.
11. Every attempt is traced with status, latency, tokens and request ID.

## Where MLflow fits

MLflow is an open-source platform for the GenAI and agent lifecycle. For the problems above, three parts matter. Tracing gives you the per-attempt record from Step 6. [LLM-as-a-judge evaluation](https://mlflow.org/llm-as-a-judge) lets you score the answers your fallback paths produce, so a provider switch does not quietly lower quality. The AI Gateway provides centralized, cross-provider governance and prompt management, which is where provider switching belongs in a multi-team setup. You can explore the full picture on the [Agent and LLM engineering page](https://mlflow.org/genai).

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

## Our perspective

> Teams usually treat error handling as a coding chore and observability as a separate project. We think that split is the root of most painful LLM incidents. A retry policy is a hypothesis about how a provider behaves, and the only way to test a hypothesis is to look at what actually happened on every attempt.
>
> That is why we build tracing, evaluation and gateway governance into one open-source platform. Everything lives under Linux Foundation governance with no enterprise feature paywalls, and OpenTelemetry compatibility means your traces are not locked to one framework. When a fallback fires at 2 a.m., you should be able to see the failed call, the retry delay, the model that answered instead, and a quality score for that answer in one place.
>
> Our advice: start with the classification table, own retries in one layer, and trace from day one. Tuning comes easily once the data exists.
>
> _Kevin_

## FAQ

### Which LLM API errors should I retry?

Retry timeouts, 408, 429 and 5xx responses such as 500, 502, 503, 504 and Anthropic's 529, using backoff with jitter. Do not retry 400, 401, 404 or 422 until you have fixed the request, credentials or configuration, because the same call will fail the same way.

### How many times should I retry, and how long should I wait?

Use a small, bounded number, commonly three to five attempts, with exponential delays and jitter, and follow `Retry-After` whenever the provider sends it. Provider SDKs often retry a couple of times on their own, so count those attempts before adding your own.

### What timeout should I use for LLM calls?

It depends on the workload. Fireworks suggests 30 to 60 seconds for interactive chat and several minutes or more for agentic and large-model calls. Measure your own latency percentiles and set the timeout just above the slowest legitimate response.

### Is it safe to retry an LLM request?

For plain generation, generally yes on transient errors, since the call does not change server-side state. The danger is in agent steps with side effects such as sending messages or creating records. Retry the model call, and protect those actions with idempotency keys.

### When should I switch to a fallback model?

Switch when failures persist past your retry budget, when a 429 reflects a quota cap with no `Retry-After`, or when error rates cross a threshold in a short window. Evaluate the fallback model's output quality in advance so the switch is safe.

## Sources

- [Retry strategy, Generative AI on Vertex AI (Google Cloud)](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/retry-strategy)
- [Error codes (OpenAI API guides)](https://developers.openai.com/api/docs/guides/error-codes)
- [Claude API errors (Anthropic)](https://platform.claude.com/docs/pt-BR/api/errors)
- [Reliability and error handling (Fireworks)](https://docs.fireworks.ai/guides/reliability)
