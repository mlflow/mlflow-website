---
title: "AI Agent Evaluation vs Observability: What's the Difference?"
description: "AI agent evaluation scores whether an agent did the job well. Observability records what it did. Learn how they differ, where they overlap, and how to run both."
slug: ai-agent-evaluation-vs-observability
tags:
  [
    AI agent evaluation vs observability,
    agent observability,
    agent evaluation,
    LLM observability vs evaluation,
    online evaluation,
    offline evaluation,
    LLM-as-a-judge,
    AI agent tracing,
  ]
date: 2026-11-18
---

AI agent observability records what an agent did: every model call, tool call, and intermediate step, with latency, token counts, and errors. AI agent evaluation scores whether what it did was good, using code checks, LLM judges, or human review. Observability gives you the evidence. Evaluation gives you the verdict. You need both, and the traces you collect for the first are the raw material for the second.

The failure this guide is about is a quiet one: the dashboard stays green, every status code is 200, and the agent still gives a wrong answer. Observability alone cannot catch that, and the two words are often used as if they meant the same thing, so we wrote down where the line sits. It is for engineers who run an agent in production, or are about to.

Disclosure: this guide was written by MLflow maintainers. The examples use MLflow Tracing and `mlflow.genai.evaluate`, and the concepts hold in any tool.

## TL;DR

Six points to take away.

- **Observability answers "what happened?"** It captures traces of each run: inputs, outputs, tool calls, latency, cost, and errors.
- **Evaluation answers "was it good?"** It applies graders (scorers) to an agent's output and its trajectory, and returns scores you can compare across versions.
- **They share one data model.** A trace is the input to both a dashboard and a scorer, so tracing is the first step for either.
- **Observability is reactive; offline evaluation is proactive.** Anthropic's engineering team says production monitoring detects problems after users hit them, while automated evals run before launch and on every commit.
- **Online evaluation is the bridge.** It scores a sample of live traces with the same judges you use in CI, so production monitoring reports quality, not only errors.
- **A healthy agent can still be wrong.** Zero errors and fast responses say nothing about whether the answer was correct.

## Table of Contents

- [What is AI agent observability?](#what-is-ai-agent-observability)
- [What is AI agent evaluation?](#what-is-ai-agent-evaluation)
- [How do evaluation and observability differ?](#how-do-evaluation-and-observability-differ)
- [Where do the two overlap?](#where-do-the-two-overlap)
- [Do you need both for an AI agent?](#do-you-need-both-for-an-ai-agent)
- [A worked example: one trace, two jobs](#a-worked-example-one-trace-two-jobs)
- [How to run both with MLflow](#how-to-run-both-with-mlflow)
- [Common mistakes](#common-mistakes)
- [Is "AI observability" the same as "LLM observability"?](#is-ai-observability-the-same-as-llm-observability)
- [Where to start](#where-to-start)
- [FAQ](#faq)

## What is AI agent observability?

AI agent observability is the practice of recording enough about every agent run that you can reconstruct and inspect it afterwards. For an agent, that means a trace: a structured record of the request, each model call, each tool call with its arguments and result, and the final response.

MLflow's tracing documentation describes the same idea: tracing captures "the inputs, outputs, and metadata associated with each intermediate step of a request," and it records latency and token usage at each step. The same page describes MLflow Tracing as fully OpenTelemetry-compatible, which matters if your traces need to flow into tools you already run.

Observability is built for questions like these:

- Why did this request take 40 seconds?
- Which tool call failed, and with what arguments?
- How many tokens did this user session cost?
- Which step did the agent take just before it looped?

Notice what these questions have in common. Each one is about the run, not about whether the result was right. A trace can show that an agent called `issue_refund` with a valid order ID and got a `200` back. It cannot tell you that the customer was not eligible for a refund.

> **Quick answer:** Agent observability tells you what an agent did on a given run: the steps, tool calls, latency, cost, and errors. It does not, by itself, tell you whether the outcome was correct.

## What is AI agent evaluation?

AI agent evaluation is the practice of measuring whether an agent does its job well, using repeatable tests that return scores. Anthropic's engineering team defines an eval as "a test for an AI system: give an AI an input, then apply grading logic to its output to measure success" in its post [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) (January 9, 2026).

Three terms from that post are worth borrowing:

- **Grader:** "logic that scores some aspect of the agent's performance." A single task can have several graders.
- **Transcript (also called a trace or trajectory):** "the complete record of a trial, including outputs, tool calls, reasoning," and intermediate results.
- **Outcome:** the final state of the environment. A booking agent's outcome is whether the reservation exists in the database, not whether the agent says it does.

The same post says agent evaluations typically combine three grader types: code-based, model-based, and human. Code-based graders are fast, cheap, and objective but brittle. Model-based graders handle nuance but are non-deterministic and need calibration against human judgment. Human graders are the most reliable, but they are expensive and slow, so they are mostly used to calibrate the model-based ones.

Evaluation is built for questions like these:

- Did the new prompt make answers more correct than the old one?
- Does the agent pick the right tool for this request?
- Did the change we shipped on Tuesday cause a regression?
- Is this LLM judge's score trustworthy?

> **Quick answer:** Agent evaluation applies graders to an agent's outputs and trajectories to produce scores. It tells you whether a run, or a version of the agent, is good enough, and lets you compare versions.

## How do evaluation and observability differ?

The cleanest split is by the question each one answers and the moment it answers it.

| Dimension             | Observability                                       | Evaluation                                                      |
| --------------------- | --------------------------------------------------- | --------------------------------------------------------------- |
| Core question         | What happened on this run?                          | Was the result good, and is this version better?                |
| Main artifact         | Traces, spans, metrics, logs                        | Scores, pass rates, regression reports                          |
| Who judges            | A person reading a trace, or alert rules on metrics | Scorers: code checks, LLM judges, human labels                  |
| When it runs          | Continuously on live traffic                        | Offline on a dataset (CI, before release) and online on samples |
| Typical failure found | Errors, timeouts, loops, cost spikes                | Wrong answers, wrong tools, policy violations, regressions      |
| Needs ground truth?   | No                                                  | Sometimes (expected answers); judges can run without it         |
| Output is a           | Record you inspect                                  | Number you compare                                              |

> **Quick answer:** Observability is a record of behavior that you inspect. Evaluation is a measurement of quality that you compare. Observability answers "what happened?" and evaluation answers "was it good?"

A second difference is timing. Anthropic's post makes the point directly: production monitoring tracks "metrics and errors in live systems," and it is reactive, because problems reach users before you detect them. Evals run without real users and can run on every commit. In the post's words, automated evals are "especially useful pre-launch and in CI/CD," while monitoring "kicks in post-launch to detect distribution drift and unanticipated real-world failures." It concludes that the most effective teams combine the two.

The third difference is what a failure looks like. Observability surfaces loud failures: exceptions, timeouts, a tool returning an error. Agents also fail quietly. They answer fluently and wrongly, call a tool that was allowed but not appropriate, or skip a required step. A dashboard of green status codes will not show any of that. Scorers will, because they read the content and the trajectory rather than the status.

## Where do the two overlap?

The overlap is a single idea: the trace. Once an agent emits traces, three things become possible that were not possible before.

**Scorers can read the trace.** MLflow's documentation on evaluating traces says "Scorers have access to the complete MLflow traces, including spans, attributes, and outputs." That lets you grade the path as well as the answer, for example the tool call trajectory, sub-agent routing, or retrieved document recall.

**Production traces become test cases.** [Langfuse's documentation](https://langfuse.com/docs/evaluation/overview) describes evaluation as running both offline, "before you ship a change," and online, on live production traces. The point is not which tool you use. The point is that a failure you see in a production trace should end up as a case in your offline suite.

**Evaluation scores become observability data.** When a judge scores a sampled share of live traffic, quality becomes a time series next to latency and cost. You can alert on "correctness dropped from 0.91 to 0.78" the same way you alert on a latency spike.

```text
                 ┌──────────────────────────────┐
                 │        Agent in production   │
                 └──────────────┬───────────────┘
                                │ emits
                                ▼
                       ┌─────────────────┐
                       │     Traces      │  ← observability
                       └───┬─────────┬───┘
            inspect, alert │         │ sample + score
                           ▼         ▼
                 ┌──────────────┐  ┌──────────────────┐
                 │  Dashboards  │  │  Online judges   │  ← evaluation
                 └──────────────┘  └────────┬─────────┘
                                            │ failures become test cases
                                            ▼
                                   ┌──────────────────┐
                                   │ Offline dataset  │ → CI gate
                                   └──────────────────┘
```

## Do you need both for an AI agent?

Yes, for any agent that does real work. Each one covers a blind spot in the other.

**Observability without evaluation** leaves you able to explain every failure you notice and unable to notice the quiet ones. You find out about wrong answers from users, which is the slowest and most expensive channel. You also have no way to say whether Tuesday's prompt change helped.

**Evaluation without observability** gives you scores with no explanation. A correctness score of 0.72 is a number. The trace behind a failing case is the reason. Without traces you cannot see which step went wrong, and your dataset will be built from imagined inputs rather than real ones.

A rule of thumb that holds up in practice:

| If you are...                              | Start with                                               |
| ------------------------------------------ | -------------------------------------------------------- |
| Prototyping, no users yet                  | Tracing on, plus a small offline eval set                |
| Shipping to a first group of real users    | Tracing, error and latency alerts, sampled online judges |
| Changing prompts, models, or tools often   | An offline eval suite as a CI gate                       |
| Running a multi-step or multi-agent system | Trace-reading scorers for tool calls and routing         |

> **Quick answer:** Use observability to see and debug what your agent does. Use evaluation to measure whether it does it well. Run offline evaluation before each release, and online evaluation on a sample of live traces after.

## A worked example: one trace, two jobs

Take a support agent that can look up orders and issue refunds. A customer writes, "My order arrived broken, I want my money back."

The agent looks up the order, decides the order qualifies, calls `issue_refund`, and replies "Done, your refund is on its way."

**What observability shows.** The trace has four spans: the user request, a model call that chose the `lookup_order` tool, the tool call itself, and a second model call that chose `issue_refund`. Total latency is 3.2 seconds, the cost is a few thousand tokens, and every status code is `200`. Nothing is red.

**What evaluation shows.** A scorer that checks policy finds that the order was placed 94 days ago and the refund window is 30 days. The agent should have escalated to a human. A second scorer, which checks whether the response promised something the policy does not allow, fails too.

Neither tool alone tells the full story. The dashboard says the system is healthy. The score says the behavior is wrong. The trace then lets you see exactly where: the model never compared the order date with the policy. That is the loop in practice. Evaluation finds the problem, observability explains it, and the failing trace becomes a permanent test case.

(This scenario is an illustration we constructed, not data from a customer.)

## How to run both with MLflow

MLflow is an open-source AI engineering platform governed by the Linux Foundation, and it uses one trace for both jobs. The steps below use APIs from the [MLflow GenAI documentation](https://mlflow.org/docs/latest/genai/). Treat the code as a starting point and check parameter names against the docs for your installed version.

### Step 1: Trace the agent

Add the decorator to the functions that make up a run, and mark tools as tool spans.

```python
import mlflow

mlflow.set_experiment("support-agent")


@mlflow.trace(span_type="TOOL")
def lookup_order(order_id: str) -> dict:
    ...


@mlflow.trace(span_type="TOOL")
def issue_refund(order_id: str) -> dict:
    ...


@mlflow.trace(span_type="AGENT")
def support_agent(question: str) -> str:
    ...
```

For supported libraries, `mlflow.<library>.autolog()` traces calls with one line, for example `mlflow.openai.autolog()`. For production, the tracing docs recommend the lightweight `mlflow-tracing` package, which they describe as a 95% smaller footprint than the full `mlflow` package, and with async logging enabled, trace logging runs in the background so it does not slow the application. For more on instrumenting, see our pages on [agent observability](https://mlflow.org/genai/observability) and [LLM tracing](https://mlflow.org/llm-tracing).

### Step 2: Evaluate offline against a dataset

An evaluation has three parts: a dataset, scorers, and a predict function. The dataset holds `inputs` and optional `expectations`.

```python
from mlflow.genai.scorers import Correctness, Guidelines

dataset = [
    {
        "inputs": {"question": "My order from 94 days ago arrived broken. Refund me."},
        "expectations": {"expected_response": "Escalates to a human; no refund issued."},
    },
]

results = mlflow.genai.evaluate(
    data=dataset,
    predict_fn=support_agent,
    scorers=[
        Correctness(),
        Guidelines(
            name="refund_policy",
            guidelines="Never promise a refund for an order older than 30 days.",
        ),
    ],
)
```

This is the same pattern we describe in [Continuous LLM Evaluation](https://mlflow.org/articles/continuous-llm-evaluation), including how to build the dataset and wire the run into CI.

### Step 3: Score the trajectory, not only the answer

Built-in scorers include `ToolCallCorrectness`, which judges whether the agent called the right tools with the right arguments, `ToolCallEfficiency`, which flags redundant tool calls, and `ConversationCompleteness` and `UserFrustration` for multi-turn conversations. You can also write a code scorer that reads the trace directly.

```python
from mlflow.entities import SpanType, Trace
from mlflow.genai.scorers import ToolCallCorrectness, scorer


@scorer
def at_most_one_refund(trace: Trace) -> bool:
    refunds = trace.search_spans(span_type=SpanType.TOOL, name="issue_refund")
    return len(refunds) <= 1


results = mlflow.genai.evaluate(
    data=dataset,
    predict_fn=support_agent,
    scorers=[ToolCallCorrectness(), at_most_one_refund],
)
```

Code scorers are free and exact, so use them for hard rules like "never refund twice". Use LLM judges for qualities code cannot express. If you rely on judges, calibrate them first; our guide to [LLM judge calibration](https://mlflow.org/articles/llm-judge-calibration) shows how.

### Step 4: Evaluate traces you already have

You do not need to re-run the agent to score it. Traces from `mlflow.search_traces()` can be passed to `mlflow.genai.evaluate()` directly, with no `predict_fn`. The documentation notes the function "will automatically extract the inputs, outputs, and other intermediate information from the trace object."

```python
from datetime import datetime, timedelta

since = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
traces = mlflow.search_traces(filter_string=f"timestamp > {since}")

results = mlflow.genai.evaluate(data=traces, scorers=[ToolCallCorrectness()])
```

This is how you turn yesterday's production traffic into a quality report, and how you reuse one set of traces across several evaluation runs without paying to regenerate them.

### Step 5: Score live traffic automatically

MLflow's automatic evaluation runs LLM judges on traces and multi-turn conversations as they are logged. It runs asynchronously, so it does not slow the application. You register a judge that uses a gateway model, then start it with a sample rate. Replace `my-llm-endpoint` with the name of your own AI Gateway endpoint.

```python
from mlflow.genai.scorers import Guidelines, ScorerSamplingConfig

refund_judge = Guidelines(
    name="refund_policy",
    guidelines="Never promise a refund for an order older than 30 days.",
    model="gateway:/my-llm-endpoint",
).register()

refund_judge.start(sampling_config=ScorerSamplingConfig(sample_rate=0.2))
```

The documentation lists prerequisites: a running MLflow server, tracing enabled, and a configured AI Gateway endpoint for the judge model. Multi-turn evaluation needs session IDs on the traces. It suggests a high sample rate during development and a lower one in production to control cost. The docs also note that failed evaluations are not retried automatically, so treat a missing score as "not evaluated", not "passed".

With all five steps in place, one experiment holds traces you can inspect, offline runs you can compare, and live quality scores you can chart.

## Common mistakes

**Treating a green dashboard as a quality signal.** Latency, error rate, and token counts describe health. They say nothing about correctness. Add at least one scorer that reads the content.

**Evaluating only the final answer.** An agent can reach the right answer through a wrong or wasteful path, such as calling a destructive tool when a read-only one would do. Score the trajectory too.

**Building the eval set from imagined inputs.** Real traces show what users actually ask. Seed your dataset from them and add every production failure you find.

**Trusting an uncalibrated judge.** A model-based grader is a second black box. Compare its scores to human labels on a sample before you gate a release on it.

**Using the same sample for tuning and reporting.** If you iterate on prompts until the offline suite passes, the suite stops measuring generalization. Keep a held-out set and refresh it from production.

**Scoring every live trace forever.** Judge calls cost money and add load to your gateway. Start with a high sample rate, then lower it once the signal is stable.

**Buying two tools when one data model would do.** Many platforms in this space, including Langfuse, LangSmith, Arize Phoenix, Braintrust, and MLflow, offer both tracing and evaluation. The useful question is whether the same trace feeds both, so you do not copy data between systems. Our [comparison of agent observability tools](https://mlflow.org/top-5-agent-observability-tools) and [agent evaluation frameworks](https://mlflow.org/top-5-agent-evaluation-frameworks) cover how each one handles this.

## Is "AI observability" the same as "LLM observability"?

Not quite, and the difference matters for the words you search with. LLM observability usually means watching individual model calls: prompts, completions, latency, tokens, and cost. AI observability, or agent observability, covers the larger system around those calls: tool use, retrieval, multi-step plans, multiple agents, and sessions that span many turns. A single agent run contains many LLM calls, and the failures that matter often sit between them.

In practice the tooling is the same. If your traces capture the model calls, the tool calls, and how they nest, you have both. We cover the model-call layer in more depth in our guide to [open-source LLM tracing tools](https://mlflow.org/articles/best-open-source-llm-tracing-tools-2026).

## Where to start

If you have no tracing today, turn it on first. Everything else depends on it, and it costs one decorator or one `autolog()` line. If you have tracing and no evaluation, pick the three worst failures you have seen in production, turn them into dataset rows, and write one scorer for each. That is a real eval suite, and you can grow it from there.

Then run it. The [MLflow GenAI documentation](https://mlflow.org/docs/latest/genai/) has quickstarts for tracing and evaluation, and the [MLflow GitHub repository](https://github.com/mlflow/mlflow) is the place to report issues or ask questions.

## FAQ

### What is the difference between AI agent evaluation and observability?

Observability records what an agent did on a run: its steps, tool calls, latency, cost, and errors. Evaluation scores whether that behavior was good, using code checks, LLM judges, or human review. Observability is something you inspect, and evaluation is something you compare across runs and versions.

### Is agent evaluation part of observability?

They are separate practices that share data. Some platforms bundle them, and some treat online evaluation as part of observability because it scores live traces. A useful test: if it tells you what happened, it is observability. If it returns a quality score you can compare, it is evaluation.

### What is the difference between offline and online evaluation?

Offline evaluation runs your agent against a fixed dataset before you ship a change, typically in CI. Online evaluation scores a sample of live production traces after release. Offline catches regressions early, and online catches problems your dataset did not anticipate.

### Can I do agent evaluation without tracing?

You can score final outputs without traces, but you lose the ability to grade tool calls, routing, and other intermediate steps, and you lose the explanation for a failing score. For agents, tracing is the better foundation.

### Do I need an LLM judge to evaluate an agent?

No. Code-based checks cover hard rules like valid output format, required tool calls, and latency budgets, and they are cheap and exact. LLM judges are for qualities code cannot express, such as helpfulness or tone, and should be calibrated against human labels.

### Which comes first, observability or evaluation?

Observability, in most cases. Tracing is quick to add, and the traces it collects supply the inputs for your first evaluation dataset. Add scorers as soon as you have traces from real or realistic usage.

### Does MLflow support both agent evaluation and observability?

Yes. Open-source MLflow provides OpenTelemetry-compatible tracing and `mlflow.genai.evaluate` with built-in and custom scorers. The same traces can be inspected in the UI, passed to evaluation, and scored automatically as they arrive. MLflow is Apache-2.0 licensed and hosted by the Linux Foundation. Any managed MLflow offering from a vendor is a separate product from the open-source project.

## Next step

Add `@mlflow.trace` to one agent function today, run a handful of real requests, and open the MLflow UI to read the traces. Then pick one failure you found and write the first scorer for it with the [MLflow evaluation guide](https://mlflow.org/genai/evaluations).
