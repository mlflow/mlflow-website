---
title: "How to Evaluate an AI Agent Using Production Traces"
description: "Evaluate an AI agent on real production traces: score tool calls with trace-based judges, grade whole conversations, and turn failures into regression tests."
slug: evaluate-ai-agent-production-traces
tags:
  [
    evaluate ai agent production traces,
    agent evaluation,
    trace-based evaluation,
    multi-turn agent evaluation,
    tool call evaluation,
    online evaluation,
    MLflow evaluation,
  ]
date: 2026-11-14
---

To evaluate an AI agent using production traces, record every run as a trace, pull a sample of those traces, and score them without re-running the agent. Use trace-based judges for the path the agent took (which tools it called, with what arguments) and session-level judges for whole conversations. Then turn each failure into a test case, so the next version has to pass it.

**Last reviewed: November 14, 2026.**

This guide is for engineers who already have an agent in production and a tracing setup, and want to know whether the agent is any good. We work on MLflow, so the code uses MLflow. The steps carry over to any tool that stores full agent traces.

## What you will build

- A tagged trace stream, with tool calls and conversation sessions recorded
- Trace-level scores for tool selection, tool arguments, and efficiency
- Session-level scores for completeness and user frustration
- A custom judge for one failure mode that matters to your product
- A loop that sends failed production traces back into a dataset
- Automatic scoring of a sample of live traffic

If you need the broader program around this, such as CI gates, golden datasets, and a 30-day rollout, read [Continuous LLM Evaluation](/articles/continuous-llm-evaluation/) first. This guide does not repeat it. It goes deeper on one question: how do you judge an _agent_ from what it did in production?

## Why can't you evaluate an agent on its final answer alone?

An agent can return a correct answer after a wrong path: a tool called with the wrong arguments, a retry loop that burned tokens, or a lookup that returned nothing and was ignored. Scoring the answer alone misses all of it, and the path is where agents break.

Reliability is the second problem. In the [τ-bench paper](https://arxiv.org/abs/2406.12045), researchers tested tool-using agents in simulated customer-service domains. They report that the function-calling agents of the time, including gpt-4o, succeed on fewer than 50% of tasks, and are inconsistent: pass^8, the chance of solving the same task correctly in all of 8 attempts, is below 25% in the retail domain.

One passing run says little. You need many real runs, scored the same way, and production is the only place you get them in volume. The paper is from 2024 and tested gpt-4o-class models, so newer models will score higher, but the lesson about repeated runs has not changed.

## What is trace-based agent evaluation?

Trace-based agent evaluation scores an agent from the recorded trace of its run, instead of re-running it on a test input. A trace holds the user input, every model call, every tool call with its arguments and result, and the final output, so a judge can grade the whole path.

This has three practical benefits:

- **It scores real behavior.** Production users ask things you did not predict.
- **It is cheap to repeat.** You generate traces once and re-score them with new judges, without paying for the agent's model calls again. The [MLflow docs on evaluating traces](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/traces/) note this reuse for offline evaluation.
- **It sees the intermediate steps.** Scorers get the full trace, including spans and attributes, so they can check the tool-call trajectory and not only the output.

Anthropic's engineering team makes the same split in [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents). Offline evals run on thousands of tasks without touching users, but "can create false confidence if it doesn't match real usage patterns." Production monitoring "reveals real user behavior at scale" but is reactive. Their advice is to combine automated evals for iteration, production monitoring for ground truth, and periodic human review for calibration.

## Which tool should you use for production-trace evaluation?

MLflow is a good fit when you want tracing, judges, datasets, and automatic scoring in one open-source server. It is Apache-2.0 licensed and a Linux Foundation project.

Disclosure: this guide was written by MLflow maintainers. Read the MLflow-specific code as one option. Other platforms such as Langfuse, LangSmith, and Braintrust take their own approach to scoring production traces, and our [comparison of agent observability tools](/top-5-agent-observability-tools) covers the trade-offs. The workflow below does not depend on the tool.

| Step                | What you score                          | MLflow API                                                       |
| ------------------- | --------------------------------------- | ---------------------------------------------------------------- |
| Record              | Nothing yet; capture tools and sessions | `@mlflow.trace(span_type="TOOL")`, `mlflow.update_current_trace` |
| Score the path      | Tool choice, arguments, redundant calls | `ToolCallCorrectness`, `ToolCallEfficiency`                      |
| Score the answer    | Relevance to the request                | `RelevanceToQuery`                                               |
| Score the dialogue  | Completeness, frustration across turns  | `ConversationCompleteness`, `UserFrustration`                    |
| Score a custom case | A failure only your product has         | `make_judge` with `{{ trace }}`                                  |
| Keep scoring        | A sample of live traffic                | `Scorer.register()` and `.start()`                               |

## Step 1: How do you make traces that an evaluator can use?

Mark tool functions with `span_type="TOOL"` and tag each trace with a session ID. The built-in tool-call judges need at least one span of type `TOOL`, and the conversation judges need a session ID to group turns.

```python
import mlflow

mlflow.set_experiment("support-agent")


@mlflow.trace(span_type="TOOL")
def lookup_order(order_id: str) -> dict:
    return {"order_id": order_id, "status": "shipped"}


@mlflow.trace(span_type="AGENT")
def support_agent(question: str, session_id: str) -> str:
    mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
    order = lookup_order("A-1042")
    return f"Your order {order['order_id']} has {order['status']}."
```

If you use a framework such as LangGraph or the OpenAI SDK, call its `autolog()` function instead of decorating by hand, and still set the session ID. For the full list of what a trace captures, see our page on [LLM and agent observability](https://mlflow.org/genai/observability).

Put the session ID on every turn. Without it, a five-turn conversation looks like five unrelated requests, and conversation-level scores are impossible.

## Step 2: How do you pull production traces to evaluate?

Search the tracking server for a recent window and ask for trace objects. A recent window means you score the agent you actually shipped.

```python
from datetime import datetime, timedelta

since = datetime.now() - timedelta(days=1)
traces = mlflow.search_traces(
    filter_string=f"timestamp > {int(since.timestamp() * 1000)}",
    max_results=200,
    return_type="list",
)
```

The [trace search guide](https://mlflow.org/docs/latest/genai/tracing/search-traces/) lists the supported filters, such as status and tags. Do not sample only successes. Errors, slow runs, and runs where a user gave a thumbs-down are the cases you most need to see, so include them on purpose.

## Step 3: How do you score tool calls from a trace?

Use the two built-in tool-call judges. `ToolCallCorrectness` checks whether the agent picked the right tools and passed correct arguments for the request. `ToolCallEfficiency` checks for redundant calls. Neither needs ground-truth labels in its default mode.

```python
from mlflow.genai.scorers import (
    RelevanceToQuery,
    ToolCallCorrectness,
    ToolCallEfficiency,
)

results = mlflow.genai.evaluate(
    data=traces,
    scorers=[
        ToolCallCorrectness(),
        ToolCallEfficiency(),
        RelevanceToQuery(),
    ],
)
```

You do not pass a `predict_fn`. MLflow reads the inputs, outputs, and spans from each trace. The judges are LLM-based, and the default judge model is an OpenAI model, so set `OPENAI_API_KEY` or pass `model=` to use another provider (on a Databricks tracking server the default differs). See the [tool-call judge docs](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call/) for modes that compare against expected tool calls.

Open the run in the MLflow UI and read the rationale on the lowest-scoring traces. Anthropic's guide says plainly that "you won't know if your graders are working well unless you read the transcripts and grades from many trials."

## Step 4: How do you evaluate multi-turn conversations?

Group traces into sessions and use the session-level judges. `ConversationCompleteness` returns whether the assistant addressed all of the user's requests by the end. `UserFrustration` returns `none`, `resolved`, or `unresolved`, so you can tell a rough patch that recovered from one that ended badly.

```python
from mlflow.genai.scorers import ConversationCompleteness, UserFrustration

sessions = mlflow.search_sessions(max_results=50)
session_traces = [trace for session in sessions for trace in session]

results = mlflow.genai.evaluate(
    data=session_traces,
    scorers=[ConversationCompleteness(), UserFrustration()],
)
```

Multi-turn evaluation needs MLflow 3.10 or later, and the docs mark the feature as experimental, so expect the API to change. Our post on [multi-turn evaluation](https://mlflow.org/blog/multiturn-evaluation) walks through the sessions view and a simulation workflow for conversations you have not seen yet.

Session scores catch things that no single turn shows: an agent that forgets a constraint from turn two, or answers each question well while the user's real goal never gets met.

## Step 5: How do you write a judge for a failure only your agent has?

Write a trace-based judge with `make_judge`. When the instructions contain `{{ trace }}`, the judge gets the full trace and can inspect spans on its own. Name one specific failure, give clear categories, and keep it to one question per judge.

```python
from typing import Literal

from mlflow.genai.judges import make_judge

refund_judge = make_judge(
    name="refund_policy_followed",
    instructions=(
        "Analyze the execution {{ trace }}. Decide whether the agent checked "
        "the order status with a tool before promising a refund.\n"
        "Answer 'checked' if a lookup tool ran before the refund promise, "
        "'skipped' if the agent promised a refund without a lookup, and "
        "'not_applicable' if no refund was discussed."
    ),
    feedback_value_type=Literal["checked", "skipped", "not_applicable"],
    model="openai:/gpt-5-mini",
)

results = mlflow.genai.evaluate(data=traces, scorers=[refund_judge])
```

Set a model explicitly on a trace-based judge, as in the example. The [custom judge docs](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges/create-custom-judge/) show more examples, including loop detection and tool optimization.

Do not trust a new judge on day one. The MT-Bench paper, [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685), found in 2023 that strong judges like GPT-4 reach over 80% agreement with human preferences on MT-Bench and Chatbot Arena, the same level humans agree with each other. The same abstract lists position, verbosity, and self-enhancement biases. Airbnb's engineering team recommends [a golden dataset of 50 to 100 examples that includes bad examples](https://airbnb.tech/ai-ml/eval-driven-development-lessons-from-evaluating-genai-at-scale/) for calibrating a judge, and the same mix works for a regression set. Label a set that size by hand, compare, and rewrite the instructions until the judge agrees with you. Our guide to [LLM judge calibration](/articles/llm-judge-calibration) covers the method, and the [LLM-as-a-judge overview](https://mlflow.org/llm-as-a-judge) explains the scorer types.

## Step 6: How do you turn failed production traces into regression tests?

Annotate the failing traces with the expected behavior, then add them to an evaluation dataset. The next version of the agent runs against that dataset, so a fixed failure stays fixed.

```python
from mlflow.genai.datasets import create_dataset

failing_ids = ["<failing-trace-id>"]
for trace_id in failing_ids:
    mlflow.log_expectation(
        trace_id=trace_id,
        name="expected_tool_calls",
        value=[{"name": "lookup_order", "arguments": {"order_id": "A-1042"}}],
    )

dataset = create_dataset(
    name="support_agent_regressions",
    experiment_id=mlflow.get_experiment_by_name("support-agent").experiment_id,
)
dataset.merge_records([mlflow.get_trace(trace_id) for trace_id in failing_ids])
```

Choose the traces you merge. A dataset full of easy cases hides regressions, so add the failures plus a few clean cases, which proves your scorers can tell good from bad.

To run the dataset against a new agent version, call `mlflow.genai.evaluate(data=dataset, predict_fn=support_agent, scorers=[...])`. The [dataset workflow guide](https://mlflow.org/docs/latest/genai/datasets/end-to-end-workflow/) shows the full path.

This is the loop that one paper proposes under the name evaluation-driven development: the paper on [evaluation-driven development and operations of LLM agents](https://arxiv.org/abs/2411.13768) argues that evaluation should run through development and operations as a continuous function, not a gate before launch.

## Step 7: How do you score live traffic automatically?

Register a judge and start it with a sampling rate. MLflow then scores new traces and sessions as they arrive, with no scheduled job to maintain. This is online evaluation, and it is how you notice drift without waiting for a user complaint.

```python
from mlflow.genai.scorers import ScorerSamplingConfig, ToolCallCorrectness

judge = ToolCallCorrectness(model="gateway:/my-llm-endpoint")
registered = judge.register(name="tool_call_correctness")
registered.start(sampling_config=ScorerSamplingConfig(sample_rate=0.2))
```

Three limits to know from the [automatic evaluation docs](https://mlflow.org/docs/latest/genai/eval-monitor/automatic-evaluations/):

- It supports LLM judges, built-in or from `make_judge`. Plain code scorers are not supported on a standard server.
- The judge model runs through an AI Gateway endpoint, which you configure first.
- A new judge scores traces and sessions that are at most one hour old. It does not backfill history, so use `mlflow.genai.evaluate` for that.

Start at a low sample rate. Judge calls cost money, and a small slice of traffic is enough to see a trend. Raise the rate for a new release, then lower it again.

## What mistakes should you avoid?

Most failed trace-evaluation programs fail in the same few ways.

- **Scoring only the answer.** You miss wrong tool calls that happened to produce a plausible reply.
- **Skipping the human check.** The [Who Validates the Validators](https://arxiv.org/abs/2404.12272) study describes "criteria drift": people need criteria to grade outputs, but grading outputs helps them define the criteria. Expect your rubric to change after you read real traces.
- **One judge for everything.** A single "is this good?" judge gives a vague score. Use one judge per failure mode.
- **Mixing up prompts and sampling.** If you change both a judge prompt and the sample at once, you cannot tell whether quality moved or the measurement did. Change one at a time.
- **Sampling only what users flagged.** Complaints are a biased sample. Score a random slice too.
- **Treating one run as a result.** Given the pass^k results above, repeat the important cases and look at the spread.

## When is this approach the wrong choice?

Trace-based evaluation needs real traffic, so it does not help before launch. For a new agent, build a small dataset by hand or simulate conversations, then switch to production traces once users arrive. It also needs privacy review: traces contain user input, so check what you store and who can read it before you point judges at it.

It does not replace outcome checks either. If you can verify the final state directly, such as a refund row in a database, write a code check and use the LLM judge only for what code cannot see. Anthropic's guidance is to choose deterministic graders where possible and LLM graders where necessary.

## FAQ

### How do I evaluate an AI agent using production traces?

Capture each run as a trace, pull a sample, and score it with judges that read the whole trace. In MLflow that means `mlflow.search_traces` to fetch the sample and `mlflow.genai.evaluate` with scorers such as `ToolCallCorrectness` and `UserFrustration`. Then add the failures to a dataset to guard against regressions.

### How is trace-based evaluation different from offline evaluation?

Offline evaluation runs the agent on a fixed dataset you chose in advance. Trace-based evaluation scores runs that already happened, so it reflects real user behavior but cannot test a new version by itself. Use both: traces to find failures, a dataset to test the fix.

### How many production traces do I need?

There is no fixed number. Start with the code above, which pulls up to 200 traces from the last day. Include errors and slow runs on purpose, and read the lowest scores by hand. Once your judges agree with your labels, add automatic scoring at a low sample rate.

### Can I evaluate multi-turn conversations, not just single requests?

Yes. Set a session ID on every trace, then use session-level judges such as `ConversationCompleteness` and `UserFrustration`. This needs MLflow 3.10 or later and is marked experimental in the docs.

### Do I need ground-truth labels?

Not to start. The default tool-call judges and the conversation judges work without labels. Labels help later: add expected tool calls to failing traces so your regression tests can check exact behavior.

### What does it cost to run LLM judges on production traffic?

Each judged trace is at least one extra model call, and a trace-based judge that explores a long trace costs more. Control it with the sample rate, a cheaper judge model for simple checks, and a filter string that targets only the traces you care about.

## Next step

Pick one agent and one failure mode this week. Add `span_type="TOOL"` to its tools, pull a day of recent traces, and run `ToolCallCorrectness` over them. Read the ten lowest scores, then follow the [MLflow guide to evaluating production traces](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/traces/) to build your first judge. For the monitoring side, see [MLflow AI monitoring](https://mlflow.org/ai-monitoring), and for the wider evaluation toolkit, see [LLM and agent evaluation in MLflow](https://mlflow.org/llm-evaluation).

## Sources

Every external claim in this guide comes from one of these sources.

- [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045), Yao et al., 2024
- [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents), Anthropic Engineering, January 9, 2026
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685), Zheng et al., 2023
- [Who Validates the Validators?](https://arxiv.org/abs/2404.12272), Shankar et al., 2024
- [Evaluation-Driven Development and Operations of LLM Agents: A Process Model and Reference Architecture](https://arxiv.org/abs/2411.13768), Xia et al., 2024
- [Eval-driven development: lessons from evaluating GenAI at scale](https://airbnb.tech/ai-ml/eval-driven-development-lessons-from-evaluating-genai-at-scale/), Airbnb Engineering
- MLflow documentation pages linked inline above, checked against MLflow 3.18.0 source
