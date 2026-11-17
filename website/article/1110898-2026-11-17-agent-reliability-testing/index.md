---
title: "Agent Reliability Testing: A Practical Guide for Engineers"
description: "How to test AI agent reliability: step, trajectory and outcome tests, pass^k consistency, regression suites and production monitoring, with MLflow in the loop."
slug: agent-reliability-testing
tags: [agent reliability testing]
date: 2026-11-17
image: https://media.babylovegrowth.ai/blog-images/organization-30814/1794747661945_Engineer-reviewing-an-agent-reliability-test.jpeg
---

![Engineer reviewing an agent reliability test](https://media.babylovegrowth.ai/blog-images/organization-30814/1794747661945_Engineer-reviewing-an-agent-reliability-test.jpeg)

Agent reliability testing answers one question: will this agent do the right thing, by an acceptable path, every time you run it, not just once? You answer it by running each task many times, scoring the final outcome and the full trace of reasoning and tool calls, and repeating those checks before every release and on live traffic. MLflow is built for that loop: tracing records what the agent did, and automated LLM-as-a-Judge evaluation scores it, in one open-source platform.

This guide is for teams shipping agents who already know how to write unit tests and now find that those tests say little about a system that plans, calls tools and behaves differently on each run.

## TL;DR

- Reliability is a distribution, not a pass/fail bit: run every task several times and report consistency (pass^k), not a single success rate.
- Test at three levels: single steps, whole trajectories and final outcomes. Score the end state of the environment, not the agent's closing message.
- Small per-step error compounds quickly: 95% per step is only about 36% over 20 steps.
- Turn every production incident into a regression case, and monitor success, tool errors, latency, tokens and retries on live traffic.
- With MLflow you can trace each run, score traces with LLM judges and keep the same checks running from CI to production.

## Table of Contents

- [What reliability testing means for an agent](#what-reliability-testing-means-for-an-agent)
- [Why per-step accuracy is the wrong number](#why-per-step-accuracy-is-the-wrong-number)
- [The three levels of agent tests](#the-three-levels-of-agent-tests)
- [Build the test set from real tasks](#build-the-test-set-from-real-tasks)
- [Trace everything, then grade the traces](#trace-everything-then-grade-the-traces)
- [Diagnose failures before changing anything](#diagnose-failures-before-changing-anything)
- [From pre-release gates to production monitoring](#from-pre-release-gates-to-production-monitoring)
- [A two-week starting plan](#a-two-week-starting-plan)
- [Promo: tracing, evaluation and gateway in one open-source platform](#promo-tracing-evaluation-and-gateway-in-one-open-source-platform)
- [Our perspective](#our-perspective)
- [FAQ](#faq)
- [Sources](#sources)

## What reliability testing means for an agent

In classic software engineering, [reliability testing](https://zetcode.com/terms-testing/reliability-testing/) evaluates how consistently and dependably a system behaves over time, rather than whether a single call returns the right value. The idea carries over to agents, but the object under test changes.

An agent acts over many steps with limited human involvement. It chooses tools, passes arguments, reads results and decides what to do next. Singapore's IMDA, in its [Starter Kit for Testing Agentic Systems](https://www.imda.gov.sg/assets/603572eb-724a-46cf-ac12-14fe38e1558c.pdf), notes that this differs from conventional software testing because new components such as tool usage need testing and the entire trajectory has to be evaluated, not only the answer.

Three consequences shape everything below:

1. **Outputs vary.** The same input can produce different paths, so one passing run proves little.
2. **The path matters.** An agent can reach a correct answer through a flawed route, and that route will fail on the next input.
3. **Failures compound.** Every extra step is another chance to go wrong.

### Shared vocabulary

Anthropic's engineering team [defines a useful set of terms](https://www.anthropic.com/research/demystifying-evals-for-ai-agents) that we will use throughout:

- **Task:** a single test with defined inputs and success criteria.
- **Trial:** one attempt at a task. You run several, because outputs vary.
- **Trace (transcript, trajectory):** the complete record of a trial, including tool calls and reasoning.
- **Outcome:** the final state of the environment. For a flight-booking agent, that is whether the reservation actually exists in the database, not what the agent said at the end.

## Why per-step accuracy is the wrong number

The arithmetic of multi-step work is unforgiving. If each step succeeds independently with probability p, a task of n steps succeeds with probability p^n. At 95% per step and 20 steps, that is 0.95^20, about 36%. An agent that looks excellent in a step-level demo is unreliable on a long task.

![Per-step accuracy compounded across twenty steps](https://media.babylovegrowth.ai/blog-images/organization-30814/1794747714082_Per-step-accuracy-compounded-across-twenty-steps.jpeg)

The same logic applies across repeated interactions. This is the idea behind pass^k: the probability that an agent succeeds on all k trials of a task. An agent that passes a task 75% of the time passes three consecutive trials only about 42% of the time (0.75 cubed). For a user who interacts with your agent several times a week, that gap is the difference between a good demo and a product people abandon.

Academic work formalizes this. The [ReliabilityBench](https://arxiv.org/html/2601.06112) paper proposes a reliability surface across three dimensions: consistency under repeated execution, robustness to perturbations of the task, and fault tolerance when infrastructure fails. Those three axes make a good checklist for your own test plan:

| Dimension       | Question                                          | How to test it                                                |
| --------------- | ------------------------------------------------- | ------------------------------------------------------------- |
| Consistency     | Does it succeed on every one of k trials?         | Run each task k times, report pass^k and the spread           |
| Robustness      | Does a reworded or noisy input change the result? | Paraphrase, add typos, reorder context, then compare outcomes |
| Fault tolerance | Does it recover when a tool or API fails?         | Inject timeouts, empty results and malformed responses        |

DeepSense.ai frames the [goal for AI systems](https://deepsense.ai/blog/ai-testing-vs-software-testing-how-to-build-reliable-production-ai-systems/) as characterizing the distribution of possible answers and deciding what share of it is acceptable. That is the right mindset: set a threshold for each task class, such as "at least 95% of trials must reach the correct end state," and test against it.

## The three levels of agent tests

A complete suite covers three levels. Each catches a different class of bug, and each costs a different amount to run and maintain.

### Step-level tests

A step test constrains the agent to one action from a pre-built state and checks the decision it makes. Given this conversation and these tool results, does it call the right tool with the right arguments? These tests are cheap, fast and stable, so they belong in every pull request. They are where you catch prompt regressions, schema changes in tool definitions and bad argument formatting.

### Trajectory-level tests

A trajectory test runs the agent to completion and scores the path. Useful checks include:

- Were tool calls well-formed, and were arguments grounded in what the agent had actually seen?
- Did the ordering respect constraints, such as verifying identity before touching an account?
- Did the run stay within a hard step budget, with a loop detector to catch repeated calls?

This level exists because a correct answer is not proof of a sound process. One practitioner [puts it plainly](https://nmadiraju.com/posts/agent-evaluation/): a system that reaches the goal by an invalid path, such as bypassing a security control, is not production-worthy. The same point appears in [DeepSense.ai's guidance](https://deepsense.ai/blog/ai-testing-vs-software-testing-how-to-build-reliable-production-ai-systems/) that you must inspect the trace, the sequence of decisions, tool calls and intermediate states.

### Outcome-level tests

An outcome test asserts on the end state of the environment. Did the ticket get created, the row get updated, the refund get issued exactly once? Check the state itself rather than the agent's summary, which can claim success it did not achieve. Multi-turn simulations, where a simulated user converses with the agent, give the highest fidelity but are the hardest to keep stable, so reserve them for your critical journeys.

| Level      | What it checks                  | Cost to run    | Run it                                    |
| ---------- | ------------------------------- | -------------- | ----------------------------------------- |
| Step       | One decision from a fixed state | Low            | Every commit                              |
| Trajectory | Tool calls, order, budget       | Medium         | Every merge, nightly                      |
| Outcome    | Final environment state         | Medium to high | Before release, on a sample in production |

## Build the test set from real tasks

Start small and real. Twenty to fifty tasks drawn from actual user requests are more valuable than hundreds of synthetic ones. For each task, write down:

1. The input and any starting environment state.
2. The success criteria, phrased as a check on the end state wherever possible.
3. Constraints on the path: forbidden tools, required verification steps, a step and token budget.
4. How many trials you will run.

Keep two suites apart. A **capability suite** holds hard tasks the agent does not yet pass reliably and tracks progress. A **regression suite** holds tasks it already passes and must keep passing. Regression suites are how you make sure that [improving one workflow does not break others](https://www.prismetric.com/ai-agent-evaluation/), and every production incident should become a new case in them.

## Trace everything, then grade the traces

You cannot diagnose an agent by looking at its final answer. You need the full trace of each trial: the prompts, the tool calls and results, the intermediate reasoning and the latency and token count of every span. This is the "black box" problem, and tracing is how you open it.

With MLflow, tracing is the foundation. A single line of auto-instrumentation records calls from a supported library, for example:

```python
import mlflow

mlflow.openai.autolog()  # record each model call as a trace
```

Once traces exist, [MLflow AI observability](https://mlflow.org/ai-observability) gives you a view of agentic reasoning across the whole run, and because MLflow builds on OpenTelemetry it fits alongside the frameworks and tooling you already use. Our walkthrough on [seeing inside a multi-agent system](https://mlflow.org/blog/observability-multi-agent-part-1) shows what that looks like when several agents hand work to each other.

Traces also make failure analysis tractable. IMDA points out that analyzing voluminous, unstructured logs of reasoning and tool calls at scale is a distinctive challenge of agent testing and recommends progressive automation. In practice that means scoring traces automatically and reading the failures yourself.

![Trace everything, then grade the traces](https://media.babylovegrowth.ai/blog-images/organization-30814/1794747731854_Trace-everything-then-grade-the-traces.jpeg)

### Use LLM judges for what code cannot check

Some criteria are checkable with code: a tool was called, a database row exists, the step count stayed under budget. Others are not: was the answer grounded in the retrieved documents, did the tone follow policy, was the plan sensible? For those, [LLM-as-a-Judge evaluation](https://mlflow.org/llm-as-a-judge) uses a model to grade outputs and traces against written criteria, so you can run it on every trial rather than on a hand-reviewed sample. Our overview of [LLM-as-a-judge](https://mlflow.org/blog/llm-as-judge) explains how the pattern works, and the [agent and LLM evaluation page](https://mlflow.org/genai/evaluations) shows how it fits into a full evaluation workflow.

A judge is itself a model, so it can be wrong. Treat it like any instrument and calibrate it against human labels: measure its agreement, correct for its bias and report the uncertainty. We cover the method in [LLM judge calibration: measure, correct and report](https://mlflow.org/articles/llm-judge-calibration).

## Diagnose failures before changing anything

When a task fails, resist the urge to edit the prompt. First decide what failed. Anthropic recommends [reading transcripts and grades from many trials](https://www.anthropic.com/research/demystifying-evals-for-ai-agents) to tell whether the agent made a mistake or the grader rejected a valid solution. A quick triage:

1. **Grader bug.** The agent was right and the check was too strict or the judge misread. Fix the grader.
2. **Bad task.** The success criteria are ambiguous. Fix the task.
3. **Tool failure.** The tool returned an error or unexpected schema and the agent handled it poorly. Fix the tool contract or add recovery logic.
4. **Reasoning failure.** The agent picked the wrong tool, looped or lost context. Change the prompt, the tool descriptions or the architecture.

Group failures by category across many trials. If 60% of your failures are one tool returning an empty result, you have a clear priority and a measurable fix. Rerun the same trials after each change and compare pass^k, not a single run.

## From pre-release gates to production monitoring

Reliability testing does not end at release. IMDA states that post-deployment, agents need continuous monitoring and periodic re-testing, in particular as tool and service integrations change. A tool vendor changing a response format can break an agent that passed every test last week.

### A release gate that means something

Define a gate that blocks a release when any of these fail:

- Regression suite pass^k below your threshold for any critical task class.
- A drop in capability-suite results beyond the run-to-run noise you measured.
- Any violation of a hard path constraint, such as an unauthorized tool call.
- Cost or latency per task above budget.

### What to watch in production

Monitor [task success, tool errors, latency, token consumption, retry frequency, user feedback and policy violations](https://www.prismetric.com/ai-agent-evaluation/) to surface new failure modes. Score a sample of live traces with the same judges and checks you use offline, so a drop shows up as a metric instead of a support ticket. When you find a bad trace, add it to the regression suite. That loop, trace, grade, fix, add a regression case, is what makes reliability improve over time rather than oscillate.

## A two-week starting plan

| Days     | Work                                                                                   |
| -------- | -------------------------------------------------------------------------------------- |
| 1 to 2   | Turn on tracing for the agent in development and staging. Collect 30 real tasks.       |
| 3 to 5   | Write outcome checks and path constraints. Add step tests for the riskiest tool calls. |
| 6 to 8   | Add LLM judges for criteria code cannot check. Label 50 traces by hand and compare.    |
| 9 to 10  | Run every task five or more times. Record pass^k and the failure categories.           |
| 11 to 12 | Fix the top failure category. Rerun and compare.                                       |
| 13 to 14 | Wire the regression suite into CI as a release gate. Sample live traffic for scoring.  |

## Promo: tracing, evaluation and gateway in one open-source platform

MLflow is an open-source platform for the lifecycle of GenAI and LLM applications and AI agents. It covers deep tracing of agentic reasoning, automated evaluation with LLM-as-a-Judge, prompt management and versioning, and a cross-provider AI Gateway, with Linux Foundation governance and no enterprise feature paywalls. Start from the [MLflow Agent and LLM engineering page](https://mlflow.org/genai) to see how the pieces fit your stack.

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

## Our perspective

> We built MLflow on a simple belief: an agent you cannot see inside is an agent you cannot make reliable. Most reliability problems we hear about from teams are not exotic. They are a tool returning something unexpected on step six, a run that quietly looped, or a judge nobody ever checked against a human.
>
> That is why we treat tracing, evaluation and governance as one lifecycle instead of three products. The trace is the raw material, the judge turns it into a score, and the score becomes a gate in your release process and a monitor in production. When those live in separate tools, the loop breaks at the handoffs, and the regression case from last week's incident never gets written.
>
> Our advice is to start measuring consistency now, even with a small test set. Report pass^k, read the failing traces yourself, calibrate your judges and keep every checked incident as a regression case. All of this works with the frameworks you already use through OpenTelemetry, and every feature is open source.
>
> _Kevin_

## FAQ

### What is the difference between agent reliability testing and agent evaluation?

Evaluation is the broader practice of scoring how good an agent is: accuracy, helpfulness, safety and cost. Reliability testing is the part focused on consistency: does the agent succeed repeatedly, under perturbation and when tools fail? In practice you use the same traces, judges and datasets for both.

### How many times should I run each test?

Enough to see the spread. Five to ten trials per task is a common starting point for critical tasks, and more if results are noisy. Because the odds compound (75% per trial gives roughly 42% over three), a handful of trials exposes problems that a single run hides.

### Can an LLM judge be trusted to grade agent runs?

It can be a reliable instrument when you validate it. Compare its grades with human labels on a sample, look at where they disagree and adjust the criteria or correct for bias. Re-check after you change the judge model or the rubric.

### Do I need to test in production if I have a strong pre-release suite?

Yes. Tools, data and user behavior change after release, and your suite only covers what you anticipated. Scoring a sample of live traces catches new failure modes, and each one you find becomes a new regression case.

## Sources

- [Starter Kit for Testing Agentic Systems, IMDA Singapore](https://www.imda.gov.sg/assets/603572eb-724a-46cf-ac12-14fe38e1558c.pdf)
- [ReliabilityBench: Evaluating LLM Agent Reliability](https://arxiv.org/html/2601.06112)
- [Demystifying evals for AI agents, Anthropic](https://www.anthropic.com/research/demystifying-evals-for-ai-agents)
- [AI Testing vs Software Testing, DeepSense.ai](https://deepsense.ai/blog/ai-testing-vs-software-testing-how-to-build-reliable-production-ai-systems/)
- [Agent Evaluation: Trajectories, Tool-Calls, Task Completion](https://nmadiraju.com/posts/agent-evaluation/)
- [AI Agent Evaluation: How to Evaluate Agent Performance, Prismetric](https://www.prismetric.com/ai-agent-evaluation/)
- [Reliability Testing Tutorial: Definition, Types, and Best Practices](https://zetcode.com/terms-testing/reliability-testing/)

## Recommended

- [Make LLM Red Teaming Run on Every Release for Engineers](https://mlflow.org/articles/red-teaming-llm)
