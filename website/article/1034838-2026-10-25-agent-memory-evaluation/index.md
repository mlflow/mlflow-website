---
title: "Agent Memory Evaluation: Datasets, Metrics, and a Practical Setup"
description: "How to evaluate agent memory: what LoCoMo, LongMemEval, MemoryAgentBench and others test, which metrics matter, and how to build your own memory evals."
slug: agent-memory-evaluation
tags: [agent evaluation datasets, agent memory evaluation]
date: 2026-10-25
image: https://media.babylovegrowth.ai/blog-images/organization-30814/1793283650960_Developer-testing-an-agent-s-conversational-memory.jpeg
---

![Developer testing an agent’s conversational memory](https://media.babylovegrowth.ai/blog-images/organization-30814/1793283650960_Developer-testing-an-agent-s-conversational-memory.jpeg)

Agent memory evaluation measures whether an agent can store, retrieve, update and forget information across turns and sessions, and then use it to act correctly. In practice you do this in two layers: run public datasets such as LoCoMo and LongMemEval to compare memory designs, then build a small dataset from your own agent's traces to find out whether memory works on your tasks. This guide covers both layers, the metrics worth tracking, and how to wire them into a repeatable loop.

## TL;DR

- Memory evaluation tests four competencies: accurate retrieval, test-time learning, long-range understanding and selective forgetting.
- Public datasets differ in context length, task type and whether the agent acts or only answers. Pick by the failure you fear most.
- Track accuracy, retrieval recall, capacity and efficiency together. A memory system that is accurate but slow can still be unusable.
- Public benchmarks rank memory designs. Your own traces, turned into an evaluation dataset, tell you whether your agent remembers what matters.
- Tracing plus LLM-as-a-Judge scorers lets you separate retrieval failures from reasoning failures.

## Table of Contents

- [What agent memory evaluation actually tests](#what-agent-memory-evaluation-actually-tests)
- [How memory benchmarks work](#how-memory-benchmarks-work)
- [The main agent evaluation datasets compared](#the-main-agent-evaluation-datasets-compared)
- [Task types and metrics that matter](#task-types-and-metrics-that-matter)
- [Why benchmark scores don't predict production behavior](#why-benchmark-scores-dont-predict-production-behavior)
- [How to choose a dataset for your use case](#how-to-choose-a-dataset-for-your-use-case)
- [Building an evaluation dataset from your own agent](#building-an-evaluation-dataset-from-your-own-agent)
- [Scoring memory with traces and LLM-as-a-Judge](#scoring-memory-with-traces-and-llm-as-a-judge)
- [Our perspective: treat memory as part of the agent, not a add-on](#our-perspective-treat-memory-as-part-of-the-agent-not-a-add-on)
- [Run memory evaluations with MLflow](#run-memory-evaluations-with-mlflow)
- [FAQ](#faq)
- [Sources](#sources)

## What agent memory evaluation actually tests

Memory evaluation is not a reasoning or planning test. It assesses [how well agents store, retrieve and update long-term information](https://arxiv.org/html/2507.05257v2) across multi-turn and multi-session interactions. The MemoryAgentBench authors break this into four competencies: accurate retrieval, test-time learning, long-range understanding and selective forgetting.

Those four map neatly onto production failures you have probably already seen:

- **Accurate retrieval:** the user told the agent their shipping region three sessions ago, and the agent asks again.
- **Test-time learning:** the user corrects a preference once, and the agent repeats the old behavior.
- **Long-range understanding:** the answer depends on combining facts scattered across a long history.
- **Selective forgetting:** the user changes their mind, and the agent keeps citing the stale value. MemoryAgentBench treats [revising, overwriting or removing stored information when evidence contradicts it](https://arxiv.org/html/2507.05257v2) as a core skill, not an edge case.

If you only test recall of a single fact, you cover the first competency and miss the other three.

## How memory benchmarks work

Most benchmarks follow the same shape. They [feed the agent text chunks incrementally, let it update its memory, and then ask related questions](https://arxiv.org/html/2507.05257v2) that require retrieval, learning, long-range understanding or forgetting. The score is usually downstream answer correctness.

A newer line of work argues that this static design leaves something out. The AMemGym authors note that reusable memory is [intrinsically linked to the interaction loop](https://arxiv.org/html/2603.01966v1): actions produce feedback, feedback updates memory, and memory guides the next action. The MemoryArena authors go further and suggest memory should be [evaluated as an inseparable part of agentic behavior](https://arxiv.org/html/2602.16313v2), with memorization and action tightly coupled across sessions.

That distinction matters for you. If your agent only answers questions about past chats, a question-answering benchmark is a reasonable proxy. If it books, edits, deploys or otherwise acts, you want an interdependent, multi-session test.

![How memory benchmarks work](https://media.babylovegrowth.ai/blog-images/organization-30814/1793283708863_How-memory-benchmarks-work.jpeg)

## The main agent evaluation datasets compared

The table below summarizes the datasets most often used for agent memory, based on what each paper reports.

| Dataset          | What it assesses                                                              | Notable detail                                                                                                                                         |
| ---------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| LoCoMo           | Long-term conversational memory                                               | [About 26k tokens per context](https://d6108366.hf-mirror.com/papers/2601.14287)                                                                       |
| LongMemEval      | Long-term conversational memory at larger scale                               | [Average length of about 115k tokens](https://d6108366.hf-mirror.com/papers/2601.14287)                                                                |
| MemoryAgentBench | Retrieval, test-time learning, long-range understanding, selective forgetting | [Incremental multi-turn format](https://arxiv.org/html/2507.05257v2)                                                                                   |
| MemBench         | Factual and reflective memory in participation and observation scenarios      | [Information extraction, cross-session reasoning, knowledge updating, temporal reasoning, reflective summarization](https://arxiv.org/html/2506.21605) |
| MemoryArena      | Interdependent multi-session tasks                                            | [Functional reuse of information rather than simple recall, scored with a Task Progress Score](https://arxiv.org/html/2602.16313v2)                    |
| AMA-Bench        | Long-horizon memory in agentic applications                                   | [Real-world machine-generated traces plus synthetic tasks](https://arxiv.org/html/2602.22769v4)                                                        |
| AMemGym          | Interactive, on-policy memory evaluation                                      | [Agents tested on their own conversational choices](https://arxiv.org/html/2603.01966v1)                                                               |

A few reading notes:

- **LoCoMo versus LongMemEval** is mostly a question of scale. Both target conversational memory, but the difference in context length (roughly 26k versus 115k tokens) means a design that works on the first can fall over on the second.
- **MemBench** is useful when you care about more than recall. Its authors stress that memory is [not monolithic and spans factual and reflective levels](https://arxiv.org/html/2506.21605), which is the right framing if your agent summarizes or draws conclusions from past interactions.
- **AMA-Bench** is the closest to what an agent engineer sees day to day, because it includes machine-generated traces, not just human chat.

## Task types and metrics that matter

Question types vary by dataset, and the labels you will meet most often are single-hop, multi-hop, temporal and open-domain questions. Single-hop needs one stored fact. Multi-hop needs several combined. Temporal asks about order or recency. Open-domain mixes stored memory with general knowledge. Reporting scores per category, not a single average, shows where a memory design breaks.

For metrics, MemBench reports four that work well as a default set:

1. **Memory accuracy:** is the final answer right?
2. **Retrieval recall:** did the memory layer surface the relevant item at all?
3. **Memory capacity:** [the threshold where accuracy drops](https://arxiv.org/html/2506.21605) as stored history grows.
4. **Memory efficiency:** processing time, since [high-latency memory systems can be unacceptable in practice](https://arxiv.org/html/2506.21605).

The cost angle deserves attention too. The Chain-of-Memory researchers argue that [complex memory construction can incur high costs for marginal gains](https://huggingface.co/papers/2601.14287) and advocate lightweight construction with better use of retrieved context. Whatever you think of that position, it is a reminder to log tokens and latency next to accuracy, so you can see what each point of accuracy costs.

## Why benchmark scores don't predict production behavior

Public datasets are a starting line. Three gaps show up when you move to production.

**Off-policy bias.** Most static benchmarks replay a fixed conversation. The AMemGym team warns that this kind of off-policy evaluation introduces [reuse bias that can mislead memory-augmented agent development](https://arxiv.org/html/2603.01966v1), because it ignores how an agent's own behavior shapes what it will need to remember later. Your agent asks different clarifying questions than the benchmark's scripted one did.

**Distribution mismatch.** Your users' sessions contain tool outputs, partial instructions, domain jargon and contradictions that no public set reproduces.

**Scope mismatch.** A benchmark answer is graded in isolation. In production, a wrong memory shows up three steps later as a wrong tool call.

The fix is not to abandon public datasets. Use them to shortlist memory designs, then validate on data that looks like yours.

## How to choose a dataset for your use case

Start from the failure you most need to prevent, then pick the dataset that stresses it.

| If your main risk is...                   | Start with                              | Then add                                        |
| ----------------------------------------- | --------------------------------------- | ----------------------------------------------- |
| Forgetting facts from long chat histories | LoCoMo, then LongMemEval                | Per-category scoring for temporal and multi-hop |
| Stale or contradicted information         | MemoryAgentBench (selective forgetting) | Your own correction scenarios                   |
| Agents that act on memory across sessions | MemoryArena                             | On-policy interactive tests                     |
| Long tool-heavy trajectories              | AMA-Bench                               | Traces from your own agent                      |
| Summaries and inferred preferences        | MemBench (reflective memory)            | Human review of a sample                        |

A practical rule: run one conversational benchmark for a baseline, one benchmark that matches your riskiest competency, and then stop. Past that, effort is better spent on your own data.

## Building an evaluation dataset from your own agent

This is where most of the value is. The recipe has five steps.

1. **Capture traces.** Instrument the agent so every session records inputs, retrieved memories, tool calls and final output. You cannot grade retrieval if you cannot see what was retrieved.
2. **Mine memory-dependent moments.** Filter traces for turns where the correct answer depends on something said earlier: a stated preference, a correction, an earlier tool result.
3. **Write expectations.** For each moment, record the fact that should be recalled and, for corrections, the stale value that must not appear.
4. **Cover the four competencies.** Aim for a mix: retrieval, a learn-once-use-later case, a multi-fact synthesis case and a changed-mind case.
5. **Version the dataset.** Treat it like code. When a production failure appears, add it as a new row so it can never silently regress.

A minimal row might look like this:

```python
row = {
    "inputs": {
        "history": [
            {"session": 1, "user": "I'm vegetarian."},
            {"session": 3, "user": "Actually, I eat fish now."},
        ],
        "question": "Suggest a dinner for me.",
    },
    "expectations": {
        "must_reflect": "pescatarian diet",
        "must_not_assert": "strictly vegetarian",
    },
}
```

The `must_not_assert` field is the one teams skip. It is what turns a forgetting test from a vibe into a pass/fail check.

## Scoring memory with traces and LLM-as-a-Judge

Exact-match scoring fails on memory, because correct answers are phrased many ways. An LLM judge with a narrow rubric handles this well, provided you keep the rubric specific: one judge for "reflects the current value", another for "does not assert the stale value", rather than one vague "memory quality" score.

In MLflow, the shape of an evaluation run is a dataset, a prediction function and a list of scorers. A sketch:

```python
import mlflow
from mlflow.genai.scorers import scorer

@scorer
def reflects_current_value(inputs, outputs, expectations):
    return expectations["must_reflect"].lower() in outputs.lower()

@scorer
def avoids_stale_value(inputs, outputs, expectations):
    return expectations["must_not_assert"].lower() not in outputs.lower()

@mlflow.trace
def run_agent(history, question):
    ...  # your agent, with memory read and write steps

results = mlflow.genai.evaluate(
    data=dataset,
    predict_fn=run_agent,
    scorers=[reflects_current_value, avoids_stale_value],
)
```

Swap the string checks for an LLM-based judge when phrasing varies. The key point is that the agent runs under tracing, so each failing row links to the trace behind it. Check the current MLflow documentation for exact scorer signatures in your installed version.

### Separate retrieval failures from reasoning failures

When a memory test fails, the cause is one of three things: the memory was never written, it was written but not retrieved, or it was retrieved and ignored. Without traces you guess. With them, the retrieval span shows what came back, and you can compute retrieval recall directly: did the relevant item appear in the retrieved set? If yes and the answer is still wrong, the problem is in the prompt or the model, not the memory store. If no, fix indexing or the query rewrite.

![Three causes of agent memory test failures](https://media.babylovegrowth.ai/blog-images/organization-30814/1793283693035_Three-causes-of-agent-memory-test-failures.jpeg)

### Report capacity and latency

Run the same dataset at increasing history lengths and plot accuracy against stored volume. The point where accuracy drops is your capacity threshold. Record end-to-end latency per turn alongside it, so a memory upgrade that adds a second per request is visible before it ships.

## Our perspective: treat memory as part of the agent, not a add-on

We think the most useful shift in this field is the move from scoring memory in isolation to scoring it inside the agent loop. The research points the same way: memory and action are coupled, so a memory score that ignores the actions it drives is incomplete.

That is why we built MLflow around the whole lifecycle instead of one slice. Tracing records the reasoning, the retrieval and the tool calls in one place. Evaluation with LLM-as-a-Judge scorers turns those traces into repeatable numbers. Prompt management and versioning let you change how memory is injected and compare the result against the last version. The AI Gateway keeps provider changes from muddying the comparison. Because it is all open source under Linux Foundation governance, with no enterprise feature paywalls, any team can run this loop on its own agents, with the OpenTelemetry-based tracing working across the frameworks they already use.

Our advice: pick one public benchmark, build fifty rows from your own traces, and run both on every memory change. A small dataset you trust beats a large one you never run.

> _— Kevin_

## Run memory evaluations with MLflow

MLflow gives you tracing, evaluation, prompt versioning and a cross-provider gateway in one open-source platform, so the memory evaluation loop above does not need a patchwork of tools. Start with the free open-source release at [MLflow](https://mlflow.org), and bring your own agent framework.

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

## FAQ

### What is the difference between LoCoMo and LongMemEval?

Both assess long-term conversational memory. They differ mainly in scale: [LoCoMo contexts run about 26k tokens, while LongMemEval averages about 115k](https://d6108366.hf-mirror.com/papers/2601.14287). A memory design that holds up on LoCoMo may degrade on LongMemEval, so running both shows how it scales.

### Which metrics should I track for agent memory?

Track accuracy, retrieval recall, capacity and efficiency. MemBench defines [capacity as the point where accuracy drops and efficiency as processing time](https://arxiv.org/html/2506.21605). Add token cost if you pay per call.

### Can I rely on public benchmarks alone?

No. Static benchmarks can introduce [reuse bias](https://arxiv.org/html/2603.01966v1) because they ignore how your agent's own behavior shapes later memory needs. Use them to shortlist designs, then confirm on a dataset built from your own traces.

### How big should my own memory evaluation dataset be?

Start small: a few dozen rows covering retrieval, learning from a correction, multi-fact synthesis and changed-mind cases. Add a row for every production memory failure you find. Coverage of the four competencies matters more than raw size.

## Sources

- [Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions (MemoryAgentBench)](https://arxiv.org/html/2507.05257v2)
- [AMemGym: Interactive Memory Benchmarking for Assistants](https://arxiv.org/html/2603.01966v1)
- [Benchmarking Agent Memory in Interdependent Multi-Session Settings (MemoryArena)](https://arxiv.org/html/2602.16313v2)
- [MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents](https://arxiv.org/html/2506.21605)
- [AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications](https://arxiv.org/html/2602.22769v4)
- [Chain-of-Memory: Lightweight Memory Construction (paper page)](https://huggingface.co/papers/2601.14287)
- [LoCoMo and LongMemEval context lengths (paper page mirror)](https://d6108366.hf-mirror.com/papers/2601.14287)
