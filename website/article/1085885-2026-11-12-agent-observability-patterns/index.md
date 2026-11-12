---
title: "Agent Observability Patterns for Production AI Agents"
description: "Seven agent observability patterns for production teams: span trees, four telemetry streams, trace-linked evaluation, alerts and safe sampling, with MLflow examples."
slug: agent-observability-patterns
tags: [agent observability patterns]
date: 2026-11-12
image: https://media.babylovegrowth.ai/blog-images/organization-30814/1794291177160_Engineer-follows-an-agent-trace-from-alert-to-span.jpeg
---

![Engineer follows an agent trace from alert to span](https://media.babylovegrowth.ai/blog-images/organization-30814/1794291177160_Engineer-follows-an-agent-trace-from-alert-to-span.jpeg)

The core agent observability patterns are few. Trace every decision as a span tree, collect four streams of telemetry (traces, metrics, logs, evaluations), attach quality scores to traces, alert on behavior drift instead of only uptime, and sample and redact with care. Agents decide what to do between one call and the next, so a green dashboard can still hide a wrong answer. Below we walk through each pattern, show what to capture, and show how we approach it in MLflow.

## TL;DR

- Trace the whole run as a hierarchical span tree: the agent invocation, each model call, each tool call, each retrieval and each memory read or write.
- Collect four complementary streams: traces, metrics, logs and evaluations. Any one alone leaves a blind spot.
- Connect traces to evaluation, so every production run can carry a quality score and not only a latency number.
- Alert on behavior (cost per run, retry rate, tool-selection drift, eval-score drop), not only on errors.
- Decide sampling, redaction and retention before launch, because traces contain prompts, documents and tool outputs.

## Table of Contents

- [Why agents need different observability](#why-agents-need-different-observability)
- [Pattern 1: Model the run as a span tree](#pattern-1-model-the-run-as-a-span-tree)
- [Pattern 2: Use OpenTelemetry conventions so traces cross boundaries](#pattern-2-use-opentelemetry-conventions-so-traces-cross-boundaries)
- [Pattern 3: Collect four streams, not one](#pattern-3-collect-four-streams-not-one)
- [Pattern 4: Capture the full set of artifacts](#pattern-4-capture-the-full-set-of-artifacts)
- [Pattern 5: Attach evaluation to traces](#pattern-5-attach-evaluation-to-traces)
- [Pattern 6: Alert on behavior, not only on errors](#pattern-6-alert-on-behavior-not-only-on-errors)
- [Pattern 7: Plan sampling, redaction and retention up front](#pattern-7-plan-sampling-redaction-and-retention-up-front)
- [Perspective: why we think open tracing wins](#perspective-why-we-think-open-tracing-wins)
- [Put the patterns to work with MLflow](#put-the-patterns-to-work-with-mlflow)
- [A rollout order that works](#a-rollout-order-that-works)
- [FAQ](#faq)
- [Sources](#sources)

## Why agents need different observability

Traditional application monitoring answers a narrow question: is the system healthy? Latency, error rate and throughput tell you that. For an agent, they leave out most of what matters. One write-up on [agent observability](https://www.mongodb.com/company/blog/technical/agent-observability-monitoring-decisions-not-requests) puts it well: the unit of behavior for an agent is a decision between one call and the next, not a single request and response. The same source separates three jobs. Monitoring tells you whether the system is healthy. Observability reconstructs what the agent did and why. Evaluation judges whether what it did was good.

This is also why the final output is a weak signal on its own. Gartner, as relayed by [TechTarget](https://www.techtarget.com/news/Key-enterprise-strategies-for-AI-agent-observability/834029/), explains that agent behavior varies with the model, the prompt, the available tools and intermediate results. The end result gives only a partial indication of whether the agent behaved correctly. You have to look at the path it took.

A concrete example: a support agent returns a polite, plausible refund answer. The HTTP call succeeded in 1.8 seconds. But the trace shows it skipped the policy lookup tool, guessed from memory, and retried the order API three times. Nothing in a standard dashboard flags that. A trace does.

If you want the fundamentals first, our [developer guide to agent observability](https://mlflow.org/articles/what-is-agent-observability-a-2026-developer-guide) covers the definitions. This article is about the patterns you apply once you are running agents for real.

![Why agents need different observability](https://media.babylovegrowth.ai/blog-images/organization-30814/1794291215674_Why-agents-need-different-observability.jpeg)

## Pattern 1: Model the run as a span tree

The primary artifact in agent observability is the trace, a hierarchical span tree for the full lifecycle of one agent task. According to the [MongoDB engineering write-up](https://www.mongodb.com/company/blog/technical/agent-observability-monitoring-decisions-not-requests), that tree includes the parent agent invocation, each model call, each tool execution and each memory read or write. Engineers use it to understand behavior that only shows up at runtime.

A practical structure, following the [recommendation from AI Stack Current](https://aistackcurrent.com/ai-engineering/agent-observability-and-tracing/), starts at the application boundary and nests downward:

1. A workflow or task span for the whole request.
2. Agent or orchestration spans for each planning or routing step.
3. Model spans for each LLM call.
4. Retrieval and memory spans.
5. Tool spans.
6. Validation and approval spans.
7. External-service spans for anything the tools call.

The [Secure AI Agent Architecture book](https://agent-axiom.github.io/agent-arch/en/book/part-v/chapter-11/) adds two spans that teams often forget: a span for a policy decision when it changes behavior, and a span for a human approval wait when one exists. Those are the spans that explain why a run took four minutes instead of four seconds.

In MLflow, you can get this tree from an existing agent with minimal code. Decorate your own functions and let framework integrations handle the rest:

```python
import mlflow

mlflow.set_experiment("support-agent")

@mlflow.trace(span_type="TOOL")
def lookup_policy(topic: str) -> str:
    ...

@mlflow.trace(span_type="AGENT")
def answer(question: str) -> str:
    policy = lookup_policy(question)
    ...
```

Each decorated call becomes a span, nested under its caller, with inputs, outputs and timing recorded. Because MLflow tracing builds on OpenTelemetry, the same data can fit into the telemetry pipeline you already run.

## Pattern 2: Use OpenTelemetry conventions so traces cross boundaries

Agents rarely live in one process. A planner calls a retrieval service, which calls a vector store, which calls an embedding model. If each layer logs in its own format, you cannot follow a request across them.

The OpenTelemetry GenAI semantic conventions address this by modeling an agent run as a span tree, with named operations for [agent creation and invocation, workflow invocation, tool execution, retrieval, planning and memory operations](https://www.covasant.com/glossary-of-agentic-ai/ai-agent-observability). Following shared names means a trace that starts in a Python agent and continues in a Go service still reads as one run.

Three habits make this work in practice:

- Propagate trace context through every hop, including queues and background workers.
- Name spans by operation (for example, a tool call named after the tool), not by a generic label such as "step".
- Keep custom attributes consistent across teams: model name, model version, prompt version, tenant or environment.

Our positioning here is simple: we built MLflow around OpenTelemetry compatibility so that agent traces are not locked into one format, and so that they work with the AI and ML frameworks your team already uses.

## Pattern 3: Collect four streams, not one

Traces are necessary and not sufficient. [Provectus describes four complementary streams](https://provectus.com/blog/agent-telemetry-building-traces-metrics-ci-cd-for-claude-agents/) for agent telemetry:

| Stream      | What it holds                                                              | What it answers                         |
| ----------- | -------------------------------------------------------------------------- | --------------------------------------- |
| Traces      | The execution path as a graph of steps                                     | What did the agent do, in what order?   |
| Metrics     | Latency percentiles, token consumption, cost per task, error rates         | Is it fast, cheap and stable?           |
| Logs        | Immutable, timestamped records                                             | What happened, for forensics and audit? |
| Evaluations | Quality scores from LLM-as-a-judge, rule-based validators or user feedback | Was the outcome good?                   |

The [AgentOps pattern](https://aipatternbook.com/agentops) frames the same idea at the agent layer with four streams of its own: trajectory (the ordered thoughts, tool calls and results), cost (tokens in and out, cached tokens, tool invocations, model version), quality (periodic sampled evaluation of live runs) and autonomy compliance (whether the agent stayed inside its approval policy).

The overlap is the lesson. Whatever vocabulary you use, you need the path, the numbers, the record and the judgment. A team with traces and metrics but no evaluation can tell you an agent was fast and cheap and has no idea whether it was right.

## Pattern 4: Capture the full set of artifacts

A common mistake is tracing only the LLM call. A [survey on evidence tracing](https://arxiv.org/html/2606.04990v4) defines agent traces more broadly, as the recorded artifacts generated or consumed during a run: instructions, retrieval queries, retrieved documents, tool calls, tool outputs, memory operations, observations, intermediate claims, inter-agent messages, actions and final responses.

That list is a useful checklist when you review your own instrumentation. Ask of each item: if this went wrong, could I see it in a trace?

- **Retrieval queries and retrieved documents.** Most "hallucinations" in retrieval agents are retrieval failures. You can only see that if the documents are in the trace.
- **Tool outputs.** A tool that returns an empty list and a tool that returns an error look the same if you log only the final answer.
- **Memory operations.** Stale or wrong memory reads explain behavior that seems random across sessions.
- **Inter-agent messages.** In multi-agent systems, the handoff is often where context gets lost.

The same survey makes a distinction worth keeping in mind. Traceability makes artifacts inspectable, while provenance goes further by modeling the typed relations among them. For high-stakes agents, connecting the semantic support for an answer (which document backed which claim) with the procedural execution (which step produced it) is the stronger design.

## Pattern 5: Attach evaluation to traces

This is the pattern that separates a debugging tool from a quality system. A trace tells you what happened. An evaluation tells you whether it was good. Joined together, you can ask questions like: which tool sequences correlate with low scores, or did quality drop after we changed the prompt?

A workable loop looks like this:

1. Log every production run as a trace, with prompt and model versions as attributes.
2. Sample a slice of traces for automated evaluation. LLM-as-a-judge scorers are well suited to open-ended criteria such as relevance, groundedness and policy adherence. Rule-based checks cover the deterministic ones, like "did the agent call the refund tool without an order ID?"
3. Store the scores on the trace, so a low score links straight to the run that earned it.
4. Review the lowest-scoring and most expensive traces weekly, and turn real failures into regression test cases.
5. Re-run those cases against every new prompt or model version before release.

The [MongoDB write-up](https://www.mongodb.com/company/blog/technical/agent-observability-monitoring-decisions-not-requests) is explicit that evaluation is its own job, separate from reconstructing behavior. We agree, and it is the reason MLflow treats tracing, automated LLM-based evaluation and prompt management as one lifecycle and not three tools. A prompt change, its traces and its eval scores stay connected.

Two cautions for judge-based scoring. Keep the judging criteria narrow and written down, since vague rubrics give noisy scores. And spot-check judge output against human review on a small set regularly, so you know the scorer still reflects what your users value.

## Pattern 6: Alert on behavior, not only on errors

An agent can fail without throwing an exception. The useful alerts watch for drift. Wolf McNally, writing on [AgentOps](https://aipatternbook.com/agentops), argues that alerts should fire on cost per run, retry rate, tool-selection drift, eval-score drop and policy violations. The goal is to notice a regression in behavior before users do, not after support tickets arrive.

Here is how we would translate that into a starter alert set:

| Signal               | Why it matters                                          | Typical first response                                           |
| -------------------- | ------------------------------------------------------- | ---------------------------------------------------------------- |
| Cost per run rising  | Loops, oversized context or a model swap                | Inspect the longest traces for repeated tool calls               |
| Retry rate rising    | A flaky tool or a malformed tool schema                 | Check tool spans for error patterns                              |
| Tool-selection drift | The agent now picks different tools for the same intent | Compare tool-call distributions before and after the last change |
| Eval-score drop      | Quality regression                                      | Diff prompt, model and retrieval versions between score windows  |
| Policy violation     | The agent stepped outside its approval rules            | Review the policy and approval spans for that run                |

Set thresholds from a baseline week of normal traffic and not from guesses. Alert on percentiles (p95 latency, p95 cost per task) so a few long runs do not hide behind an average.

For debugging, work from the alert to a single trace, then outward. Open the trace, find the first span whose output looks wrong, and read upstream from there. In most incidents the visible failure is several spans downstream of the cause.

You can browse more of our material on [monitoring agent performance](https://mlflow.org/articles/tags/monitoring-agent-performance) and [agent observability metrics](https://mlflow.org/articles/tags/agent-observability-metrics).

## Pattern 7: Plan sampling, redaction and retention up front

Traces are rich because they hold prompts, retrieved documents and tool outputs. That also makes them sensitive. Decide these things before launch, not after your first audit question:

- **What gets redacted.** Strip or mask personal data, credentials and customer content at the point of capture, not in a later cleanup job. Test the redaction against real examples.
- **What gets sampled.** Keep full traces for errors, policy violations, low eval scores and expensive runs. Sample the healthy majority. Sampling decisions that depend on how a run ended need the whole trace first, so plan where that decision is made in your collector pipeline.
- **Who can read traces.** Treat trace storage like production data. Apply the same access controls and audit logging.
- **How long you keep them.** Match retention to your internal policy and any regulation that covers the data the agent touches. Ask your security and legal teams for the specific requirement for your industry, since it differs by sector and state.

An audit trail is also a benefit here. Observability can record what data was accessed, which tools were used and what actions were taken under which permissions, which is what security reviewers ask for. Build the redaction rules first, and the audit story gets easier.

## Perspective: why we think open tracing wins

> We have watched teams ship agents with excellent infrastructure dashboards and almost no understanding of agent behavior. The dashboard said the service was up. The customer said the answer was wrong. Both were true.
>
> Our view is that observability for agents has to be a lifecycle, not a viewer. Seeing a trace is the start. You then need to score it, tie it to the prompt version that produced it, route model calls through a governed gateway, and feed the failures back into your tests. That is why MLflow covers tracing, automated evaluation, prompt management and an AI Gateway in one open-source platform.
>
> We also think this layer should not carry a paywall or a proprietary format. MLflow is fully open source, governed under the Linux Foundation, and builds on OpenTelemetry, so your traces stay portable across the frameworks you choose today and the ones you adopt next year. Start with Pattern 1 and Pattern 5. A span tree plus a quality score on every sampled run answers most of the questions your team will ask in its first incident.
>
> _Kevin_

## Put the patterns to work with MLflow

If you are ready to apply this, [MLflow's AI observability](https://mlflow.org/ai-observability) page shows how tracing, evaluation and the AI Gateway fit together for LLM applications and agents. The core platform is free and open source, so you can instrument one agent this week and extend from there.

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

Teams that need enterprise support or a managed setup can start from the [MLflow home page](https://mlflow.org).

## A rollout order that works

You do not need all seven patterns on day one. A sequence we find sensible:

1. **Week one:** trace one agent end to end, with model, tool and retrieval spans, and add prompt and model version attributes.
2. **Week two:** add cost and latency metrics at p50 and p95, and set a baseline.
3. **Week three:** turn on sampled evaluation with two or three judges tied to your top failure modes.
4. **Week four:** add behavioral alerts and a redaction and retention policy, then review your first low-scoring traces as a team.

More guidance is collected under [best practices for agent observability](https://mlflow.org/articles/tags/best-practices-for-agent-observability) and [improving agent observability](https://mlflow.org/articles/tags/improving-agent-observability).

## FAQ

### What is the difference between agent observability and LLM observability?

LLM observability centers on single model calls: prompt, completion, tokens and latency. Agent observability covers the whole multi-step run, including planning, tool calls, retrieval, memory and the decisions between calls. It reconstructs why the agent acted as it did, which a single request log cannot do.

### Which spans should I add first?

Start with a top-level workflow span, one span per model call and one span per tool call. Add retrieval and memory spans next, then validation, policy and human-approval spans. That order covers most debugging needs with the least instrumentation work.

### Do I need OpenTelemetry to observe agents?

Not strictly, but it helps. Shared conventions let a trace follow a request across services and languages, and keep your data portable. MLflow tracing builds on OpenTelemetry for that reason.

### How do I know if my agent is getting worse in production?

Track evaluation scores on sampled live traces alongside cost per run, retry rate and tool-selection patterns. A drop in score or a shift in tool choice after a prompt or model change is the earliest sign. Compare against a baseline window, and open the lowest-scoring traces to find the cause.

### Should I store every trace?

Usually no. Keep full traces for errors, low scores, policy violations and unusually expensive runs, and sample the rest. Redact sensitive data at capture, and set retention with your security and legal teams.

## Sources

- [Agent Observability: Monitoring Decisions, Not Requests](https://www.mongodb.com/company/blog/technical/agent-observability-monitoring-decisions-not-requests)
- [Key enterprise strategies for AI agent observability](https://www.techtarget.com/news/Key-enterprise-strategies-for-AI-agent-observability/834029/)
- [From Agent Traces to Trust: A Survey of Evidence Tracing](https://arxiv.org/html/2606.04990v4)
- [Agent Telemetry: Building Traces, Metrics & CI/CD for Claude Agents](https://provectus.com/blog/agent-telemetry-building-traces-metrics-ci-cd-for-claude-agents/)
- [AgentOps, Encyclopedia of Agentic Coding Patterns](https://aipatternbook.com/agentops)
- [What Is AI Agent Observability? Traces, Sampling, and Limits](https://www.covasant.com/glossary-of-agentic-ai/ai-agent-observability)
- [Chapter 11. Traces, Spans, and Structured Events](https://agent-axiom.github.io/agent-arch/en/book/part-v/chapter-11/)
- [Agent observability and tracing, AI Stack Current](https://aistackcurrent.com/ai-engineering/agent-observability-and-tracing/)

## Recommended

- [AI observability for production: Seeing Inside Your Multi-Agent System with MLflow](https://mlflow.org/blog/observability-multi-agent-part-1)
- [AI Observability for LLMs & Agents](https://mlflow.org/ai-observability)
