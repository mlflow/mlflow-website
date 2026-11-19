---
title: "Agent Tool Call Logging: What to Record and How"
description: "What an agent tool call log should record, how to tie calls into traces, redact sensitive data and retain logs, with a working MLflow tracing setup."
slug: agent-tool-call-logging
tags: [agent tool call logging]
date: 2026-11-19
image: https://media.babylovegrowth.ai/blog-images/organization-30814/1794956517146_Engineer-investigating-a-failed-agent-tool-trace.jpeg
---

![Engineer investigating a failed agent tool trace](https://media.babylovegrowth.ai/blog-images/organization-30814/1794956517146_Engineer-investigating-a-failed-agent-tool-trace.jpeg)

Agent tool call logging means capturing, for every call an agent makes to a tool, the tool name, the exact arguments the model generated, the raw result, the status, the timing, and the trace and session it belongs to. Plain log lines can't do that job, because they don't connect a tool call to the model turn that chose it. Traces can. With [MLflow Tracing](https://mlflow.org/genai/observability), each tool call becomes a typed span inside the trace of the run, so you can search, filter and replay it.

This guide covers the fields to log, how to structure them, how to handle sensitive data and retention, and a working setup you can adapt.

## TL;DR

- Log tool calls as spans inside a trace, not as free-floating log lines. A span ties the call to the run, the model turn and any retries.
- Minimum record per call: call ID, trace and session ID, tool name, exact arguments, raw result or error, status, timestamp and duration.
- Redact sensitive data at write time, before it reaches storage, and keep result bodies on a shorter retention than metadata.
- A tool call log shows what happened. It does not prove why the model chose the tool, so keep the surrounding context too.
- MLflow is open source, OpenTelemetry-compatible and has a TOOL span type, so tool calls are searchable from the UI or the SDK.

## Table of Contents

- [Why plain logs fall short for tool calls](#why-plain-logs-fall-short-for-tool-calls)
- [What each tool call record should contain](#what-each-tool-call-record-should-contain)
- [Capture reasoning context, but do not mistake it for proof](#capture-reasoning-context-but-do-not-mistake-it-for-proof)
- [Connect tool calls across runs, retries and agents](#connect-tool-calls-across-runs-retries-and-agents)
- [Instrument tool calls with MLflow](#instrument-tool-calls-with-mlflow)
- [Protect sensitive data in tool call logs](#protect-sensitive-data-in-tool-call-logs)
- [Use the logs for debugging, evaluation and usage](#use-the-logs-for-debugging-evaluation-and-usage)
- [A rollout checklist](#a-rollout-checklist)
- [Our perspective](#our-perspective)
- [Start tracing tool calls with MLflow](#start-tracing-tool-calls-with-mlflow)
- [FAQ](#faq)
- [Sources](#sources)

## Why plain logs fall short for tool calls

A classic log line is a standalone, timestamped event. In an agent, one answer comes from a retrieval step, one or more model turns and several tool calls, and entries from concurrent runs interleave in the same stream. [One production write-up](https://quantaintelligence.ai/blog/structuring-logs-and-traces-for-ai-agents-what-to-record-at-each-step) makes the point well: the questions you actually ask, such as why the agent picked a tool or which document it cited, are relational rather than point-in-time. Each step therefore needs a span to join the run, plus structured key-value fields to be queryable.

The common telemetry model has three tiers. The [AWS Well-Architected agentic AI lens](https://docs.aws.amazon.com/wellarchitected/latest/agentic-ai-lens/agentperf01-bp02.html) describes a session (a complete conversation), a trace (one request-response cycle) and spans (discrete operations such as a reasoning iteration, an LLM call, a tool invocation, a memory lookup or a handoff between agents). A tool call is one span type among several. That matters for the logging design: you log the tool call as a span in a trace, and you give the trace enough context that the call is interpretable alone.

## What each tool call record should contain

Start with the minimum that lets you reconstruct what happened. [Tuurt's production guidance](https://tuurt.com/en/blog/logging-tool-calling-llm-agents-production/) lists a call identifier, a session or trace identifier, the tool name, the exact model-generated arguments, the raw API result, the final status and a timestamp. For calls that change state, it adds a snapshot of the prior state.

The standards bodies converge on the same core. [Microsoft's guidance for observability in AI systems](https://learn.microsoft.com/en-us/security/zero-trust/sfi/observability-ai-systems) says production systems should log request identity context, timestamps, conversation and run identifiers, user inputs, system responses, retrieval provenance and agent and tool invocations, including tool name, arguments, permissions and outputs, using OpenTelemetry GenAI semantic conventions. The [OWASP Agent Observability Standard](https://aos.owasp.org/spec/trace/extend_opentelemetry/) maps this to spans: a request span captures the tool ID, inputs and an execution ID, and a result span records the execution ID and the result, including content and error status.

Here is a practical field set, grouped by what it answers.

| Question                      | Fields                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------ |
| Which call is this?           | call or execution ID, trace ID, span ID, parent span, session ID               |
| What was asked?               | tool name (stable, low cardinality), exact arguments as generated by the model |
| What came back?               | raw result or a summary and hash, error type, status                           |
| How did it go?                | start and end timestamps or duration, retry count                              |
| Who and under what authority? | user, agent and version, permissions or scopes, approval state                 |
| What could it change?         | whether the call mutates state, whether it is reversible, prior-state snapshot |

The last row comes from a [risk-focused schema](https://singleaxis.ai/briefings/what-to-log-around-a-tool-call), which argues that fields for side effects and reversibility are what turn a log into a risk record. If an agent can refund an order or delete a file, you want that flagged on the record, not inferred later.

Two cautions on the fields:

1. **Log the real arguments, not a paraphrase.** The arguments the model generated are the thing that hit your API. If validation rewrites them, log both the original and the validated form.
2. **Keep the tool name stable.** A name like `search_orders` is filterable. A name with an embedded customer ID is not.

## Capture reasoning context, but do not mistake it for proof

A tool call is the visible result of a decision you can't see directly. The same [risk schema](https://singleaxis.ai/briefings/what-to-log-around-a-tool-call) frames the boundary as the decision rather than the call: logging only the call keeps the last step and drops what the agent had in context when it chose. Tuurt makes a complementary point: reasoning attached to a tool choice should be kept, but never as a substitute for the real arguments and result, since reasoning explains intent and not what actually happened against the API.

In practice, use two layers. Operational spans localize where time went and which calls failed. A second layer records the context the model saw and the action it chose, which [MongoDB's observability team](https://www.mongodb.com/company/blog/technical/agent-observability-monitoring-decisions-not-requests) calls a reasoning trace. When you investigate an incident, treat the arguments and results as the record of fact and the reasoning text as a hint about intent.

Another prototype habit to drop: saving only the final message the agent showed the user. Tuurt calls this the first thing to fail in production, because it leaves no trail of intermediate steps.

## Connect tool calls across runs, retries and agents

A tool call log is only useful if you can follow the run around it. Three rules help.

**Propagate context across boundaries.** The AWS agentic AI lens says session identifiers must flow through every span, and trace context should propagate across agent and tool boundaries using W3C Trace Context, so an asynchronous, multi-service workflow stays one connected graph. If a tool runs in a separate service, such as an MCP server, pass the trace context along or the call appears orphaned.

**Treat retries as part of one logical operation.** [A 2026 field guide to the OpenTelemetry GenAI conventions](https://niteagent.com/blog/2026-08-07-otel-genai-agent-trace-field-guide/) states that when a request is retried automatically, the span should cover the whole logical operation including all retries, with the attempt count recorded as an attribute or a correlated log event. That keeps latency honest: the user waited for all attempts, not just the last one.

**Record meaningful events inside the span.** A [production observability walkthrough](https://dev.to/raju_dandigam/from-local-traces-to-production-observability-for-google-ai-agents-1l50) suggests span events for moments such as a retry being scheduled, an action blocked, a confirmation requested or a fallback chosen. These make a failed call explainable without extra log searches.

Also set the attributes you will want to filter on, such as agent name, operation name, provider and model, when the span is created, since sampling decisions can depend on them. The same field guide recommends this.

![Connect tool calls across runs, retries and agents](https://media.babylovegrowth.ai/blog-images/organization-30814/1794956564052_Connect-tool-calls-across-runs-retries-and-agents.jpeg)

## Instrument tool calls with MLflow

MLflow Tracing is [OpenTelemetry-compatible and supports the GenAI semantic conventions](https://mlflow.org/docs/latest/genai/tracing/) for export and ingestion, so it fits next to an existing observability stack. It captures inputs, outputs and metadata for each intermediate step. For supported libraries, one line turns on automatic tracing:

```python
import mlflow

mlflow.set_experiment("support-agent")
mlflow.openai.autolog()  # or another supported library, e.g. anthropic
```

With automatic tracing, [MLflow sets the span type for you](https://mlflow.org/docs/latest/genai/concepts/span/). For tools you wrote yourself, mark them as tool spans. The `TOOL` span type represents a tool execution, typically by an agent, such as a search query:

```python
import mlflow

@mlflow.trace(span_type="TOOL")
def search_orders(customer_id: str, status: str) -> list[dict]:
    # call your order API here
    return fetch_orders(customer_id, status)
```

The decorator records the function's inputs and outputs as the span's inputs and outputs, and nests the span under whichever agent span is active. Exceptions raised inside the function are captured on the span, so failed calls show up with their error.

Once calls are typed, you can query them. Span type is useful for searching and filtering particular spans in a large trace, in the UI or through the SDK:

```python
from mlflow.entities import SpanType

trace = mlflow.get_trace("<trace_id>")
tool_spans = trace.search_spans(span_type=SpanType.TOOL)
for s in tool_spans:
    print(s.name, s.inputs, s.outputs)
```

That gives you the answer to "what did the agent call, with what, and what came back" for one run. Running the same query over many traces gives you usage numbers: which tools are called most, which fail most, which are slowest. For a step-by-step introduction, see the [MLflow Tracing getting-started post](https://mlflow.org/blog/ai-observability-mlflow-tracing), and for how the approach scales to several agents, the [guide to tracing tools for multi-agent systems](https://mlflow.org/articles/best-llm-tracing-tools-for-multi-agent-systems-in-2026). More posts on this exact topic are collected under the [tool call tracing tag](https://mlflow.org/articles/tags/tool-call-tracing).

![Instrument tool calls with MLflow](https://media.babylovegrowth.ai/blog-images/organization-30814/1794956547846_Instrument-tool-calls-with-MLflow.jpeg)

### Make sure the record is written even when the call fails

The entry must exist whether the tool succeeds, times out or raises. Tuurt's guidance is to write the log entry in a `finally` block, before the exception propagates or the result returns to the agent, so a later failure can't leave the real call untraced. Tracing decorators and context managers handle this for you when they wrap the call. If you write custom logging around a tool, put the write in the same place.

## Protect sensitive data in tool call logs

Tool arguments and results often contain customer names, emails, account numbers or secrets. Logging them verbatim creates a second copy of that data with its own access and retention problems.

What the guidance says:

- **Redact at write time.** AWS says [PII redaction should happen before logs reach storage](https://docs.aws.amazon.com/wellarchitected/latest/agentic-ai-lens/agentops05-bp03.html), and that compliance audit trails should go to immutable storage such as S3 Object Lock in Compliance mode. The production walkthrough above notes that post-export redaction may be too late.
- **Hash, truncate or redact per policy.** The OWASP trace spec attaches inputs and outputs to each atomic action, with sensitive data hashed, truncated or redacted per privacy policy.
- **Keep secrets out.** [Microsoft's agentic AI design patterns](https://github.com/microsoft/agenticai-design-patterns/blob/main/assets/agentops-and-readiness/agent-production-readiness/docs/pillars/04-ai-security-and-observability.md) call for audit trails that record user, agent, version, tool, operation and approval for each action, exclude secrets and minimize sensitive content. They also ask that each tool call have a documented identity pattern and be authorized per operation by the downstream system.
- **Store less by default.** [Obot's audit schema](https://obot.ai/blog/what-to-log-when-an-agent-calls-a-tool-ai-agent-audit-trail-schema/) advises keeping only status, size, duration and a content hash for results by default, with full bodies stored separately under shorter retention.

If you handle regulated data, confirm retention periods and residency requirements with your security and legal teams. Microsoft's guidance says retention should be governed by data contracts that balance forensics, privacy, data residency, minimization and legal obligations.

### A tiered retention starting point

The AWS lens recommends tiered retention: operational logs for 30 to 90 days, compliance logs for 1 to 7 years and debug logs for 7 to 14 days. Treat those as a starting point to adapt, not a rule. A sensible split for tool calls:

| Tier        | What it holds                                      | Retention (AWS range) |
| ----------- | -------------------------------------------------- | --------------------- |
| Debug       | Full arguments and result bodies, redacted         | 7 to 14 days          |
| Operational | Metadata, status, duration, hashes, span events    | 30 to 90 days         |
| Compliance  | Who, which tool, what operation, approval, outcome | 1 to 7 years          |

AWS also prescribes a standardized JSON schema for agent operations: timestamp, trace ID, agent ID, session ID, operation type, decision rationale and outcome. Using one schema across all agents is what makes the compliance tier searchable years later.

## Use the logs for debugging, evaluation and usage

Once tool calls are traces, several jobs become queries rather than projects.

**Debugging.** Filter to tool spans with error status, open the parent trace and read the model turn that produced the arguments. Most tool failures are argument problems: a missing field, a wrong type, an ID the model invented. Seeing the exact arguments next to the error settles it quickly.

**Evaluation.** Tool-call traces are the input for scoring agent behavior: did the agent pick the right tool, were the arguments valid, did it use the result. MLflow supports automated [LLM-as-a-judge evaluation](https://mlflow.org/llm-as-a-judge) for LLMs and agents, so you can score traces you already collect rather than building a separate dataset.

**Usage and cost.** Aggregate by tool name, agent and user to see what is used and what is slow. If you also want to see how much of your product's inbound traffic originates from AI assistants, [tracking LLM traffic in GA4](https://babylovegrowth.ai/blog/google-analytics-check-llm-traffic) covers the web analytics side, while tool-call traces cover what happens after the agent starts working.

**Incident investigation.** For state-changing calls, a prior-state snapshot makes rollback an inverse operation with known data instead of manual reconstruction, as Tuurt notes. Pair it with the side-effect and reversibility flags so responders can sort the risky calls first.

## A rollout checklist

1. Turn on automatic tracing for your model and orchestration libraries, then mark custom tools with the `TOOL` span type.
2. Set a session ID on every run and propagate trace context to any tool that runs in another service.
3. Define your redaction rules and apply them before data is exported.
4. Record retries as attempts inside one span, with the attempt count.
5. Flag mutating calls, and snapshot prior state for those.
6. Split retention into debug, operational and compliance tiers.
7. Review a sample of traces weekly, and add evaluation on tool choice and argument validity.

## Our perspective

> We built MLflow around a simple view: an agent is a black box until every step it takes is a first-class citizen in your telemetry. A tool call is the step with the most consequences, because it is where the agent touches your systems, so it deserves the same rigor as any production API request.
>
> That is why we treat tool calls as typed spans in a trace, compatible with OpenTelemetry, rather than as strings in a log file. You keep your existing observability stack, and you get the join between the model turn, the arguments and the result that incident reviews need.
>
> Logging is also only the first half. The same traces feed evaluation, prompt work and gateway governance, and MLflow is fully open source under Linux Foundation governance with no enterprise feature paywalls. We would rather you start with the trace you already have than design a logging schema from scratch.
>
> _Kevin_

## Start tracing tool calls with MLflow

MLflow is an open-source platform for GenAI and agent lifecycle management: tracing of agentic reasoning, automated evaluation, prompt management and versioning, and a cross-provider AI Gateway. Its tracing integrates with major AI frameworks and is built on OpenTelemetry. You can add the autolog line to an existing agent today and see tool spans in the UI.

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

Start at [mlflow.org](https://mlflow.org), or read the [observability overview](https://mlflow.org/genai/observability) for how tracing fits into the wider platform.

## FAQ

### What is the difference between tool call logging and tracing?

Logging records standalone events. Tracing records the same events as spans linked into one run. For agents, tracing is the better fit because a tool call only makes sense alongside the model turn that chose it, the retries around it and the other calls in the same request.

### Should I log full tool arguments and results?

Log the exact arguments, since they are what reached your API. For results, store status, size, duration and a hash by default, and keep full bodies only where you need them, redacted and on a short retention. Apply redaction before data reaches storage.

### How long should I keep tool call logs?

There is no single answer. AWS's guidance suggests 7 to 14 days for debug logs, 30 to 90 days for operational logs and 1 to 7 years for compliance logs. Your legal and security teams set the final periods based on your obligations.

### Can a tool call log show why the agent chose a tool?

Not by itself. The log proves what was called and what came back. To understand the choice, also keep the context the model saw and any reasoning text, and treat that as a clue to intent rather than as proof.

### Does MLflow work with OpenTelemetry?

Yes. MLflow Tracing is OpenTelemetry-compatible and supports the GenAI semantic conventions, so you can export traces from an OpenTelemetry-instrumented app to MLflow and fit it into an existing stack.

## Sources

- [Structuring Logs and Traces for AI Agents: What to Record](https://quantaintelligence.ai/blog/structuring-logs-and-traces-for-ai-agents-what-to-record-at-each-step)
- [AWS AGENTPERF01-BP02: agent performance telemetry](https://docs.aws.amazon.com/wellarchitected/latest/agentic-ai-lens/agentperf01-bp02.html)
- [Tool Calling in Production LLM Agents](https://tuurt.com/en/blog/logging-tool-calling-llm-agents-production/)
- [Microsoft: Observability for generative AI and agentic AI systems](https://learn.microsoft.com/en-us/security/zero-trust/sfi/observability-ai-systems)
- [OWASP Agent Observability Standard: Trace with OpenTelemetry](https://aos.owasp.org/spec/trace/extend_opentelemetry/)
- [What to Log Around a Tool Call](https://singleaxis.ai/briefings/what-to-log-around-a-tool-call)
- [Agent Observability: Monitoring Decisions, Not Requests](https://www.mongodb.com/company/blog/technical/agent-observability-monitoring-decisions-not-requests)
- [The Agent Trace: A Field Guide to OpenTelemetry GenAI](https://niteagent.com/blog/2026-08-07-otel-genai-agent-trace-field-guide/)
- [From Local Traces to Production Observability](https://dev.to/raju_dandigam/from-local-traces-to-production-observability-for-google-ai-agents-1l50)
- [AWS AGENTOPS05-BP03: structured logging](https://docs.aws.amazon.com/wellarchitected/latest/agentic-ai-lens/agentops05-bp03.html)
- [Microsoft agentic AI design patterns: security and observability](https://github.com/microsoft/agenticai-design-patterns/blob/main/assets/agentops-and-readiness/agent-production-readiness/docs/pillars/04-ai-security-and-observability.md)
- [AI Agent Audit Trail Schema](https://obot.ai/blog/what-to-log-when-an-agent-calls-a-tool-ai-agent-audit-trail-schema/)
- [MLflow Tracing documentation](https://mlflow.org/docs/latest/genai/tracing/)
- [MLflow Spans documentation](https://mlflow.org/docs/latest/genai/concepts/span/)
