---
title: "Ship GenAI Conformant OTel Traces to MLflow for LLM Engineers"
description: "Practitioner walkthrough for LLM engineers: instrument with OpenTelemetry GenAI conventions, configure a Collector, and export traces to MLflow for..."
slug: otel-for-llm
tags:
  [
    otel for machine learning,
    otel setup for LLM,
    opentelemetry llm,
    otel integration with LLM,
    how to use otel for LLM,
    best otel for LLM,
    opentelemetry for llm,
    open telemetry llm,
    opentelemetry llm tracing,
    otel for llm,
  ]
date: 2026-09-12
image: https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1789188635809_Abstract-LLM-traces-in-operations-suite.jpeg
---

![Abstract LLM traces in operations suite](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1789188635809_Abstract-LLM-traces-in-operations-suite.jpeg)

Instrument LLM calls with OpenTelemetry using the GenAI semantic conventions, route OTLP data through a Collector, and export traces and metrics to your backend of choice. That is the whole architecture. Your first move: set `OTEL_EXPORTER_OTLP_ENDPOINT` and install an OTel-aware auto-instrumentation library like OpenLLMetry or OpenLIT so spans start flowing before you write a single manual trace.

---

> **TL;DR:**
>
> - OpenTelemetry collector configuration must bind to 0.0.0.0 ports to ensure traces reach your backend, and environment variables should point to the correct endpoint.
> - Tracking token counts, latency splits, error reasons, and full trace trees is essential for debugging and managing LLM application failures and costs.
> - Use the `gen_ai.*` semantic conventions with required attributes to ensure trace interoperability across backends without custom mapping.
> - Content capture should be opt-in and include redaction policies because full prompt and response data carry significant privacy risks.
> - Connecting OTLP traces with MLflow enables high-level analysis, management, and scoring of LLM performance at scale, complementing raw observability data.

---

## Table of Contents

- [Why Observability Matters for LLM Applications](#why-observability-matters-for-llm-applications)
- [What Are the OpenTelemetry GenAI Semantic Conventions?](#what-are-the-opentelemetry-genai-semantic-conventions)
- [How Do You Configure the OpenTelemetry Collector for LLM Traces?](#how-do-you-configure-the-opentelemetry-collector-for-llm-traces)
- [Where Should You Visualize LLM Traces and Metrics?](#where-should-you-visualize-llm-traces-and-metrics)
- [Production Considerations for GenAI Telemetry](#production-considerations-for-genai-telemetry)
- [How MLflow Complements OpenTelemetry for LLM Observability](#how-mlflow-complements-opentelemetry-for-llm-observability)
- [Should You Prioritize OTel Standards or Vendor-Specific Fields?](#should-you-prioritize-otel-standards-or-vendor-specific-fields)
- [Get Integrated Tracing and Evaluation With MLflow](#get-integrated-tracing-and-evaluation-with-mlflow)
- [Sources](#sources)
- [FAQ](#faq)

## Why Observability Matters for LLM Applications

LLM calls fail in ways traditional API monitoring never anticipated. A request can return a 200 status code and still burn through 4,000 output tokens generating an answer nobody asked for, or silently loop through three tool calls before giving up. Standard uptime dashboards miss all of it.

We track LLM behavior differently because the cost and failure modes are different. Token usage drives your bill directly, so input/output token histograms let you attribute cost per request, per model, per user. Latency in LLM systems is bimodal: fast cached responses and slow multi-step agent chains sit on the same graph, and a single average masks both. Trace trees showing tool calls and agent reasoning steps are often the only way to debug why an agent picked the wrong function.

What engineers should capture:

- Token-usage metrics (input and output token counts) for cost attribution
- Latency histograms segmented by model and operation type
- Error and finish-reason events to catch truncated or malformed completions
- Full trace trees for multi-step agent and tool-calling flows
- A deliberate choice on how much prompt/response content to capture, since that data carries real privacy weight

## What Are the OpenTelemetry GenAI Semantic Conventions?

The [OpenTelemetry GenAI semantic conventions](https://clickhouse.com/resources/engineering/opentelemetry-semantic-conventions) define a shared vocabulary for LLM telemetry so traces mean the same thing regardless of which backend renders them. Two attributes are required on every GenAI span: `gen_ai.operation.name` and `gen_ai.provider.name`. A third, `gen_ai.request.model`, is conditionally required whenever the operation targets a specific model.

Span naming follows a fixed pattern: `{gen_ai.operation.name} {gen_ai.request.model}`, so a chat completion against GPT style models becomes something like `chat gpt-4o`. Span kind is almost always `CLIENT`, since the application is calling out to an external inference service.

**Pro Tip:** _Adhering to `gen_ai._` naming means any OTLP compatible backend can render your traces without a custom mapping layer. Skip the conventions and you will rewrite that translation logic every time you switch dashboards.\*

Attributes worth adopting:

- `gen_ai.operation.name` and `gen_ai.provider.name` (required)
- `gen_ai.request.model` (conditionally required)
- `gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens` for cost tracking
- `gen_ai.response.finish_reasons` for quality signals
- Optional content attributes like `gen_ai.input.messages` and `gen_ai.output.messages`, which carry real cost and privacy tradeoffs since they store full prompt and response text

## How Do You Configure the OpenTelemetry Collector for LLM Traces?

The Collector is the contract boundary between your application and every downstream backend. Your app emits OTLP; the Collector receives it, normalizes it, and routes it wherever you point it. Get this boundary wrong and nothing downstream works, no matter how clean your instrumentation is.

![Application telemetry routed through an OTel Collector](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1789188693700_Application-telemetry-routed-through-an-OTel-Collector.jpeg)

The OTLP receiver listens on two default ports: 4317 for gRPC and 4318 for HTTP, with the HTTP traces path defaulting to `/v1/traces`. The most common production failure is a binding mismatch: a Collector configured to listen on `localhost` when the application is running in a separate container or on a remote host simply never receives the data, and there is no error message to point you at the problem.

A minimal Collector receiver config looks like this:

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318
```

Binding to `0.0.0.0` instead of `127.0.0.1` is the fix for most "traces never arrive" bug reports.

On the application side, set the endpoint once, usually as an environment variable:

```
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4318/v1/traces
```

For instrumentation, you have two real options. Auto-instrumentation through [OpenLLMetry](https://github.com/traceloop/openllmetry) or OpenLIT wraps your existing LLM client calls and emits `gen_ai.*` spans without touching your business logic. Manual instrumentation gives you finer control when you need custom span attributes:

```python
with tracer.start_as_current_span("chat gpt-4o", kind=SpanKind.CLIENT) as span:
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.provider.name", "openai")
    span.set_attribute("gen_ai.request.model", "gpt-4o")
```

A working local test flow:

1. Start the Collector with the config above
2. Point your application's OTLP env vars at the Collector
3. Fire a single LLM call from your app
4. Confirm the span shows up in your trace viewer with `gen_ai.*` attributes populated

## Where Should You Visualize LLM Traces and Metrics?

Metrics and traces need different homes. Prometheus paired with Grafana handles the numeric side well: token histograms, latency percentiles by model, request rate. Jaeger, or honestly any OTLP-compatible trace viewer, handles the trace side, showing you the full tree of a request from the initial call through every tool invocation.

A [local dashboard walkthrough](https://opentelemetry.io/blog/2026/genai-observability/) using a lightweight OTLP viewer at `http://localhost:4318` is a fast way to confirm your setup works before committing to a full backend, and it even documents the settings needed to pull GenAI telemetry out of VS Code Copilot, a useful example if your team already generates spans from an IDE assistant.

Dashboard elements worth building first:

- Latency histograms broken out by `gen_ai.request.model`
- Token usage over time, split by input and output
- A breakdown of `gen_ai.response.finish_reasons` to catch truncation spikes
- A chat-style trace viewer, available in some tools when content capture is enabled, that renders `gen_ai.*` spans as readable conversation turns

To validate the pipeline, filter traces by model name, confirm token counts land in the right range for your prompts, and inspect a tool-call span to make sure nested calls appear correctly under the parent span.

## Production Considerations for GenAI Telemetry

Token-heavy workloads generate a lot of span data fast, so sampling and aggregation decisions matter more here than in typical web service tracing. Tail-based sampling that keeps error traces and slow traces while dropping routine successful calls is usually the right tradeoff.

Content capture is the biggest privacy decision you will make. Full prompt and response text in `gen_ai.input.messages` and `gen_ai.output.messages` is invaluable for debugging and dangerous for compliance if it contains customer data. Make capture opt-in, apply redaction where feasible, and set explicit retention windows.

- Use tail-based sampling to keep error and outlier traces without storing every routine call
- Make content capture opt-in, with redaction and retention policies enforced before data leaves the app
- Apply role-based access control to telemetry backends that hold prompt or response content
- Monitor Collector health endpoints directly, since a silently dying Collector produces no traces and no obvious alarm

**Pro Tip:** _Set token-usage alert thresholds per model, not globally. A GPT-4 class model burning 2,000 tokens per request is normal; the same count on a lightweight model usually signals a prompt bug._

Collector reliability deserves its own attention: allocate enough memory for buffering under load, expose health check endpoints, and double check TLS and CORS settings if any browser-based client sends OTLP directly, since a misconfigured CORS policy will block traces just as effectively as a firewall.

## How MLflow Complements OpenTelemetry for LLM Observability

OpenTelemetry gets your traces out of the application. MLflow gives teams a place to make sense of them at scale. MLflow provides production-grade observability with deep tracing of agentic reasoning, so a multi-agent pipeline that OTel captures as a flat set of spans becomes a navigable record of which agent called which tool and why.

Where MLflow adds the most value beyond raw OTLP export: centralized tracing across many agents and providers in one view, automated evaluation pipelines using LLM-as-a-Judge to score output quality at scale instead of spot-checking transcripts by hand, and an AI Gateway for managing prompts and provider credentials with real governance instead of scattered API keys.

Teams already running the [OpenTelemetry Collector setup](https://mlflow.org/blog/opentelemetry-tracing-support) above can send that same instrumentation into [MLflow's tracing pipeline](https://mlflow.org/blog/ai-observability-mlflow-tracing) without re-instrumenting anything.

## Should You Prioritize OTel Standards or Vendor-Specific Fields?

Start with the required `gen_ai.*` attributes every time. Portability across backends matters more early on than any single vendor's UI polish. Add vendor-specific fields later, behind feature flags, once you know which backend features you actually need. Standard first, extras second.

> _— Kevin_

## Get Integrated Tracing and Evaluation With MLflow

An OTel Collector and a trace viewer will tell you what happened in a single LLM call. They will not tell you whether that call was any good, or let you compare prompt versions across a hundred production traces without exporting everything to a spreadsheet. That gap is where [MLflow's GenAI platform](https://mlflow.org/genai) fits in.

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

MLflow layers on top of your existing OTLP instrumentation: agent-level tracing that groups tool calls into readable reasoning chains, automated scoring through LLM-as-a-Judge evaluation so quality regressions surface before customers report them, and a governance layer for prompt versions and provider credentials across teams. If your stack includes classical ML models alongside LLM components, [MLflow's model lifecycle tools](https://mlflow.org/classical-ml) cover that side too, under the same observability practice. Start by pointing your existing OTel traces at MLflow's tracing endpoint and run one evaluation pass against a recent batch of production traffic to see where quality actually stands.

## Sources

For exact configuration syntax, consult the OpenTelemetry GenAI semantic conventions directly and the Collector's OTLP receiver documentation for receiver and protocol details. Both are updated more often than any third-party summary.

- [Inside the LLM Call: GenAI Observability with OpenTelemetry](https://opentelemetry.io/blog/2026/genai-observability/)
- [OpenTelemetry GenAI semantic conventions (ClickHouse engineering summary)](https://clickhouse.com/resources/engineering/opentelemetry-semantic-conventions)
- [traceloop/openllmetry (OpenLLMetry repo)](https://github.com/traceloop/openllmetry)

## FAQ

### What Is the Minimum Setup for OTel With an LLM Application?

Install an auto-instrumentation library like OpenLLMetry, set `OTEL_EXPORTER_OTLP_ENDPOINT` to your Collector, and confirm a single traced LLM call shows up with `gen_ai.*` attributes populated.

### Which Attributes Are Required by the GenAI Semantic Conventions?

`gen_ai.operation.name` and `gen_ai.provider.name` are required on every span, and `gen_ai.request.model` is required whenever a specific model is targeted.

### Why Do My OTLP Traces Never Reach the Collector?

The most common cause is a binding mismatch, usually a Collector listening on `localhost` while the application runs in a separate container or remote host; bind the receiver to `0.0.0.0` instead.

### Should I Capture Full Prompt and Response Content in Traces?

Treat it as opt-in: full content in `gen_ai.input.messages` and `gen_ai.output.messages` is useful for debugging but should carry redaction and retention rules given its privacy weight.

### Can MLflow Work Alongside My Existing OTel Instrumentation?

Yes. MLflow accepts OTel-based traces and adds agent-level tracing, LLM-as-a-Judge evaluation, and prompt governance on top of the same OTLP data your Collector already routes.

## Recommended

- [Practical AI Observability: Getting Started with MLflow Tracing](https://mlflow.org/blog/ai-observability-mlflow-tracing)
- [LLM & Agent Observability](https://mlflow.org/genai/observability)
- [Agent & LLM Engineering](https://mlflow.org/genai)
