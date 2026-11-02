---
title: "Best Open-Source LLM Tracing Tools in 2026: 5 Tools Compared"
description: "Compare open-source LLM tracing tools in 2026: MLflow, Langfuse, Arize Phoenix, Opik, and OpenLLMetry. Licenses, OpenTelemetry support, and what each takes to run."
slug: best-open-source-llm-tracing-tools-2026
tags:
  [
    best LLM tracing tools,
    open-source LLM tracing,
    self-hosted LLM tracing,
    OpenTelemetry LLM tracing,
    AI agent tracing tools,
    MLflow tracing,
  ]
date: 2026-11-02
---

**Last reviewed: November 2, 2026.** The best open-source LLM tracing tools in 2026 are MLflow, Langfuse, Arize Phoenix, Opik, and OpenLLMetry. Each one records the steps an LLM or GenAI app takes, such as model calls, tool calls, and retrieval, as a trace you can inspect. Four of them run on your own infrastructure, and the fifth, OpenLLMetry, is a library that sends traces to a backend you choose.

When an agent gives a wrong answer, the final output rarely tells you why. You need to see each step: the prompt it sent, the tool it called, the documents it retrieved, and where the tokens went. An LLM tracing tool records those steps as a trace that you can inspect, search, and turn into a test case.

We work on MLflow Tracing, and this is the comparison we'd want if we were picking a tracer today: licenses, OpenTelemetry support, and what each tool takes to run.

**How we ranked them:** by how much of the tracing workflow each tool covers on its own: capturing traces from any language through OpenTelemetry, debugging them in a UI, tracking cost and sessions, evaluating what you traced, and how little you have to run to self-host. **Disclosure:** the MLflow team wrote this guide, and MLflow is ranked first. Every competitor description links to that project's own documentation, and licenses were checked against each project's LICENSE file on GitHub on the date above.

## Quick answer: what are the best open-source LLM tracing tools in 2026?

The best open-source LLM tracing tools in 2026 are:

1. **MLflow**: best self-hosted tracing that speaks OpenTelemetry in both directions, with evaluation in the same tool. It is Apache 2.0, and every feature is free.
2. **Langfuse**: best for tracing plus prompt management, with an MIT-licensed core.
3. **Arize Phoenix**: best for OpenInference tracing with a prompt playground and span replay, if a source-available license works for you.
4. **Opik**: best for Apache 2.0 tracing with typed spans and built-in evaluation.
5. **OpenLLMetry**: best for adding LLM spans to an observability stack you already run. It is instrumentation, not a platform.

## What to look for in an open-source LLM tracing tool

"Open source" covers everything from an instrumentation library to a full platform with a UI and a database. Use these criteria to compare them.

- **License.** Read the license file. Apache 2.0 and MIT are OSI-approved. Some projects are source-available, and some keep features under a commercial license.
- **OpenTelemetry in and out.** A tool that accepts OTLP can trace services written in any language. A tool that also exports OTLP lets you send the same traces to the monitoring stack you already run, so you are not locked in.
- **Agent and session views.** Multi-step agents need span trees, tool-call details, and a way to group the turns of one conversation.
- **Token usage and cost.** You need cost per trace, user, and session to keep spend in check.
- **From trace to test.** Your best regression tests come from real failures. Check whether you can score traces and save them to a dataset in the same tool.
- **What it takes to self-host.** Count the services you have to run, back up, and upgrade.

## Open-source LLM tracing tools comparison table

| Tool          | License                                | Maintained by    | How traces get in                                               | Best for                                        |
| ------------- | -------------------------------------- | ---------------- | --------------------------------------------------------------- | ----------------------------------------------- |
| MLflow        | Apache 2.0                             | Linux Foundation | One-line autolog for 40+ libraries; OTLP over HTTP in, OTLP out | OpenTelemetry-native tracing plus evaluation    |
| Langfuse      | MIT core, commercial `ee` features     | ClickHouse       | Python and TypeScript SDKs; OTLP over HTTP                      | Tracing plus prompt management                  |
| Arize Phoenix | Elastic License 2.0 (source-available) | Arize AI         | OpenInference instrumentation; OTLP over HTTP, gRPC             | Prompt playground and span replay               |
| Opik          | Apache 2.0                             | Comet            | Python and TypeScript SDKs; OTLP over HTTP                      | Typed spans with built-in evaluation            |
| OpenLLMetry   | Apache 2.0                             | Traceloop        | Instrumentation that emits standard OpenTelemetry               | LLM spans in Datadog, Grafana, or another stack |

MLflow is the only tool on this list that is governed by a foundation rather than a single company, and it has no paid tier or license-gated features.

## 1. MLflow

MLflow Tracing captures LLM and agent traces with one line of code for 40+ libraries, accepts OpenTelemetry traces from any language at its `/v1/traces` endpoint, and exports traces over OTLP to other backends. It is part of MLflow, an Apache 2.0 Linux Foundation project, so the same server also evaluates and monitors what you traced.

- **Best for:** teams that want open, OpenTelemetry-native tracing and evaluation in one self-hosted tool.
- **Where it falls short:** the MLflow server [accepts OTLP over HTTP only](https://mlflow.org/docs/latest/genai/tracing/opentelemetry/ingest/). If your services use gRPC exporters, run an OpenTelemetry Collector in front of it.
- **Key fact:** 28K+ GitHub stars, and the latest release is v3.17.2 (October 21, 2026).

### How do traces get into MLflow?

For Python and TypeScript apps, turn on automatic tracing for your library. For example, with the OpenAI SDK:

```python
import mlflow

mlflow.openai.autolog()
```

MLflow's [integrations list](https://mlflow.org/docs/latest/genai/tracing/integrations/) covers 40+ LLM and agent libraries, including OpenAI, LangChain, LangGraph, and LlamaIndex. For services written in Java, Go, Rust, or any other language with an OpenTelemetry SDK, point the OTLP exporter at the MLflow server and pass the experiment ID in a header:

```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:5000/v1/traces
export OTEL_EXPORTER_OTLP_TRACES_HEADERS=x-mlflow-experiment-id=123
```

MLflow [ingests traces in the OpenTelemetry GenAI semantic conventions](https://mlflow.org/docs/latest/genai/tracing/opentelemetry/genai-semconv/), and can export in that format when you set `MLFLOW_ENABLE_OTEL_GENAI_SEMCONV=true`. It also [translates span attributes](https://mlflow.org/docs/latest/genai/tracing/opentelemetry/attribute-mapping/) from OpenLLMetry, OpenInference, and Langfuse instrumentation. So you can keep the instrumentation you already have. In production, the [`mlflow-tracing` package](https://mlflow.org/docs/latest/genai/tracing/lightweight-sdk/) gives you tracing with far fewer dependencies than the full MLflow package.

### Key tracing features

- **Trace explorer.** MLflow 3.16 made a [redesigned trace explorer](https://github.com/mlflow/mlflow/releases/tag/v3.16.0) the default, with span links and custom trace views.
- **Token usage and cost**, recorded on each trace ([docs](https://mlflow.org/docs/latest/genai/tracing/token-usage-cost/)).
- **Users and sessions** to group multi-turn conversations ([docs](https://mlflow.org/docs/latest/genai/tracing/track-users-sessions/)).
- **Trace analytics.** MLflow 3.17 added [opt-in daily rollups for trace analytics](https://github.com/mlflow/mlflow/releases/tag/v3.17.0).
- **From trace to evaluation.** Score traces with LLM judges and code scorers, and save them to evaluation datasets, on the same server. See [the open-source LLM evaluation guide](/articles/best-open-source-llm-evaluation-tools-2026).

### How do you self-host MLflow?

You can run the full MLflow platform on your laptop with two commands:

```bash
pip install mlflow
mlflow server --port 5000
```

This starts the server and UI with a SQLite database, which is the [default backend store](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/). For production, point MLflow at PostgreSQL, MySQL, or Microsoft SQL Server, and deploy it to Kubernetes with the [official Helm chart](https://github.com/mlflow/mlflow/tree/master/charts). See the [self-hosting guide](https://mlflow.org/docs/latest/self-hosting/) for authentication and access control.

Open-source MLflow has no paid tier and no license keys. Every feature above is in the open-source release.

## 2. Langfuse

Langfuse is an open-source LLM engineering platform for tracing, prompt management, and evaluation. Its core is MIT-licensed, and some enterprise features are under a separate license. It [accepts OpenTelemetry traces](https://langfuse.com/integrations/native/opentelemetry) over HTTP at `/api/public/otel` and has native Python and TypeScript SDKs. Langfuse is maintained by [ClickHouse, which acquired it in January 2026](https://langfuse.com/blog/joining-clickhouse).

- **Best for:** teams that want tracing and prompt management in one platform and are comfortable running ClickHouse.

### Key tracing features

- **Sessions** that [group traces and show a session replay](https://langfuse.com/docs/observability/features/sessions) for multi-turn apps.
- **OpenTelemetry ingest** over [HTTP/JSON and HTTP/protobuf](https://langfuse.com/integrations/native/opentelemetry). gRPC is not supported yet. Java and Go apps can send traces through OpenLLMetry or OpenLIT.

### How do you self-host Langfuse?

Langfuse [supports self-hosting](https://langfuse.com/self-hosting) with Docker Compose for local testing, or on Kubernetes with Helm for production. It needs PostgreSQL, ClickHouse, Redis or Valkey, and S3-compatible blob storage. The core features are free, but the docs note that "some add-on features require a license key."

## 3. Arize Phoenix

Arize Phoenix is an observability and evaluation app from Arize AI, built on OpenTelemetry and the [OpenInference](https://github.com/Arize-ai/openinference) conventions. It is source-available under the Elastic License 2.0, which lets you self-host it for free but does not let you offer Phoenix as a managed service. Phoenix accepts OTLP over HTTP and gRPC.

- **Best for:** teams that want a prompt playground and span replay next to their traces, and do not need an OSI-approved license.

### Key tracing features

- **Prompt Playground** to compare prompts side by side, and **Span Replay** to rerun an LLM call with different inputs ([docs](https://arize.com/docs/phoenix)).
- **OpenTelemetry ingest** at `/v1/traces` over HTTP, or over gRPC on port 4317, with the `phoenix.otel` Python helper ([docs](https://arize-phoenix.readthedocs.io/projects/otel/)).

### How do you self-host Arize Phoenix?

Phoenix says it is ["free to self-host with no feature limitations"](https://arize.com/docs/phoenix/self-hosting). You can run it from the CLI, with Docker, or on Kubernetes with Helm. It uses SQLite by default and [PostgreSQL for production](https://arize.com/docs/phoenix/self-hosting/architecture).

## 4. Opik

Opik is an Apache 2.0 platform from Comet for tracing, evaluating, and monitoring LLM apps. Self-hosted Opik includes all features except user management. It has official Python and TypeScript SDKs and accepts OTLP over HTTP only. Spans are typed, such as LLM, tool, and guardrail spans, and a thread ID groups related traces, such as the turns of one conversation.

- **Best for:** teams that want Apache 2.0 tracing with evaluation built in, and can run several backing services.

### Key tracing features

- **Typed spans and threads.** Set a span `type` and a trace `thread_id` to filter spans and group traces into threads ([docs](https://www.comet.com/docs/opik/tracing/log_traces)).
- **OpenTelemetry ingest** over HTTP at `/api/v1/private/otel` on a self-hosted instance. Use the HTTP exporter, because the gRPC exporter is not supported ([docs](https://www.comet.com/docs/opik/tracing/opentelemetry/overview)).

### How do you self-host Opik?

Opik [supports self-hosting](https://www.comet.com/docs/opik/self-host/overview) with all features included. The [local Docker Compose setup](https://www.comet.com/docs/opik/self-host/local_deployment) runs MySQL, Redis, ClickHouse, ZooKeeper, and MinIO, and Comet says it is not meant for production. For production, use the [Helm chart](https://www.comet.com/docs/opik/self-host/kubernetes).

## 5. OpenLLMetry

OpenLLMetry is an Apache 2.0 set of OpenTelemetry instrumentations for LLM apps, from Traceloop. It has no UI or storage of its own. You add one `Traceloop.init()` call, and it sends standard OpenTelemetry traces to Traceloop or to the observability stack you already run. It has quick starts for Python, TypeScript, Go, and Ruby.

- **Best for:** teams that already run an observability backend and want LLM spans in it.

### Where can OpenLLMetry send traces?

The project lists [integrations](https://www.traceloop.com/docs/openllmetry/integrations/introduction) for Datadog, Grafana, Honeycomb, Sentry, Langfuse, the OpenTelemetry Collector, and more. Because it emits standard OTLP, you can also point it at MLflow's `/v1/traces` endpoint, and MLflow maps its attributes to trace columns.

### How do you self-host OpenLLMetry?

There is nothing to host. OpenLLMetry runs inside your app, and you choose and operate the backend that stores the traces.

## Self-hosting compared: what each tool takes to run

| Tool          | What you run                                                                     | Paid license needed for                                      |
| ------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| MLflow        | One server; SQLite to start, PostgreSQL, MySQL, or SQL Server for production     | Nothing                                                      |
| Langfuse      | Web and worker containers, PostgreSQL, ClickHouse, Redis or Valkey, blob storage | Some add-on features                                         |
| Arize Phoenix | One application; SQLite or PostgreSQL                                            | Nothing, but offering it as a managed service is not allowed |
| Opik          | Backend and frontend services, MySQL, ClickHouse, Redis, ZooKeeper, MinIO        | Nothing                                                      |
| OpenLLMetry   | Nothing; it runs in your app and sends traces to a backend you choose            | Nothing, though your backend may have its own terms          |

MLflow and Arize Phoenix each run as a single application on one database, which is the least to operate of the platforms here.

## How to choose an open-source LLM tracing tool

- **You want traces and evaluation in one tool, with the least to operate and no license keys:** MLflow.
- **Prompt management is central to how you work, and running ClickHouse is fine:** Langfuse.
- **You want a prompt playground and span replay, and you will not resell it as a service:** Arize Phoenix.
- **You want Apache 2.0 with evaluation built in, and can run more services:** Opik.
- **Your traces already go to Datadog, Grafana, or Honeycomb:** OpenLLMetry. You can still send the same spans to MLflow later.

We left out Helicone, which is also Apache 2.0. In March 2026, Mintlify [announced that it is acquiring Helicone](https://www.mintlify.com/blog/mintlify-acquires-helicone) and said Helicone will continue in maintenance mode, and its [latest GitHub release](https://github.com/Helicone/helicone/releases) is from August 2025.

## FAQ

### What is the best open-source LLM tracing tool in 2026?

MLflow, if you want tracing and evaluation in one tool. It is Apache 2.0 with no paid tier, traces 40+ libraries with one line of code, accepts and exports OpenTelemetry, and evaluates traces on the same server. Pick Langfuse if prompt management matters most, or OpenLLMetry if you want LLM spans in an existing observability stack.

### Can MLflow ingest OpenTelemetry traces from other tools?

Yes. The MLflow server exposes an OTLP/HTTP endpoint at `/v1/traces`, so any application with an OpenTelemetry SDK, in Java, Go, Rust, or another language, can send traces to it. MLflow reads the OpenTelemetry GenAI semantic conventions and maps attributes from OpenLLMetry, OpenInference, and Langfuse instrumentation. For gRPC exporters, run an OpenTelemetry Collector in front of it.

### Is Arize Phoenix open source?

Phoenix is source-available, not OSI open source. Its code is public under the Elastic License 2.0, which allows free self-hosting with no feature limits but does not allow offering Phoenix as a managed service. If you need an OSI-approved license, MLflow and Opik are Apache 2.0, and the Langfuse core is MIT.

### How do I trace an LLM app written in Java or Go?

Use OpenTelemetry. MLflow, Langfuse, and Opik accept OTLP over HTTP, so configure the HTTP exporter in your language's OpenTelemetry SDK, or run an OpenTelemetry Collector in front for gRPC. Arize Phoenix accepts both HTTP and gRPC. OpenLLMetry also has a Go quick start.

### What is the difference between LLM tracing and LLM observability?

Tracing records what happened inside one request: each model call, tool call, and retrieval, with inputs, outputs, latency, and tokens. Observability is the wider practice of watching an app over time, built on those traces plus metrics, cost, quality scores, and alerts. Tracing comes first, because you cannot evaluate or monitor what you never recorded.

## Get started with MLflow tracing

Install MLflow with `pip install mlflow`, start a server with `mlflow server`, and follow the [tracing quickstart](https://mlflow.org/docs/latest/genai/tracing/quickstart/) to record your first trace in a few minutes.

## Recommended

- [Best LLM Tracing Tools for Multi-Agent Systems in 2026](/articles/best-llm-tracing-tools-for-multi-agent-systems-in-2026)
- [Top LLM Observability Tools in 2026: A Pro Guide](/articles/top-llm-observability-tools-in-2026-a-pro-guide)
- [Best Open-Source LLM Evaluation Tools in 2026](/articles/best-open-source-llm-evaluation-tools-2026)
- [LLM Tracing | MLflow](https://mlflow.org/llm-tracing)
- [Langfuse Alternative | MLflow](https://mlflow.org/langfuse-alternative)
- [Arize Phoenix Alternative | MLflow](https://mlflow.org/arize-phoenix-alternative)
