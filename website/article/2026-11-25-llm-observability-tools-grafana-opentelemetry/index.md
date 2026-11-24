---
title: "LLM Observability Tools That Work With Grafana (and OpenTelemetry)"
description: "Which LLM observability tools work with Grafana? Five options with a documented Grafana path, from Grafana Cloud AI Observability to MLflow, with licenses and trade-offs."
slug: llm-observability-tools-grafana-opentelemetry
tags:
  [
    LLM observability Grafana,
    Grafana LLM monitoring,
    OpenTelemetry LLM tracing,
    LLM observability tools,
    Grafana Tempo,
    open source LLM observability,
    AI agent observability,
  ]
date: 2026-11-25
---

Five tools have a documented path into Grafana: Grafana Cloud AI Observability, OpenLIT, OpenLLMetry, MLflow, and a plain OpenTelemetry Collector feeding Grafana Tempo. All five speak OTLP, so the transport is portable, but attribute names differ between tools, so dashboards may need remapping. The real choice is whether you want LLM dashboards inside Grafana, an LLM-specific trace UI beside it, or both.

This page lists the five, with license, setup path, and trade-offs, using each project's own docs and repository, all checked on 2026-11-24.

Disclosure: this page was written by MLflow maintainers, and MLflow is one of the five. This page covers the open-source project only. We list the tools unranked, and we say where MLflow falls short.

## TL;DR

Short version for each tool:

- **Grafana Cloud AI Observability:** hosted dashboards for LLMs, vector databases, GPUs, and MCP servers, set up through the OpenLIT integration.
- **OpenLIT:** Apache-2.0 SDKs and a self-hosted backend that emit standard OTLP, which can go to Grafana.
- **OpenLLMetry:** Apache-2.0 instrumentation from Traceloop, with Grafana listed as a tested destination.
- **MLflow:** Apache-2.0 tracing and evaluation server that can export traces over OTLP to Grafana Cloud, with metrics going through Prometheus.
- **OpenTelemetry Collector plus Grafana Tempo:** the build-it-yourself route, with no LLM-specific UI.

## How did we pick these tools?

We started from the question people type: which LLM observability tools work with Grafana? We kept a tool only if its own docs or repository describe sending data to Grafana, Tempo, or Prometheus, or name Grafana as a supported destination. We dropped tools where we found no such statement.

That rule removes two popular ones. We found no documented Grafana integration for Langfuse or Arize Phoenix, and we say more about both near the end.

We did not rank the five. The order below is a reading order, not a verdict.

## How do the five compare?

|                         | Grafana Cloud AI Observability                      | OpenLIT                             | OpenLLMetry                             | MLflow                                        | OTel Collector + Tempo                |
| ----------------------- | --------------------------------------------------- | ----------------------------------- | --------------------------------------- | --------------------------------------------- | ------------------------------------- |
| What it is              | Hosted dashboards in Grafana Cloud                  | SDKs plus a self-hosted backend     | Instrumentation library                 | Tracing and evaluation server                 | Pipeline plus a tracing backend       |
| License                 | Hosted service                                      | Apache-2.0                          | Apache-2.0                              | Apache-2.0                                    | Tempo: AGPL-3.0                       |
| Documented Grafana path | Native, through the OpenLIT integration             | OTLP to Grafana or any OTLP backend | Grafana listed as a tested destination  | Grafana listed in the OTLP export docs        | Grafana's built-in Tempo data source  |
| What you run            | Your instrumented app                               | App plus optional OpenLIT backend   | Your app, plus somewhere to send traces | App, plus an MLflow server if you want its UI | Collector and Tempo, or Grafana Cloud |
| LLM-specific UI         | Dashboards for performance, cost, tokens, and usage | OpenLIT's own UI                    | None, you bring a backend               | MLflow trace UI and LLM-judge evaluation      | None                                  |

Grafana Cloud's [pricing page](https://grafana.com/pricing/) lists a free tier with 50 GB of traces a month, 10k active metric series, and 14-day retention, and a Pro plan from $19 a month plus usage. These figures change, so check the page before you plan around them.

## 1. Which tool gives you LLM dashboards in Grafana with the least setup?

Grafana Cloud AI Observability (the docs page is titled "Grafana Cloud OpenLIT Observability") is the shortest path if you already use Grafana Cloud. Its [docs](https://grafana.com/docs/grafana-cloud/observe-and-act/monitor-applications/ai-observability/) describe monitoring of "LLMs, vector databases, GPUs and MCP Servers," built on OpenTelemetry and set up by installing the OpenLIT integration. You get dashboards without running an LLM backend yourself.

**Evidence:** The [setup guide](https://grafana.com/docs/grafana-cloud/observe-and-act/monitor-applications/ai-observability/setup/) has you open Connections in your Grafana Cloud stack, search for AI Observability, follow the card's instructions to instrument your application, and install 5 dashboards. The [GenAI monitoring page](https://grafana.com/docs/grafana-cloud/observe-and-act/monitor-applications/ai-observability/genai/) lists performance tracking, cost management, token analytics, and usage insights, with provider coverage for OpenAI, Anthropic, Google, AWS Bedrock, and Cohere, and framework coverage for LangChain, LlamaIndex, CrewAI, and LiteLLM.

**Where it falls short:** The setup page installs 5 dashboards, including a GenAI Evaluations dashboard for AI quality and safety. The GenAI monitoring page we read mentions trace context only in agent interaction logs, so check whether you can browse individual prompts and responses the way you can in a dedicated LLM trace UI. It also needs Grafana Cloud, so it is not an option if traces must stay inside your own network.

**Best for:** teams already on Grafana Cloud that want cost, token, and latency dashboards for LLM apps this week.

## 2. Which open-source SDK sends LLM telemetry to Grafana?

OpenLIT is an Apache-2.0 project that emits standard OTLP traces and metrics, and its README says they can go to "OpenLIT, Grafana, Datadog, New Relic, SigNoz, or any OTLP backend." Its Python, TypeScript, and Go SDKs follow the OpenTelemetry GenAI semantic conventions.

**Evidence:** The [OpenLIT repository](https://github.com/openlit/openlit) says "OpenLIT is licensed under the Apache License 2.0", and its FAQ says its OTLP data "can go to OpenLIT, Grafana, Datadog, New Relic, SigNoz, or any OTLP backend." Self-hosting is `docker compose up -d` or Kubernetes, and the README says your telemetry stays under your control. OpenLIT is also the instrumentation behind Grafana Cloud AI Observability, so the same SDK works with the hosted option above.

**Where it falls short:** The README does not describe Grafana dashboards of its own, so on a self-managed Grafana you build or import the panels.

**Best for:** teams that want an Apache-2.0 SDK, the option to self-host, and the freedom to send the same data to Grafana or elsewhere.

## 3. Which instrumentation library has Grafana as a tested destination?

OpenLLMetry is an Apache-2.0 set of extensions on top of OpenTelemetry for LLM apps, built and maintained by Traceloop. Its README lists Grafana under "Supported (and tested) destinations," and its semantic conventions are now part of OpenTelemetry. It instruments your code. It does not store traces.

**Evidence:** The [OpenLLMetry repository](https://github.com/traceloop/openllmetry) states the license and the destination list. Traceloop's [Grafana integration guide](https://www.traceloop.com/docs/openllmetry/integrations/grafana) gives two routes. Traceloop's guide still shows a Grafana Agent route, so check Grafana's current docs for which agent to use. Through an agent you point `TRACELOOP_BASE_URL` at the agent's OTLP port. Without an agent you set `TRACELOOP_BASE_URL` to `https://otlp-gateway-<zone>.grafana.net/otlp` and pass a Basic auth header built from your stack ID and API key.

**Where it falls short:** The main SDK is Python, and the JavaScript and TypeScript version is a separate project. Because it ships no backend, you must pick one, and in Grafana that means Tempo traces without an LLM-specific view.

**Best for:** teams that want vendor-neutral instrumentation first and will pick the backend, including Grafana Tempo, afterwards.

## 4. Which tool adds evaluation and a trace UI beside Grafana?

MLflow is an Apache-2.0 open-source platform for agents, LLMs, and ML models, hosted by the Linux Foundation. Its OpenTelemetry [export docs](https://mlflow.org/docs/latest/genai/tracing/opentelemetry/export/) list Grafana among the supported platforms, so MLflow can send the same traces to Grafana and keep them in its own trace UI for evaluation.

**Evidence:** MLflow's [tracing docs](https://mlflow.org/docs/latest/genai/tracing/) say it is "fully compatible with OpenTelemetry and natively supports GenAI Semantic Conventions for export and ingestion." The export docs set the destination with the standard `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` variable, and say `MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT=true` sends traces to both MLflow and the other backend. By default exported spans carry MLflow's `mlflow.*` attributes, and `MLFLOW_ENABLE_OTEL_GENAI_SEMCONV=true` switches to `gen_ai.*`. The docs link to Grafana Cloud's OTLP documentation but give no Grafana-specific steps.

Grafana's [OTLP guide](https://grafana.com/docs/grafana-cloud/send-data/otlp/send-data-otlp/) fills that gap. The endpoint looks like `https://otlp-gateway-<REGION>.grafana.net/otlp`, the token comes from your stack's OpenTelemetry tile, and a signal-specific variable such as `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` needs the full `/v1/traces` path. Combining the two sets of docs gives this configuration:

```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="https://otlp-gateway-<REGION>.grafana.net/otlp/v1/traces"
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL="http/protobuf"
export OTEL_EXPORTER_OTLP_TRACES_HEADERS="Authorization=Basic%20<token from your stack's OpenTelemetry tile>"
export MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT=true
export MLFLOW_ENABLE_OTEL_GENAI_SEMCONV=true
```

We assembled this from the two docs pages and have not run it against a Grafana Cloud stack. Grafana's guide does not name the protocol, so confirm the value on your stack's OpenTelemetry tile, and set the endpoint before any trace starts.

MLflow can also export metrics. The docs describe a `mlflow.trace.span.duration` histogram with labels for span type, status, and experiment, and show a direct route into Prometheus when you start it with `--web.enable-otlp-receiver` and `--enable-feature=otlp-deltatocumulative`. The labels also include trace tags and metadata, so watch label cardinality. That caution is ours, not the docs'. Metrics export is opt-in and needs `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` set.

**Where it falls short:** MLflow is a broader project than an LLM-dashboard tool, and Grafana shows only the telemetry you export. The dual-export path is documented in MLflow's docs but has no Grafana example there. Self-hosting an MLflow server for the UI means you own a SQL backend store, artifact storage, and upgrades. Hosted MLflow services exist, but this entry describes the open-source project only.

**Best for:** teams that want Grafana for alerts and dashboards and an LLM-aware trace UI with LLM-judge evaluation next to it. For setup, see our [LLM tracing guide](/llm-tracing) and the [MLflow observability overview](/genai/observability).

## 5. What if you build it yourself with the OpenTelemetry Collector and Grafana Tempo?

An OpenTelemetry Collector receives OTLP from your app and forwards it to Grafana Tempo, which Grafana reads through its built-in Tempo data source. This works with any instrumentation above, and it is the most flexible and the least LLM-aware route. You get traces, but you build every LLM view yourself.

**Evidence:** The [Tempo docs](https://grafana.com/docs/tempo/latest/) describe "an open-source, easy-to-use, and high-scale distributed tracing backend" that works with Jaeger, Zipkin, or OpenTelemetry. Grafana has "native support using the built-in Tempo data source," and a query language called TraceQL. The [Tempo repository](https://github.com/grafana/tempo) says the core project is distributed under AGPL-3.0, which matters if your legal team reviews copyleft licenses. The OpenTelemetry [Collector docs](https://opentelemetry.io/docs/collector/configuration/) say you can use the same receiver, processor, or exporter in more than one pipeline, which is how one trace stream reaches several backends.

**Where it falls short:** Tempo knows spans, not prompts, tokens, or evaluation scores. If you need cost per model or a prompt and response viewer, you add attributes and build panels. For most teams the first four options reach that point faster.

**Best for:** platform teams that already run Tempo and want LLM traces in the same place as the rest of their services. Our [OpenTelemetry for LLM apps](/articles/otel-for-llm) walkthrough covers the Collector side.

## What about Langfuse and Arize Phoenix?

We found no Grafana integration in the Langfuse and Arize Phoenix READMEs, Langfuse's OpenTelemetry docs, or the Phoenix docs landing page. Both accept OpenTelemetry traces, so you could fan data out through a Collector, but that is a configuration you build, not a documented integration. Langfuse's [README](https://github.com/langfuse/langfuse) says it is "built on OpenTelemetry," and Phoenix's [README](https://github.com/Arize-ai/phoenix) describes OpenTelemetry-based instrumentation.

If Grafana is your main view, start with the five above. If a dedicated LLM UI matters more, see our comparisons of [Langfuse](/langfuse-alternative) and [Arize Phoenix](/arize-phoenix-alternative).

## Which one should you pick?

Start from the constraint you cannot change.

- **You already pay for Grafana Cloud and want dashboards fast:** Grafana Cloud AI Observability.
- **You want an Apache-2.0 SDK and the choice to self-host a backend:** OpenLIT.
- **You want instrumentation only and will choose the backend later:** OpenLLMetry.
- **You want Grafana for dashboards and an LLM trace UI with evaluation beside it:** MLflow, with dual export.
- **You already run Tempo and want LLM traces there:** a Collector and Tempo, and accept that the views are yours to build.

Our own view, with the obvious bias: if all you need is latency charts and alerts, stop at Tempo or Grafana Cloud AI Observability. A dedicated LLM trace UI earns its place once people start reviewing prompts and running evaluations. You can combine them. The tools in this list all use OpenTelemetry, so one app can send a copy of its traces to Grafana and another to a dedicated LLM backend. The cost is two systems to keep consistent, so decide up front which one is the system of record.

## FAQ

### Does Grafana have LLM observability?

Yes. Grafana Cloud AI Observability monitors LLMs, vector databases, GPUs, and MCP servers, and its setup uses the OpenLIT integration. Grafana's docs describe dashboards for performance, cost, tokens, and usage. It runs on Grafana Cloud.

### Can I send LLM traces to Grafana Tempo?

Yes. Tempo accepts OpenTelemetry traces, and Grafana reads them through its Tempo data source. Any OpenTelemetry-instrumented LLM app can send traces there through an OTLP exporter or a Collector.

### Can MLflow send traces to Grafana?

MLflow can export traces over OTLP, and its export docs list Grafana among the supported platforms. Setting `MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT=true` keeps the traces in MLflow too. The MLflow docs give no Grafana-specific steps, so use Grafana's OTLP guide for the endpoint and token.

### Is OpenLIT open source?

Yes. OpenLIT's repository states it is licensed under the Apache License 2.0. You can self-host it with Docker or Kubernetes.

### Do I need Grafana Cloud to use these tools?

No, except for Grafana Cloud AI Observability itself. OpenLIT, OpenLLMetry, and MLflow emit OTLP, which can go to a Collector and a Tempo or Prometheus you run yourself.

### Which tool is best for LLM observability with Grafana?

There is no single best. Choose Grafana Cloud AI Observability for the fastest dashboards, OpenLIT or OpenLLMetry for open-source instrumentation, and MLflow if you also want a trace UI and evaluation beside Grafana.

## Next step

If you want to try the MLflow route, start with a local server using `mlflow server --port 5000`, enable tracing for your agent, then add the export variables above and compare the two views. Our [LLM tracing guide](/llm-tracing) covers the setup, and [OpenTelemetry for LLM apps](/articles/otel-for-llm) covers the Collector.
