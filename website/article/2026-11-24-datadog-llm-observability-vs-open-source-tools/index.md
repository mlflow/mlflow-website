---
title: "Datadog LLM Observability vs Open-Source Tools: Langfuse, Phoenix, MLflow"
description: "Datadog LLM Observability compared with Langfuse, Arize Phoenix, and MLflow: licenses, self-hosting, OpenTelemetry, pricing models, and which fits which team."
slug: datadog-llm-observability-vs-open-source-tools
tags:
  [
    Datadog LLM Observability,
    Datadog vs Langfuse,
    open source LLM observability,
    LLM observability tools,
    OpenTelemetry LLM tracing,
    Arize Phoenix,
    self-hosted LLM observability,
  ]
date: 2026-11-24
---

If your company already pays for Datadog, its LLM observability product is the lowest-friction way to see agent traces next to your APM and infrastructure data. If you need to run the backend yourself, keep traces for months, or avoid a per-span bill, the open-source options are Langfuse, Arize Phoenix, and MLflow. The choice mostly comes down to who hosts your trace data, how long you keep it, and how the bill scales.

This page compares Datadog with the three open-source tools using each vendor's own pricing pages, documentation, and repositories, all checked on 2026-11-24.

Updated 2026-11-24.

Disclosure: this page was written by MLflow maintainers, and MLflow is one of the four tools. This page covers the open-source project only. We kept the criteria the same for every tool and say where MLflow falls short.

## TL;DR

Short version for each tool:

- **Datadog:** a proprietary SaaS product, billed per LLM span, with 15 days of trace retention by default. It is the strongest fit when your team already lives in Datadog.
- **Langfuse:** a tracing-first product. The repository is MIT licensed, except the `ee` folders, and it can be self-hosted.
- **Arize Phoenix:** an open-source, local-first tracing and evaluation tool under the Elastic License 2.0. Arize's own pricing page lists a managed option, Arize AX, separately.
- **MLflow:** Apache-2.0, hosted by the Linux Foundation, and the one that also covers experiment tracking and a model registry. It is a broader project than a tracing-only tool.
- All four accept OpenTelemetry traces in some form, so you can try more than one on the same instrumentation.

## Why compare Datadog with open-source tools at all?

A team that already runs Datadog for APM and logs will ask the same question in a planning meeting: do we add LLM traces to the tool we have, or run something purpose-built? Both answers are reasonable. The deciding factor is rarely features.

The question is also fairly new. Datadog's own [announcement](https://www.datadoghq.com/blog/llm-otel-semantic-convention) of native OpenTelemetry GenAI support is dated December 1, 2025, and says that before it, Datadog's LLM data needed its SDK or manually annotated spans sent to an HTTP API. Since then, an application instrumented once with OpenTelemetry can send the same spans to Datadog or to an open-source backend.

A note on naming. Datadog's pricing page now lists the product as Agent Observability, while its docs and blog URLs still use `llm_observability` and `llm-otel`, while the page titles now say Agent Observability. We use "Datadog" for the product on both names.

## What should you compare?

We compared the four tools on six criteria:

1. **License and governance.** Can you read, run, and fork the code, and who controls it?
2. **Where the data lives.** Can traces stay in infrastructure you control?
3. **OpenTelemetry.** Can you send standard OTLP traces without a vendor SDK?
4. **Pricing model.** What do you pay for as agent traffic grows?
5. **Retention.** How long can you look back on each tier?
6. **Operations.** What do you have to run and upgrade yourself?

We left out raw speed and UI polish. We did not find a published benchmark from any of the four for either, and a screenshot is not a benchmark.

## How do Datadog, Langfuse, Phoenix, and MLflow compare?

|                  | Datadog                                                                        | Langfuse                                      | Arize Phoenix                                                      | MLflow                                                                      |
| ---------------- | ------------------------------------------------------------------------------ | --------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| License          | Proprietary SaaS                                                               | MIT, except `ee` folders                      | Elastic License 2.0 (source-available)                             | Apache-2.0                                                                  |
| Self-hosting     | Not described in Datadog's docs or pricing page                                | Docker Compose, Helm, and Terraform templates | Docker, Kubernetes, and one-click templates                        | `mlflow server` plus a SQL backend store and artifact storage               |
| OpenTelemetry    | OTLP ingest over `http/protobuf` for GenAI conventions 1.37+ and OpenInference | Built on OpenTelemetry, OTLP endpoint         | Built on OpenTelemetry, with OpenInference instrumentation         | OTLP/HTTP ingest on the tracking server (3.6.0 and later), plus OTLP export |
| Free tier        | Free plan: up to 40K LLM spans, 15-day retention                               | Hobby: $0, 50k units, 30 days, 2 users        | Phoenix self-hosted: free. Arize AX Free: 25k trace spans, 15 days | Free to self-host, no seat limit                                            |
| First paid tier  | Pro: $160 per month annually, first 100K LLM spans                             | Core: $29 per month, 100k units, 90 days      | Arize AX Pro: $50 per month, 50k trace spans, 30 days              | Not applicable                                                              |
| Bill scales with | LLM spans, plus retention                                                      | Units (traces, observations, scores)          | Trace spans and ingestion on AX; your infrastructure on Phoenix    | Your own infrastructure                                                     |

Prices are list prices on each vendor's pricing page on 2026-11-24 and change often, so check the linked page before you decide. A Datadog LLM span, a Langfuse unit, and an Arize trace span are different measures, so price your own volume on each. Do not compare the three numbers directly.

## 1. Datadog: best when your team already runs on Datadog

Datadog's [LLM Observability docs](https://docs.datadoghq.com/llm_observability/) describe tracing of "every step of your LLM application chains and calls," evaluations of "quality, privacy, and safety," and the ability to "automatically scan and redact any sensitive data in your AI applications." The same page describes Patterns, which clusters production traffic by topic, and Insights, which flags recurring cost and reliability problems.

**Why teams pick it:** LLM traces sit beside the APM, logs, and infrastructure data you already have. One alert, one on-call rotation, and one access model cover the whole stack. If a slow agent run traces back to a saturated database, you follow one link instead of switching tools.

**OpenTelemetry:** Datadog [ingests OpenTelemetry traces over OTLP](https://docs.datadoghq.com/llm_observability/instrumentation/otel_instrumentation/) without its SDK. Your spans must follow the OpenTelemetry GenAI conventions (1.37 or later) or the supported OpenInference conventions, and you must enable conversion with the `dd_llmobs_enabled=true` resource attribute or the `DD-LLMObs-Enabled: true` header. The docs also list limits: spans without GenAI or OpenInference attributes are dropped, and "there may be a 3-5 minute delay" before traces appear.

**Pricing:** The [Datadog list pricing page](https://www.datadoghq.com/pricing/list/) bills per 10K LLM spans per month. Annual list prices are $3.50 with 15-day trace retention, $5 with 30 days, $6.50 with 60 days, and $7.50 with 90 days. The list page shows $160 per month annually for the first 100K LLM spans. The [product page](https://www.datadoghq.com/products/ai/agent-observability/) names that plan Pro and says the free plan covers up to 40K LLM spans a month. The same page defines an LLM span as "a single call to an LLM provider" and says tool, workflow, agent, embedding, and retrieval spans are free.

A worked example, based on our reading of those lists: 500K LLM spans a month on the annual plan with 15-day retention is $160 for the first 100K plus 40 blocks of 10K at $3.50, so $300 a month. The example ignores the free allowance and applies only to the paid tier. Longer retention raises the per-span rate. Confirm the arithmetic with Datadog before you budget.

**Where it falls short:** Datadog is proprietary SaaS. Its docs and pricing page do not describe self-hosting, and Datadog's [docs](https://docs.datadoghq.com/llm_observability/instrumentation/otel_instrumentation/) show a notice that the product is not supported on its `ddog-gov` sites. Retention beyond 15 days costs more per span. Langfuse's own [comparison page](https://langfuse.com/compare/datadog) makes similar points, but it is a competitor's page, so treat it as a claim and check Datadog's docs.

**Best for:** teams already on Datadog who want LLM traces in the same place and are comfortable with a vendor-hosted, per-span bill.

## 2. Langfuse: best for a tracing-first product you can self-host

Langfuse is an LLM observability platform whose [repository](https://github.com/langfuse/langfuse) is "MIT licensed, except for the `ee` folders." The README says Langfuse is "built on OpenTelemetry," so any OpenTelemetry-instrumented application, framework, or collector can send it traces.

**Evidence:** You can deploy it with Docker Compose on one machine, or with Helm and Terraform templates for AWS, Azure, and GCP. The README calls Helm on Kubernetes the preferred production deployment. Its [self-hosting docs](https://langfuse.com/self-hosting) cover the architecture and configuration options.

**Pricing:** [Langfuse Cloud](https://langfuse.com/pricing) charges per unit, where a unit is a trace, an observation, or a score. Hobby is free with 50k units and 30 days of data access. Core is $29 per month with 100k units and 90 days. Pro is $199 per month with 3 years of access. Overage is $8.00 per 100k units up to 1M units, with lower rates above that.

**Where it falls short:** The `ee` folders are not MIT, so check which enterprise features you need before you assume the whole product is permissive. A self-hosted Langfuse is more to operate than a single service, and you own upgrades and backups. Langfuse covers tracing, evaluation, and prompt management, but not APM, classical experiment tracking, or a model registry.

**Best for:** teams that want a dedicated tracing product, a permissive core license, and the option to run it themselves. For a longer one-to-one view, see our [Langfuse alternative](/langfuse-alternative) page.

## 3. Arize Phoenix: best for local-first debugging and notebooks

Arize's [pricing page](https://arize.com/pricing/) describes Phoenix as an open-source, local-first platform for tracing, evaluation, experimentation, and prompt iteration. Its [repository](https://github.com/Arize-ai/phoenix) says it is built on OpenTelemetry and that auto-instrumentation comes from the separate OpenInference project. The same instrumentation works with Phoenix and with Arize's managed platform, Arize AX.

**License:** The repository states that Phoenix "is licensed under the terms of the Elastic License 2.0 (ELv2)." ELv2 is source-available, not an OSI-approved open-source license, and it limits offering the software as a managed service. If your legal team needs an OSI license, that matters.

**Self-hosting:** The repository lists Docker, Docker Compose, and Kubernetes with Helm. Arize's [self-hosting docs](https://arize.com/docs/phoenix/self-hosting) add one-click templates for Railway, Render, Google Cloud, Azure, and AWS CloudFormation, and say Phoenix is "free to self-host with no feature limitations."

**Pricing:** Phoenix itself has no price. The managed Arize AX tiers are on the [Arize pricing page](https://arize.com/pricing/): Free is $0 with 25k trace spans a month, 1 GB of ingestion, and 15-day retention, and Pro is $50 per month with 50k trace spans, 10 GB, and 30-day retention.

**Where it falls short:** The license is the main one. Phoenix is also built around tracing, evaluation, and experimentation, so teams that want a model registry or experiment tracking for classical ML need another tool.

**Best for:** teams that want to start tracing on a laptop and later decide whether to self-host or move to a managed tier. For a longer comparison, see our [Arize Phoenix alternative](/arize-phoenix-alternative) page.

## 4. MLflow: best for tracing and evaluation in one Apache-2.0 server

MLflow is an Apache-2.0 open-source platform for agents, LLMs, and ML models, hosted by the Linux Foundation. MLflow Tracing captures LLM and agent calls as traces, and its [tracing docs](https://mlflow.org/docs/latest/genai/tracing/) say it is "fully compatible with OpenTelemetry and natively supports GenAI Semantic Conventions for export and ingestion."

**Evidence:** A running server plus a few lines of code traces an OpenAI app:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")  # server started with `mlflow server --port 5000`
mlflow.set_experiment("agent-observability")
mlflow.openai.autolog()
```

The tracking server also accepts OTLP traces. The [ingest docs](https://mlflow.org/docs/latest/genai/tracing/opentelemetry/ingest/) say this needs MLflow 3.6.0 or later, an endpoint at `/v1/traces`, the experiment ID in the `x-mlflow-experiment-id` header, and a SQL-based backend store. Only OTLP over HTTP is supported, not gRPC.

**Pairing it with Datadog:** MLflow can also export traces to another OpenTelemetry backend. The [export docs](https://mlflow.org/docs/latest/genai/tracing/opentelemetry/export/) use the standard `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` and `OTEL_EXPORTER_OTLP_TRACES_HEADERS` variables. They also say that `MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT=true` sends traces to both MLflow and the other backend, and that `MLFLOW_ENABLE_OTEL_GENAI_SEMCONV=true` emits `gen_ai.*` attributes. By default, exported spans use MLflow's own `mlflow.*` attributes.

That combination is how you would keep a full-retention copy of traces in MLflow while also sending them to Datadog. Datadog needs GenAI conventions 1.37 or later and its own header and resource attribute, so check that the attributes MLflow emits meet that bar. We have not tested that path end to end, and the docs for both sides give no Datadog-specific example.

**Where it falls short:** MLflow is a broader project than a tracing-only tool, so GenAI-only products can move faster on a narrow feature. Self-hosting means you own upgrades, storage, and access control, and fine-grained permissions for shared resources only arrived in [3.17.0](https://github.com/mlflow/mlflow/releases/tag/v3.17.0) (2026-10-07). MLflow does not replace an APM, and we have not verified that it has a built-in equivalent to Datadog's automatic sensitive-data scanning. Hosted MLflow services exist, but this entry describes the open-source project only.

**Best for:** teams that want one self-hosted Apache-2.0 server for traces, evaluation, and the model lifecycle. For setup details, see our [LLM tracing guide](/llm-tracing) and the [MLflow observability overview](/genai/observability).

## Which tool fits which team?

Three scenarios show how the criteria play out.

- **A platform team that already runs Datadog for everything.** An on-call engineer gets paged on a latency spike and wants the LLM call in the same trace view as the database. Datadog does that without a new system. Check the retention first: 15 days by default is short if you debug regressions that surface weeks later.
- **A regulated company whose traces contain customer prompts.** Traces must stay in your network, and the license may need to be OSI-approved. Datadog does not describe self-hosting, and Phoenix is under ELv2. Langfuse (MIT core) and MLflow (Apache-2.0) are the options you can run yourself. Each needs a database and storage you operate.
- **An ML team that already tracks experiments.** If your models already log to MLflow, agent traces land beside the runs and models they relate to, and LLM-judge evaluation uses the same server. A separate tracing product means a second system and a second place to look.

## Can you use Datadog and an open-source tool together?

Yes, and for many teams it is the practical answer. Because every tool in this comparison speaks OpenTelemetry, you instrument once and choose where the spans go. A collector or a dual-export setting can send a copy to Datadog for alerting and a copy to an open-source backend for long retention and evaluation.

The cost is two systems to keep consistent. Decide up front which one is the system of record, or you will argue later about whose numbers are right.

## Which one should you pick?

Start from the constraint you cannot change.

- **Your team already runs Datadog and wants one pane of alerts:** Datadog. Budget for spans, and check retention against how far back you debug.
- **Traces must stay in your network, and the license must be open source:** MLflow or Langfuse. Choose MLflow if you also want evaluation and a model registry in the same server. Choose Langfuse if you want a tracing-first product.
- **You want to start on a laptop and decide on hosting later:** Phoenix, after your legal team has read ELv2.
- **You want long retention at a predictable cost:** a self-hosted open-source backend, with the operations work that implies.

If you are choosing today, send the same agent's traces to two of them for a week. Real volume and retention needs show up in a bill faster than a feature list does.

## FAQ

### Is Datadog LLM Observability open source?

No. Datadog's LLM Observability, now listed as Agent Observability on its pricing page, is a proprietary SaaS product. Its docs and pricing page do not describe self-hosting the backend.

### Can I send OpenTelemetry traces to Datadog LLM Observability?

Yes. Datadog accepts OTLP traces over `http/protobuf` if your spans follow the OpenTelemetry GenAI conventions 1.37 or later or the supported OpenInference conventions. You must also turn on conversion with `dd_llmobs_enabled=true` or the `DD-LLMObs-Enabled: true` header.

### How much does Datadog LLM Observability cost compared with Langfuse?

They bill on different units, so no single number is fair. Datadog lists $160 per month annually for the first 100K LLM spans, then $3.50 per 10K spans with 15-day retention. Langfuse Core is $29 per month with 100k units and 90 days, where a unit is a trace, an observation, or a score. Price your own volume on both.

### Is Arize Phoenix open source?

Phoenix is source-available under the Elastic License 2.0, which is not an OSI-approved open-source license. The code is public and free to self-host. The license limits offering it as a managed service.

### Can MLflow send traces to Datadog?

MLflow can export traces over OTLP to another OpenTelemetry backend, and a dual-export setting keeps them in MLflow too. Datadog requires its own header and a GenAI-conventions-compliant span format, and we have not tested that combination end to end.

### Which is best for LLM observability?

There is no single best. For an existing Datadog estate, look at Datadog. For an open-source license and self-hosting, look at MLflow or Langfuse. For local-first debugging, look at Phoenix.

## Next step

If you want to try MLflow, start the server with `mlflow server --port 5000`, add `mlflow.openai.autolog()` to your agent, and compare the trace view with the tool you use today. Our [LLM tracing guide](/llm-tracing) walks through the setup, and [OpenTelemetry for LLM apps](/articles/otel-for-llm) explains how the export side works. For the wider field, see the [top agent observability tools](/top-5-agent-observability-tools).
