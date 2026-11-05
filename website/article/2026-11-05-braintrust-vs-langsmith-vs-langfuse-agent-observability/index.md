---
title: "Braintrust vs LangSmith vs Langfuse (and MLflow) for Agent Observability"
description: "Braintrust, LangSmith, Langfuse, and MLflow compared for agent observability: licenses, self-hosting, OpenTelemetry support, pricing models, and which fits which team."
slug: braintrust-vs-langsmith-vs-langfuse-agent-observability
tags:
  [
    Braintrust vs LangSmith vs Langfuse,
    agent observability,
    LLM observability tools,
    AI agent tracing,
    open source LLM observability,
    OpenTelemetry LLM tracing,
    self-hosted LLM observability,
  ]
date: 2026-11-05
---

For agent observability, pick Langfuse if you want a tracing-first product with an MIT-licensed core, LangSmith if you build on LangChain or LangGraph and want a managed service, Braintrust if you want a hosted evaluation workflow, and MLflow if you want Apache-2.0 tracing and evaluation in one server you run yourself. The right answer depends on three things: who hosts your trace data, how the license reads, and how the bill scales.

This page compares the four on those three points, using each vendor's own pricing page, documentation, and repository, all checked on 2026-11-05.

Disclosure: this page was written by MLflow maintainers, and MLflow is one of the four tools. This page covers the open-source project only. We kept the criteria the same for every tool and say where MLflow falls short.

## TL;DR

Short version for each tool:

- **Langfuse:** the only one of the three dedicated products with a stated open-source license for its platform, in our reading. The repository is MIT licensed, except the `ee` folders.
- **LangSmith:** the lowest-friction path if your agents already use LangChain or LangGraph. Self-hosting is an Enterprise add-on.
- **Braintrust:** priced on processed data and scores, with 14 or 30 days of retention on the self-serve plans. Hybrid self-hosting is closed to new customers from 2026-11-04, so verify the current options with Braintrust.
- **MLflow:** Apache-2.0, hosted by the Linux Foundation, and the one that also covers experiment tracking and a model registry. Its tradeoff is breadth: it is a larger project than a tracing-only tool.
- All four accept or emit OpenTelemetry traces in some form. The details differ, and the table below spells them out.

## What does agent observability need from a tool?

An agent run is a tree: model calls, tool calls, retrieval steps, and sub-agents, often across several turns. Agent observability means capturing that tree as a trace, grouping traces into sessions, attaching cost and quality scores, and letting you search the result.

We compared the tools on six criteria:

1. **License and governance.** Can you read, run, and fork the code, and who controls it?
2. **Where the data lives.** Can traces stay in your own infrastructure?
3. **OpenTelemetry.** Can you send standard OTLP traces without a vendor SDK?
4. **Pricing model.** What do you pay for as agent traffic grows?
5. **Retention.** How long can you look back on each tier?
6. **Evaluation.** Can you score the traces you collected?

We left out raw speed and UI polish. Neither has a number any of the four vendors publishes, and a screenshot is not a benchmark.

## How do Braintrust, LangSmith, Langfuse, and MLflow compare?

|                           | Braintrust                                                                                | LangSmith                                             | Langfuse                                                              | MLflow                                                                                                  |
| ------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| License                   | Platform: hosted, no open-source license stated in its docs. SDKs: Apache-2.0             | SDKs are MIT; platform license not stated in its docs | MIT, except `ee` folders                                              | Apache-2.0                                                                                              |
| Self-hosting              | "Legacy" data plane you host; closed to new customers from 2026-11-04. BYOC is documented | Enterprise add-on, runs on your Kubernetes cluster    | Docker Compose (local and testing), Helm and Terraform for production | `mlflow server` plus a SQL backend store and artifact storage (SQLite and local files work for a trial) |
| OpenTelemetry             | OTLP ingest endpoint                                                                      | OTLP ingest endpoint                                  | Built on OpenTelemetry, OTLP endpoint                                 | OTLP ingest endpoint on the tracking server (3.6.0 and later)                                           |
| Free tier                 | Starter: $0, 1 GB processed data, 14 days                                                 | Developer: $0, 1 seat, 5k base traces, 14 days        | Hobby: $0, 50k units, 30 days, 2 users                                | Free to self-host, no seat limit                                                                        |
| First paid tier           | Pro: $249/month, 30 days                                                                  | Plus: $39 per seat/month, 10k base traces, 14 days    | Core: $29/month, 100k units, 90 days                                  | Not applicable                                                                                          |
| What the bill scales with | Processed GB and scores                                                                   | Seats and traces                                      | Units                                                                 | Your own infrastructure                                                                                 |

Prices are list prices on each vendor's pricing page on 2026-11-05. Higher tiers exist, such as Langfuse Pro at $199 a month, and prices change, so check the linked page before you decide. Units, base traces, and processed GB are different measures, so price your own volume on each.

## 1. Langfuse: best for a tracing-first product you can self-host

Langfuse is an LLM observability platform whose [repository](https://github.com/langfuse/langfuse) is "MIT licensed, except for the `ee` folders." The `ee` code falls under a separate license in `ee/LICENSE`, and its [self-hosting docs](https://langfuse.com/self-hosting) say some add-on features need a license key.

**Evidence:** Langfuse says it is [built on OpenTelemetry](https://github.com/langfuse/langfuse) and that any OpenTelemetry-instrumented app, framework, or collector can send it traces. You can deploy it with Docker Compose on one machine, or with Helm and Terraform templates for AWS, Azure, and GCP ([repository](https://github.com/langfuse/langfuse)). The self-hosted architecture depends on PostgreSQL, ClickHouse, Redis or Valkey, and S3-compatible blob storage ([self-hosting docs](https://langfuse.com/self-hosting)).

**Pricing:** [Langfuse Cloud](https://langfuse.com/pricing) charges per unit. Hobby is free with 50k units and 30 days of data access. Core is $29/month with 100k units and 90 days. Pro is $199/month with 3 years of access. Overage is $8 per 100k units, with lower rates at volume.

**Where it falls short:** A self-hosted Langfuse needs PostgreSQL, ClickHouse, Redis or Valkey, and blob storage, which is more to operate than a SQL database and an artifact store. The `ee` features are not MIT, so check which ones you need before you assume the whole product is permissive.

**Best for:** teams that want a dedicated tracing product, a permissive core license, and the option to run it themselves.

## 2. LangSmith: best for LangChain and LangGraph teams that want a managed service

LangSmith is LangChain's observability and evaluation platform. It works outside LangChain too: its [OpenTelemetry guide](https://docs.langchain.com/langsmith/trace-with-opentelemetry) says LangSmith "receives spans at its OTLP endpoint," and documents tracing non-LangChain agents with a standard OpenTelemetry client.

**Evidence:** The [LangSmith pricing page](https://www.langchain.com/pricing) lists Developer at $0 per seat with 5k base traces a month, Plus at $39 per seat with 10k base traces, and Enterprise at custom pricing. Base trace retention is 14 days on Developer and Plus, and extended 180-day retention is an upgrade at an extra fee. Seats are part of the bill, so cost grows with team size as well as trace volume.

**Self-hosting:** The pricing page lists self-hosting only under Enterprise. The [self-hosted docs](https://docs.langchain.com/langsmith/self-hosted) describe it as an Enterprise add-on that runs in your own Kubernetes cluster, backed by ClickHouse, PostgreSQL, and Redis or Valkey. The [Python](https://github.com/langchain-ai/langsmith-python) and [JavaScript](https://github.com/langchain-ai/langsmith-javascript) SDK repositories are MIT licensed, but we found no open-source license for the platform itself in LangSmith's docs.

**Where it falls short:** If data residency or a permissive license is a hard requirement, the path runs through an Enterprise contract. Per-seat pricing also adds up for wide teams where many people only need to read traces.

**Best for:** teams already on LangChain or LangGraph who want a managed service and are fine paying per seat.

## 3. Braintrust: best for a hosted workflow built around scores

Braintrust is a hosted platform for tracing and evaluating LLM apps. Its [JavaScript](https://github.com/braintrustdata/braintrust-sdk-javascript) and [Python](https://github.com/braintrustdata/braintrust-sdk-python) SDKs are Apache-2.0, but we found no open-source license for the platform in its docs.

Its [pricing page](https://www.braintrust.dev/pricing) shows how it meters usage: processed data in GB and scores, not seats. Starter is $0 with 1 GB of processed data, 10k scores, and 14 days of retention. Pro is $249/month with 5 GB, 50K scores, and 30 days, and retention extends at $0.50 per GB per month.

**OpenTelemetry:** Braintrust's [OpenTelemetry docs](https://docs.braintrust.dev/integrations/sdk-integrations/opentelemetry/send-traces-and-logs) describe an OTLP traces endpoint at `https://api.braintrust.dev/otel/v1/traces`. They describe sending data into Braintrust, not exporting it to another OpenTelemetry backend.

**Self-hosting:** Braintrust's [self-hosting guide](https://docs.braintrust.dev/admin/self-hosting) describes a hybrid model. You run the data plane, which stores traces, spans, and datasets, and Braintrust runs the control plane with the UI and authentication. The guide also says that "starting November 4, 2026, it isn't available to new customers," and points to Bring your own cloud (BYOC) for data residency.

**Where it falls short:** Hybrid self-hosting is now closed to new customers, and the docs point to BYOC instead, so ask Braintrust whether that fits your data rules. A 14 or 30 day window on the self-serve plans is short if you debug regressions that surface weeks later.

**Best for:** teams that want a hosted product with scoring at the center, including unlimited human review scores on Pro, and are comfortable with a vendor-managed control plane. For a longer one-to-one view, see our [Braintrust alternative](/braintrust-alternative) page.

## 4. MLflow: best for tracing and evaluation in one Apache-2.0 server

MLflow is an Apache-2.0 open-source platform for agents, LLMs, and ML models, hosted by the Linux Foundation. MLflow Tracing captures LLM and agent calls as traces, and its [tracing docs](https://mlflow.org/docs/latest/genai/tracing/) say it is "fully compatible with OpenTelemetry and natively supports GenAI Semantic Conventions for export and ingestion." The open-source tracking server accepts OTLP traces, which needs a SQL backend store and an experiment ID header ([ingest docs](https://mlflow.org/docs/latest/genai/tracing/opentelemetry/ingest/)).

**Evidence:** A running server plus a few lines of code traces an OpenAI app:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")  # server started with `mlflow server --port 5000`
mlflow.set_experiment("agent-observability")
mlflow.openai.autolog()
```

The same server groups traces into [sessions](https://mlflow.org/docs/latest/genai/tracing/track-users-sessions/), records [token usage and cost](https://mlflow.org/docs/latest/genai/tracing/token-usage-cost/), and runs LLM-judge evaluation on traces you already collected. The tracing docs describe OpenTelemetry compatibility for both export and ingestion, and [Full OpenTelemetry Support in MLflow Tracing](https://mlflow.org/blog/opentelemetry-tracing-support) covers the ingest side.

**Where it falls short:** MLflow is a broader project than a tracing-only tool, so GenAI-only products can move faster on a narrow feature. Self-hosting means you own upgrades, storage, and access control, and fine-grained permissions for shared resources only arrived in [3.17.0](https://github.com/mlflow/mlflow/releases/tag/v3.17.0) (2026-10-07). Hosted MLflow services exist, but this entry describes the open-source project only.

**Best for:** teams that want one self-hosted Apache-2.0 server for traces, evaluation, and the model lifecycle. For one-to-one comparisons, see [MLflow vs Langfuse](/langfuse-alternative) and [MLflow vs LangSmith](/langsmith-alternative).

## Which tool fits which team?

Three scenarios show how the criteria play out.

- **A five-person startup shipping a LangGraph agent.** LangSmith's Developer tier is free for one seat, but Plus is $39 per seat, so a five-person team pays $195 a month before trace overage. Langfuse Core is $29 a month with unlimited users, 100k included units, and 90 days of data access. Units and traces are different measures, and Langfuse overage is $8 per 100k units, so price your own volume on both. If the agent isn't tied to LangChain, Langfuse Core's cost does not grow with team size, only with units past the included 100k.
- **A platform team at a regulated company.** Traces contain customer prompts, so they must stay in your network. Braintrust points new customers to Bring your own cloud (BYOC), and LangSmith self-hosting is available on Enterprise. Both platforms are proprietary. MLflow (Apache-2.0) and Langfuse (MIT core) are the open-source options you can start with today. Each needs a database and storage you operate.
- **An ML team that already tracks experiments.** If your models already log to MLflow, agent traces land beside the runs and models they relate to, and LLM-judge evaluation uses the same server. A tracing-only product means a second system to run and a second place to look.

The $195 figure is five seats at LangSmith's listed $39 per seat, with no overage. Your own number depends on trace volume.

## Which one should you pick?

Start from the constraint you cannot change.

- **Traces must stay in your network, and the license must be open source:** MLflow or Langfuse. Choose MLflow if you also want evaluation and a model registry in the same server. Choose Langfuse if you want a tracing-first product and can run PostgreSQL, ClickHouse, Redis, and blob storage.
- **Your agents run on LangChain or LangGraph, and you want a managed service:** LangSmith. Budget for seats.
- **You want hosted evaluation with scores as the main unit:** Braintrust. Check retention against how far back you debug.
- **You already run an OpenTelemetry collector:** all four can take OTLP traces, so lean on whichever backend your team will actually operate.

If you are choosing today, run the same agent through two of them for a week. Trace volume and retention needs show up in a real bill faster than a feature list does.

## FAQ

### Is Langfuse really open source?

The Langfuse repository is MIT licensed except for its `ee` folders, which fall under a separate license in `ee/LICENSE`. The core tracing product is MIT. Enterprise add-ons in the `ee` folders are not.

### Can LangSmith be self-hosted?

Yes, on the Enterprise plan. LangSmith's docs describe self-hosting as an Enterprise add-on that runs in your own Kubernetes cluster. The Developer and Plus plans do not include it.

### Can I still self-host Braintrust?

Not if you are a new customer. Braintrust's docs say hybrid self-hosting is not available to new customers starting 2026-11-04, and they point to Bring your own cloud (BYOC) for data residency.

### Which of these works with OpenTelemetry?

All four accept or emit OpenTelemetry traces. Braintrust, LangSmith, and Langfuse document OTLP ingest endpoints, and MLflow is compatible with OpenTelemetry and its GenAI semantic conventions. Check each docs page for the exact endpoint and headers.

### Which is best for agent observability?

There is no single best. For an open-source license and self-hosting, look at MLflow or Langfuse. For a managed LangChain workflow, look at LangSmith. For hosted score-driven evaluation, look at Braintrust.

## Next step

If you want to try MLflow, start the server with `mlflow server --port 5000`, add `mlflow.openai.autolog()` to your agent, and compare the trace view with the tool you use today. Our [LLM tracing guide](/llm-tracing) walks through the setup, and [best LLM tracing tools for multi-agent systems](https://mlflow.org/articles/best-llm-tracing-tools-for-multi-agent-systems-in-2026) covers the multi-agent case.
