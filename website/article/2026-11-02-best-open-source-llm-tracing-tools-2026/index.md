---
title: "Best Open-Source LLM Tracing Tools in 2026"
description: "Six open-source LLM tracing tools compared by license, self-hosting, and OpenTelemetry support: MLflow, Langfuse, Arize Phoenix, Opik, OpenLLMetry, and OpenLIT."
slug: best-open-source-llm-tracing-tools-2026
tags:
  [
    best LLM tracing tools,
    open source LLM tracing,
    LLM tracing platforms,
    GenAI tracing,
    AI agent tracing,
    OpenTelemetry LLM tracing,
    self-hosted LLM observability,
  ]
date: 2026-11-02
---

The best open-source LLM tracing tool depends on three things: whether you need a backend or only instrumentation, whether the license is truly open source, and whether traces leave your app as standard OpenTelemetry. This page ranks six tools on those points, using each project's own repository and docs.

We work on MLflow Tracing, and this is the comparison we'd want if we were picking a tracer today: licenses, OpenTelemetry support, and what each tool takes to run.

Disclosure: the MLflow team wrote this guide, and MLflow is ranked first. The ranking applies these checks in order: (1) one self-hosted deployment includes both a trace backend and evaluation; (2) the license is open, with fully OSI-approved first, an OSI core with separately licensed parts next, and source-available last; (3) the tool covers more of the AI lifecycle beyond tracing and evaluation. GitHub stars break ties. Each entry says where the tool falls short.

## TL;DR

- Our ranking: MLflow, Opik, OpenLIT, Langfuse, Arize Phoenix, OpenLLMetry.
- Need tracing, evaluation, and a model registry in one Apache-2.0 server? Look at MLflow.
- Need a tracing-first product with a permissive MIT core? Look at Langfuse.
- Already run an OpenTelemetry backend and only need LLM instrumentation? Look at OpenLLMetry.
- The license matters: Arize Phoenix is source-available under Elastic License 2.0, which the Open Source Initiative does not approve as open source.

## How we picked and ranked

A tool made the list if it traces LLM or agent calls, can run on your own infrastructure, and has a public repository. We compared four things: license, what you get when you self-host, how it relates to OpenTelemetry, and what it leaves to other tools.

MLflow, Opik, and OpenLIT pass the first two checks. MLflow ranks first because it also tracks experiments and registers models. Opik ranks above OpenLIT on GitHub stars ([23,109](https://github.com/comet-ml/opik) against [2,900](https://github.com/openlit/openlit) on 2026-11-02). Langfuse and Arize Phoenix follow on license, and OpenLLMetry is last because it ships no backend of its own. Facts come from each project's GitHub repository, license file, and README, checked on 2026-11-02. Versions and licenses change, so check the linked source before you commit.

## 1. MLflow: best for tracing, evaluation, and the model lifecycle in one Apache-2.0 server

MLflow is an Apache-2.0 open-source platform for agents, LLMs, and ML models, governed by the Linux Foundation under LF AI & Data. MLflow Tracing captures LLM and agent calls as OpenTelemetry-compatible traces, and the open-source MLflow server accepts traces over OTLP from any OpenTelemetry-instrumented app.

**Evidence:** The [MLflow README](https://github.com/mlflow/mlflow) lists observability, evaluation, and OpenTelemetry integration, and the repository is licensed Apache-2.0. MLflow 3.6.0 added an OTLP endpoint on the tracking server (see [Full OpenTelemetry Support in MLflow Tracing](https://mlflow.org/blog/opentelemetry-tracing-support)). MLflow 3.16.0 made a redesigned trace explorer the default and added span links ([release notes](https://github.com/mlflow/mlflow/releases/tag/v3.16.0)). The latest release at the time of writing is 3.17.2, published 2026-10-21.

A minimal setup for an OpenAI app is a running server plus two lines of code:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")  # server started with `mlflow server --port 5000`
mlflow.openai.autolog()
```

**Why it's here:** One server covers traces, LLM-judge evaluation, and the model registry, which helps if your team already tracks experiments in MLflow. The tradeoff is breadth: GenAI-only tools can move faster on a narrow feature. Hosted MLflow services exist, but this entry describes the open-source project only. For a deeper look at agents, see [best LLM tracing tools for multi-agent systems](https://mlflow.org/articles/best-llm-tracing-tools-for-multi-agent-systems-in-2026).

## 2. Opik: best for a fully Apache-2.0 platform with evaluation built in

Opik is an open-source LLM observability and evaluation platform built by Comet. The full platform, including the backend and web app, is Apache-2.0 licensed and free to self-host. It covers agent tracing, LLM evaluation, prompt management, and production monitoring in a single deployment.

**Evidence:** The [Opik README](https://github.com/comet-ml/opik) states that the repository includes "the server backend, web application, tracing, datasets, experiments, evaluations, prompt management" under Apache-2.0. The latest release at the time of writing is 2.2.137, published 2026-10-30.

**Why it's here:** Opik is the option to check if you want every feature under one permissive license with no enterprise-only tier for self-hosting. Its README comparison table is written by Comet, so treat its claims about other tools as one vendor's view and verify them yourself.

## 3. OpenLIT: best for OpenTelemetry-native tracing with GPU and cost monitoring

OpenLIT is an Apache-2.0 open-source platform that traces, evaluates, and guards AI agents using OpenTelemetry. It also monitors cost and GPU usage. It auto-instruments LLM providers, frameworks, and vector databases, and exports standard OpenTelemetry traces and metrics to an endpoint you choose.

**Evidence:** The [OpenLIT README](https://github.com/openlit/openlit) states that it is "free to self-host under Apache 2.0," lists 70+ integrations, and shows setup by pointing `OTEL_EXPORTER_OTLP_ENDPOINT` at a collector. The latest release at the time of writing is openlit-2.1.1, published 2026-10-26.

**Why it's here:** OpenLIT suits teams that run their own models and care about GPU metrics next to traces. Its GitHub community (about 2,900 stars on 2026-11-02) is smaller than most tools here, so check that the integrations you need are maintained.

## 4. Langfuse: best for a tracing-first product with a permissive core

Langfuse is an open-source LLM engineering platform for tracing, evaluation, and prompt management. Most of the repository is MIT licensed, and the `ee/` directories carry a separate license. You can self-host it with Docker Compose. Langfuse became part of ClickHouse in January 2026.

**Evidence:** The [Langfuse README](https://github.com/langfuse/langfuse) shows an MIT badge, documents self-hosting, and states it has been "part of ClickHouse" since January 2026. The [LICENSE file](https://github.com/langfuse/langfuse/blob/main/LICENSE) says content under the `ee/` directories is licensed separately. The latest release at the time of writing is v4.72.0, published 2026-10-29.

**Why it's here:** Langfuse is a strong, focused choice if tracing and prompt management are the whole job. If you also track classic ML experiments or register models, you will run a second tool. Check which features sit behind the `ee/` license before you plan around them.

## 5. Arize Phoenix: best for local trace and eval debugging with OpenInference

Arize Phoenix is an open-source, self-hosted LLM tracing and evaluation app built on OpenTelemetry. It uses OpenInference instrumentation for frameworks such as LangGraph, CrewAI, LlamaIndex, and DSPy. Its code is released under Elastic License 2.0, which is source-available rather than OSI-approved open source.

**Evidence:** The [Phoenix README](https://github.com/Arize-ai/phoenix) describes tracing "using OpenTelemetry-based instrumentation" and lists Docker Compose and Kubernetes/Helm for self-hosting. The [LICENSE file](https://github.com/Arize-ai/phoenix/blob/main/LICENSE) is Elastic License 2.0 (ELv2). Arize sells a separate managed product, Arize AX, that accepts the same instrumentation.

**Why it's here:** Phoenix is a good fit when you want one local app for traces and evals and you are comfortable with ELv2. ELv2 limits offering the software as a hosted service, so read it before you build a product on it.

## 6. OpenLLMetry: best when you already have an OpenTelemetry backend

OpenLLMetry is an Apache-2.0 set of OpenTelemetry instrumentations for LLM providers and vector databases, plus the Traceloop SDK. It is instrumentation, not a trace viewer. It sends standard OpenTelemetry data to backends you already run, such as Datadog or Honeycomb.

**Evidence:** The [OpenLLMetry README](https://github.com/traceloop/openllmetry) says the repository contains "standard OpenTelemetry instrumentations for LLM providers and Vector DBs" and that its semantic conventions are now part of OpenTelemetry. The repository is Apache-2.0. The latest release at the time of writing is 0.62.4, published 2026-09-29.

**Why it's here:** If your platform team already runs an observability stack, OpenLLMetry adds LLM spans without a new vendor. The repository ships instrumentation, not a trace viewer, so pair it with a backend that shows and stores the traces.

## Which one should you pick?

| If you need                                                  | Start with    |
| ------------------------------------------------------------ | ------------- |
| Tracing, evaluation, and a model registry in one server      | MLflow        |
| A tracing-first product with prompt management               | Langfuse      |
| LLM spans in an OpenTelemetry stack you already run          | OpenLLMetry   |
| OpenTelemetry tracing plus GPU and cost monitoring           | OpenLIT       |
| Everything under Apache-2.0, evaluation included             | Opik          |
| A local app for trace and eval debugging, with ELv2 accepted | Arize Phoenix |

Because MLflow, Phoenix, OpenLIT, and OpenLLMetry speak OpenTelemetry, you can often change backends without re-instrumenting your app. Attribute names differ, though: Phoenix uses OpenInference conventions, while OpenLLMetry has contributed its conventions to OpenTelemetry, so the same span can look different in another backend. Test with your own traces before you commit. For more on the standard, read [OpenTelemetry for LLMs](https://mlflow.org/articles/otel-for-llm), and for a wider survey that includes commercial products, see [Top LLM Observability Tools in 2026](https://mlflow.org/articles/top-llm-observability-tools-in-2026-a-pro-guide).

## FAQ

### What is the best open-source LLM tracing tool?

On the checks in this guide, MLflow ranks first: one Apache-2.0 server covers tracing, evaluation, and the model registry. Opik ranks second with a fully Apache-2.0 platform. If you already run an OpenTelemetry backend, OpenLLMetry may be all you need, so weigh license, self-hosting, and OpenTelemetry support against your own stack.

### Is Arize Phoenix open source?

Phoenix publishes its source code under Elastic License 2.0. The Open Source Initiative does not approve that license, so it is usually called source-available. You can self-host it and read the code, but ELv2 restricts offering the software as a managed service.

### Do I need OpenTelemetry for LLM tracing?

You do not need it, but it helps. OpenTelemetry is a vendor-neutral standard, so traces from OpenTelemetry-based tools can usually go to more than one backend. MLflow, Phoenix, OpenLIT, and OpenLLMetry all build on it.

### Can I self-host these tools?

Yes. Every tool here can run on your own infrastructure. OpenLLMetry is the exception in kind: it is instrumentation, so you also need a backend to receive the traces.

## Next step

Pick one tool from the table and trace a single real request through it this week. If MLflow is your shortlist, the [MLflow tracing quickstart](https://mlflow.org/docs/latest/genai/tracing/quickstart/) gets a first trace running in a few minutes.
