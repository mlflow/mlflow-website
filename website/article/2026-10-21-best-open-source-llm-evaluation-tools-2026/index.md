---
title: "Best Open-Source LLM Evaluation Tools in 2026: 6 Self-Hosted Tools Compared"
description: "Compare the best open-source LLM evaluation tools you can self-host in 2026: MLflow, Langfuse, Arize Phoenix, Opik, Promptfoo, and Ragas. See licenses, features, and what each takes to run."
slug: best-open-source-llm-evaluation-tools-2026
tags:
  [
    best open-source LLM evaluation tools,
    self-hosted LLM evaluation,
    open-source LLM evaluation 2026,
    LLM evaluation tools compared,
    RAG evaluation tools,
    MLflow evaluation,
  ]
date: 2026-10-21
---

**Last reviewed: October 21, 2026.** _This is Part 2 of a two-part series. [Part 1](/articles/best-llm-evaluation-platforms-2026) compares hosted and commercial LLM evaluation platforms._

An open-source LLM evaluation tool lets you test whether your LLM app or agent gives correct, safe, and useful answers, on your own infrastructure and without a vendor contract. The best open-source LLM evaluation tools in 2026 go beyond a metric library. They trace your app, score outputs with LLM judges and code, collect human feedback, and keep monitoring after you ship.

This guide compares six tools that you can self-host for free. For each one, it covers the license, the evaluation features, and what you need to run it.

**How we ranked them:** by how much of the evaluation loop each tool covers on its own, from tracing and judges to datasets, human review, production monitoring, and optimization. This guide is published by the MLflow team. Every competitor description links to that project's own documentation, and licenses were checked against each project's GitHub repository on the date above.

## Quick answer: what are the best open-source LLM evaluation tools in 2026?

The best open-source LLM evaluation tools in 2026 are:

1. **MLflow**: best self-hosted, end-to-end evaluation platform. It is free, Apache 2.0, and every feature is in the open-source release.
2. **Langfuse**: best for teams that want tracing, evaluation, and prompt management in one MIT-licensed platform.
3. **Arize Phoenix**: best for OpenTelemetry-based tracing and experiments, if a source-available license works for you.
4. **Opik**: best for Apache 2.0 tracing and evaluation with test suites and PyTest integration.
5. **Promptfoo**: best for red teaming and comparing prompts and models from the command line.
6. **Ragas**: best for RAG metrics and synthetic test data in a lightweight Python library.

## Table of contents

- [What to look for in an open-source LLM evaluation tool](#what-to-look-for-in-an-open-source-llm-evaluation-tool)
- [Open-source LLM evaluation tools comparison table](#open-source-llm-evaluation-tools-comparison-table)
- [1. MLflow](#1-mlflow)
- [2. Langfuse](#2-langfuse)
- [3. Arize Phoenix](#3-arize-phoenix)
- [4. Opik](#4-opik)
- [5. Promptfoo](#5-promptfoo)
- [6. Ragas](#6-ragas)
- [Self-hosting compared: what each tool takes to run](#self-hosting-compared-what-each-tool-takes-to-run)
- [Frameworks vs. platforms](#frameworks-vs-platforms)
- [How to choose an open-source LLM evaluation tool](#how-to-choose-an-open-source-llm-evaluation-tool)
- [FAQ](#faq)
- [Get started with MLflow evaluation](#get-started-with-mlflow-evaluation)

## What to look for in an open-source LLM evaluation tool

"Open source" covers a wide range of projects, from a single Python library to a full platform with a UI and a database. Use these criteria to compare them.

- **License.** Check the actual license file. Apache 2.0 and MIT are OSI-approved open-source licenses. Some projects are source-available instead, and some keep certain features under a separate commercial license.
- **Feature gating.** Check whether every feature you need is in the free release, or whether some require a paid license key.
- **LLM-as-a-judge and code scorers.** You need both. LLM judges score open-ended quality. Code scorers check exact rules, such as format or tool arguments.
- **Tracing.** The best test cases come from real traces. A tool that traces your app can turn a failure into a test case.
- **Human review.** Domain experts should be able to label traces and correct judges without writing code.
- **Production monitoring.** The same scorers should run on live traffic, not only in offline tests.
- **What it takes to self-host.** Count the services you have to run, back up, and upgrade.

## Open-source LLM evaluation tools comparison table

| Tool          | License                                | Maintained by                   | Type      | Best for                                |
| ------------- | -------------------------------------- | ------------------------------- | --------- | --------------------------------------- |
| MLflow        | Apache 2.0                             | Linux Foundation (LF AI & Data) | Platform  | Self-hosted, end-to-end evaluation      |
| Langfuse      | MIT core, commercial `ee` features     | ClickHouse                      | Platform  | Tracing, evals, and prompt management   |
| Arize Phoenix | Elastic License 2.0 (source-available) | Arize AI                        | Platform  | OpenTelemetry tracing and experiments   |
| Opik          | Apache 2.0                             | Comet                           | Platform  | Test suites and PyTest-based evaluation |
| Promptfoo     | MIT                                    | OpenAI                          | Framework | Red teaming and prompt/model comparison |
| Ragas         | Apache 2.0                             | Vibrant Labs                    | Framework | RAG metrics and synthetic test data     |

MLflow is the only tool on this list that is governed by a foundation rather than a single company, and it has no paid tier or license-gated features.

## 1. MLflow

MLflow is an Apache 2.0, Linux Foundation project that covers the whole evaluation loop in one free tool: OpenTelemetry tracing, LLM judges and code scorers, automatic issue detection, human review, production monitoring, and prompt and judge optimization. There is no paid tier, and every feature is in the open-source release.

- **Best for:** teams that want one self-hosted tool to trace, evaluate, find issues, and optimize.
- **Key fact:** 30M+ monthly downloads, 28K+ GitHub stars, and 900+ contributors.

### How do you self-host MLflow?

You can run the full MLflow platform on your laptop with two commands:

```bash
pip install mlflow
mlflow server --port 5000
```

This starts the server and UI with a SQLite database, which is the [default backend store](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/). For production, point MLflow at PostgreSQL, MySQL, or Microsoft SQL Server, and deploy it to Kubernetes with the [official Helm chart](https://github.com/mlflow/mlflow/tree/master/charts) in the MLflow repository. MLflow includes [built-in authentication and role-based access control](https://mlflow.org/docs/latest/self-hosting/security/role-based-access-control/), and you can add single sign-on through an [OIDC plugin or a reverse proxy](https://mlflow.org/docs/latest/self-hosting/security/sso/). See the [self-hosting guide](https://mlflow.org/docs/latest/self-hosting/) for the details.

### Automatic Issue Detection

[Automatic Issue Detection](https://mlflow.org/docs/latest/genai/eval-monitor/ai-insights/detect-issues/) uses an LLM to analyze the traces you already collect and finds quality and operational issues for you. You do not need to define metrics first, so it works even when you do not yet know what to measure. It shows you the failure patterns, and then you know which judges to build.

### Built-in optimizers for prompts and judges

- **Prompt optimization.** [`mlflow.genai.optimize_prompts()`](https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/) improves a prompt against your evaluation data with optimizers such as GEPA. It works with LangChain, LangGraph, OpenAI Agents, Pydantic AI, and your own code.
- **Judge alignment.** [SIMBA](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/alignment/) and [MemAlign](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/memalign/) tune an LLM judge with human feedback, so the judge agrees with your domain experts. MemAlign is experimental and shows improvement with a handful of labeled examples.

### Key evaluation features

- **LLM judges and code scorers.** Use [predefined judges](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined/), write [custom judges](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges/) in plain language, or write [code scorers](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/custom/) in Python.
- **Third-party metrics.** Run [metrics from DeepEval, Ragas, Arize Phoenix, TruLens, and Guardrails AI](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/third-party/) as MLflow scorers, through the same API as your own judges.
- **Human review.** [Review queues](https://mlflow.org/docs/latest/genai/assessments/review-queues/) (experimental) assign traces to reviewers and collect structured feedback and ground-truth labels in the UI.
- **Production monitoring.** [Automatic evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/automatic-evaluations/) runs your judges on traces and multi-turn conversations as they are logged.
- **Multi-turn testing.** [Conversation simulation](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/conversation-simulation/) tests agents before you have real users.

### Free, with no feature gating

Open-source MLflow has no paid tier and no license keys. Every feature in this section is in the open-source release. If you need a feature, [open a feature request on GitHub](https://github.com/mlflow/mlflow/issues). The maintainers review suggestions and feature requests from the community.

## 2. Langfuse

Langfuse is an MIT-licensed LLM engineering platform with tracing, LLM-as-a-judge and code evaluators, experiments, human annotation, and prompt management. Its open-source edition includes all core features with no usage limits, and a set of enterprise administration features requires a paid license key. Source: [Langfuse evaluation docs](https://langfuse.com/docs/evaluation/overview).

- **Best for:** teams that want tracing, evaluation, and prompt management in one platform.
- **Key fact:** 36K+ GitHub stars, and the core is MIT-licensed.

### Key evaluation features

- **LLM-as-a-judge.** Score production traces or datasets with LLM evaluators.
- **Experiments.** Run test cases against datasets and compare results side by side.
- **Code evaluators.** Write custom checks in Python or TypeScript.
- **Human annotation.** Review traces with your team and build golden datasets.
- **Prompt management.** Version prompts and roll back to earlier versions.

### How do you self-host Langfuse?

Langfuse runs web and worker containers on PostgreSQL, ClickHouse, Redis or Valkey, and S3-compatible object storage. This architecture is the same one Langfuse Cloud uses. Most features are in the MIT-licensed core. Project-level RBAC, audit logs, data retention policies, server-side data masking, and SCIM are among the features that [require an enterprise license key](https://langfuse.com/self-hosting/license-key).

## 3. Arize Phoenix

Arize Phoenix is a tracing and evaluation tool built on OpenTelemetry, with LLM evaluators, pre-built metrics, versioned datasets, and experiments. It is source-available under the Elastic License 2.0, not an OSI-approved open-source license. You can self-host it for free, but you cannot offer it to others as a hosted service. Source: [Phoenix evaluation docs](https://arize.com/docs/phoenix/evaluation/llm-evals/evaluator-traces).

- **Best for:** teams that want OpenTelemetry tracing and experiments in a tool that is simple to run, and whose use fits the Elastic License 2.0.
- **Key fact:** Arize AX is Arize's separate managed platform. Phoenix is the self-hosted tool.

### Key evaluation features

- **LLM evaluators.** Run pre-built or custom evaluators that use an LLM as a judge.
- **Pre-built metrics.** Faithfulness, correctness, document relevance, tool use, and deterministic checks.
- **Datasets and experiments.** Version datasets and track experiment results.
- **OpenTelemetry tracing.** Instrument LLM apps through OpenTelemetry.

### How do you self-host Arize Phoenix?

Phoenix runs as a single application that stores data in SQLite, or in PostgreSQL for production. Source: [Phoenix Docker deployment docs](https://arize.com/docs/phoenix/self-hosting/deployment-options/docker). Read the [license](https://github.com/Arize-ai/phoenix/blob/main/LICENSE) before you build a product on Phoenix.

## 4. Opik

Opik, from Comet, is an Apache 2.0 platform to trace, evaluate, and monitor LLM apps, RAG systems, and agents. It offers LLM-as-a-judge metrics for hallucination and relevance, test suites, datasets and experiments, human annotation, and PyTest integration for CI. You can self-host the full platform, and Comet also runs a hosted version. Source: [Opik evaluation docs](https://www.comet.com/docs/opik/evaluation/metrics/overview).

- **Best for:** teams that want an Apache 2.0 platform with assertion-based test suites and PyTest integration.
- **Key fact:** 22K+ GitHub stars.

### Key evaluation features

- **LLM-as-a-judge metrics.** Hallucination, answer relevance, context precision and recall, and moderation.
- **Test suites.** Assertion-based tests with pass or fail results.
- **Datasets and experiments.** Compare results side by side.
- **Human annotation.** Review queues for your team.
- **CI integration.** Run evaluations from PyTest.

### How do you self-host Opik?

Opik's [Docker Compose deployment](https://github.com/comet-ml/opik/tree/main/deployment/docker-compose) runs its backend and frontend services with MySQL, ClickHouse, Redis, ZooKeeper, and MinIO. Kubernetes deployment uses Helm.

## 5. Promptfoo

Promptfoo is an MIT-licensed command-line tool and library for testing prompts, models, and agents. It is strongest at red teaming and security scans, and at side-by-side comparisons across model providers, which you can run locally or in CI. Source: [Promptfoo red teaming docs](https://www.promptfoo.dev/docs/red-team/quickstart/).

- **Best for:** developers and security teams who red-team LLM apps and compare models from the command line.
- **Key fact:** 26K+ GitHub stars.

### Key evaluation features

- **Red teaming.** Scan for vulnerabilities and measure attack success rates.
- **Model comparison.** Compare outputs across OpenAI, Anthropic, Azure, Bedrock, Ollama, and other providers.
- **CI integration.** Run checks in pull requests.
- **Custom assertions.** Define your own test cases and evaluation logic.

### How do you self-host Promptfoo?

Promptfoo runs locally, so evaluations do not need a server. It also has a Docker image for sharing results, but its [self-hosting docs](https://www.promptfoo.dev/docs/usage/self-hosting/) say that the self-hosted server is not recommended for production: it uses a local SQLite database, it does not scale across replicas, and it has no built-in authentication.

## 6. Ragas

Ragas is an Apache 2.0 Python library for evaluating RAG and LLM apps. It provides metrics such as faithfulness, context precision, and context recall, and it can generate test datasets when you have none. It runs locally with no server, and its metrics also run inside MLflow as scorers. Source: [Ragas metrics docs](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/).

- **Best for:** teams that evaluate RAG pipelines and want a lightweight library with synthetic test data generation.
- **Key fact:** maintained by Vibrant Labs, with 16K+ GitHub stars.

### Key evaluation features

- **RAG metrics.** Context precision, context recall, faithfulness, and response relevancy.
- **General metrics.** LLM-based and traditional metrics for agents and general LLM quality.
- **Test data generation.** Generate evaluation datasets without existing test data.
- **Integrations.** Works with LangChain, LlamaIndex, and other frameworks.

### How do you self-host Ragas?

You don't host Ragas. It is a library that you install with `pip`. To store results, compare runs, or add human review, run Ragas metrics inside a platform such as [MLflow](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/third-party/ragas/).

## Self-hosting compared: what each tool takes to run

| Tool          | What you run                                                                | Paid license needed for                                        |
| ------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------- |
| MLflow        | One server; SQLite to start, PostgreSQL or MySQL for production; Helm chart | Nothing                                                        |
| Langfuse      | Web and worker containers, PostgreSQL, ClickHouse, Redis, object storage    | Project-level RBAC, audit logs, data retention, SCIM, and more |
| Arize Phoenix | One application; SQLite or PostgreSQL                                       | Nothing, but offering it as a hosted service is not allowed    |
| Opik          | Backend and frontend services, MySQL, ClickHouse, Redis, ZooKeeper, MinIO   | Nothing                                                        |
| Promptfoo     | Nothing for local runs; an optional single-server Docker image              | Production server features are in the enterprise product       |
| Ragas         | Nothing; it is a Python library                                             | Nothing                                                        |

MLflow and Arize Phoenix each run as a single application on one database, which is the least to operate of the platforms here. With MLflow, you can add replicas later with the Helm chart.

## Frameworks vs. platforms

The six tools fall into two groups.

- **Frameworks** are libraries of metrics and tests that you run in code. Ragas and Promptfoo are frameworks. They are quick to start, but you store results, compare runs, and collect human feedback somewhere else.
- **Platforms** add tracing, storage, a UI, datasets, human review, and production monitoring. MLflow, Langfuse, Arize Phoenix, and Opik are platforms.

You do not have to choose one or the other. MLflow runs Ragas, DeepEval, Phoenix, and TruLens metrics as [third-party scorers](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/third-party/), so you can keep the metrics your team already trusts and add tracing, review, and monitoring around them.

## How to choose an open-source LLM evaluation tool

Start from your constraints, not from the feature list.

| If your team...                                                                     | Choose        |
| ----------------------------------------------------------------------------------- | ------------- |
| Wants one free, self-hosted tool to trace, evaluate, find issues, and optimize      | MLflow        |
| Needs every feature, including access control, with no license key                  | MLflow        |
| Wants tracing, evals, and prompt management, and already runs ClickHouse            | Langfuse      |
| Wants simple OpenTelemetry tracing and experiments, and accepts Elastic License 2.0 | Arize Phoenix |
| Wants assertion-based test suites with PyTest                                       | Opik          |
| Needs red teaming or model comparison from the command line                         | Promptfoo     |
| Needs RAG metrics in a lightweight library                                          | Ragas         |

Before you commit, run a short pilot on one real agent workflow. Self-host the tool, collect traces, find at least three real failures, and check how the tool helps you measure and fix them.

## FAQ

### What is the best open-source LLM evaluation tool in 2026?

MLflow is the best open-source LLM evaluation tool for most teams in 2026. It is free under the Apache 2.0 license, has no feature gating, and covers tracing, LLM judges, automatic issue detection, human review, production monitoring, and prompt and judge optimization. Langfuse is a strong alternative for tracing and prompt management, and Ragas is a good choice for RAG metrics.

### What is the best self-hosted LLM evaluation platform?

MLflow is the best self-hosted LLM evaluation platform for most teams. You can start it with `pip install mlflow` and `mlflow server`, move to PostgreSQL and Kubernetes for production, and use every feature, including access control, without a license key. Langfuse and Opik are also self-hostable, with larger deployments built around ClickHouse.

### Can I self-host MLflow for free?

Yes. MLflow is open source under the Apache 2.0 license, and you can self-host every feature for free, including evaluation, tracing, review queues, and prompt optimization. There is no paid tier and no license key.

### Is Arize Phoenix open source?

Arize Phoenix is source-available, not open source by the OSI definition. It uses the Elastic License 2.0, which lets you use and self-host Phoenix for free but does not allow you to offer it to others as a hosted or managed service.

### What is the best open-source RAG evaluation tool?

Ragas is a popular open-source library for RAG metrics such as faithfulness, context precision, and context recall, and it can generate test data. To trace your RAG pipeline, store results, and add human review, run Ragas metrics inside MLflow as third-party scorers.

### What is the difference between Langfuse and MLflow?

Both are open-source platforms for tracing and evaluating LLM apps. MLflow is Apache 2.0, governed by the Linux Foundation, and has no license-gated features. It also covers prompt and judge optimization and classic ML. Langfuse is MIT-licensed with some enterprise features under a commercial license.

## Get started with MLflow evaluation

Install MLflow with `pip install mlflow`, start a server with `mlflow server`, and follow the [evaluation quickstart](https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/) to run your first LLM judge in a few minutes.

## Recommended

- [Best LLM Evaluation Platforms in 2026 (Part 1)](/articles/best-llm-evaluation-platforms-2026)
- [Langfuse Alternative | MLflow](https://mlflow.org/langfuse-alternative)
- [Arize Phoenix Alternative | MLflow](https://mlflow.org/arize-phoenix-alternative)
- [LLM Evaluation | MLflow](https://mlflow.org/llm-evaluation)
- [Self-Hosting MLflow](https://mlflow.org/docs/latest/self-hosting/)
