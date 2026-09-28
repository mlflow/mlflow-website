---
title: "OpenTelemetry First Model Endpoint Observability for Engineers"
description: "Guide for engineers to instrument GenAI endpoints with OpenTelemetry: PSI and token telemetry, LLM-as-a-Judge quality checks, runbooks, and MLflow cookbooks."
slug: model-endpoint-observability
tags:
  [
    observability for ML models,
    API endpoint monitoring,
    how to monitor model endpoints,
    model endpoint observability,
    real-time model diagnostics,
    model performance tracking,
  ]
date: 2026-09-26
image: https://media.babylovegrowth.ai/blog-images/organization-30814/1790445198644_Engineer-inspecting-model-endpoint-traces.jpeg
---

![Engineer inspecting model endpoint traces](https://media.babylovegrowth.ai/blog-images/organization-30814/1790445198644_Engineer-inspecting-model-endpoint-traces.jpeg)

Model endpoint observability is the continuous collection of metrics, traces, and logs from a deployed model's requests, token usage, and outputs, giving engineers the visibility to catch performance and data-quality problems before they cascade. It relies on OpenTelemetry and GenAI-specific telemetry as the shared standard for capturing this data. Done well, it shrinks the time between "something feels off" and "we found it."

---

> **TL;DR:**
>
> - Effective model endpoint observability requires measuring latency percentiles, error rates, token usage, and queue depths to identify performance and data-quality issues early.
> - Use OpenTelemetry to instrument endpoints, collect spans and metrics asynchronously, and route data to vendor-neutral backends for flexible, scalable monitoring.
> - Regular diagnostics should include synthetic probes, PSI metrics, token spans, and queue inspections to quickly detect stalls, resource bottlenecks, or silent failures.
> - Set alert thresholds on latency, drift, or queue depth, and automate investigation or retraining processes for prompt responses to quality degradations.
> - MLflow integrates observability signals like traces and drift detection, enabling unified monitoring and automated rollback or retraining based on real-time model health metrics.

---

## Table of Contents

- [What to measure: metrics, traces, logs, and GenAI-specific telemetry](#what-to-measure-metrics-traces-logs-and-genai-specific-telemetry)
- [Instrumentation and data flow: OpenTelemetry, collectors, exporters, and pipeline design](#instrumentation-and-data-flow-opentelemetry-collectors-exporters-and-pipeline-design)
- [Health checks and diagnostics for model endpoints](#health-checks-and-diagnostics-for-model-endpoints)
- [Alerting, SLOs, and thresholds for model endpoints](#alerting-slos-and-thresholds-for-model-endpoints)
- [Operational playbook: triage steps and runbook for common endpoint incidents](#operational-playbook-triage-steps-and-runbook-for-common-endpoint-incidents)
- [How MLflow supports these observability patterns](#how-mlflow-supports-these-observability-patterns)
- [Why OTel-first GenAI observability is the pragmatic path forward](#why-otel-first-genai-observability-is-the-pragmatic-path-forward)
- [Try MLflow's observability cookbooks for your next deployment](#try-mlflows-observability-cookbooks-for-your-next-deployment)
- [Where to go next for implementation details](#where-to-go-next-for-implementation-details)
- [Sources](#sources)
- [FAQ](#faq)

## What to measure: metrics, traces, logs, and GenAI-specific telemetry

Traditional golden signals still apply to model endpoints, but they need a GenAI lens. Latency percentiles (p50/p95/p99), error rate, and throughput tell you whether the endpoint is healthy at the infrastructure level. Token counts, finish reasons, and prompt/response spans tell you whether the model itself is behaving.

We recommend instrumenting endpoints to capture:

- **Golden signals**: latency percentiles, error rate, and requests per second at the endpoint level.
- **Model-specific telemetry**: token usage, finish reasons, and spans for intermediate tool calls in agentic workflows.
- **Data-quality checks**: null rates, schema validation failures, and distribution drift measured with Jensen-Shannon divergence, PSI, or the Kolmogorov-Smirnov test.
- **Queue and stall indicators**: queue depth and inter-token latency, which often flag GenAI degradation before CPU usage moves at all.

**PSI (Pressure Stall Information) metrics expose time lost to resource stalls at the node, pod, and container level**, and they're often a more reliable signal than raw CPU utilization for catching latency spikes in GPU-backed inference. A GPU can look underutilized on paper while requests queue behind it, and PSI is what surfaces that gap.

## Instrumentation and data flow: OpenTelemetry, collectors, exporters, and pipeline design

The practical pipeline looks like this: instrument the serving application with an OpenTelemetry SDK, route spans and metrics through a collector, then export to your metrics, tracing, and logging backends. This keeps instrumentation vendor-neutral and lets you swap backends without touching application code.

OpenTelemetry's GenAI semantic conventions define which attributes to attach to spans, including model name, token usage, and finish reasons, so [teams can compare token consumption and latency across providers](https://opentelemetry.io/) without inventing their own schema. To avoid adding inference latency, capture spans and metrics asynchronously and sample aggressively on high-volume endpoints rather than tracing every request in full detail.

| Pipeline stage   | Purpose                                | Example choice            |
| ---------------- | -------------------------------------- | ------------------------- |
| Instrumentation  | Emit spans, metrics, logs from the app | OpenTelemetry SDK         |
| Collector        | Batch, filter, and route telemetry     | OpenTelemetry Collector   |
| Backend          | Store and query telemetry              | Tracing and metrics store |
| Evaluation layer | Score trace quality offline            | LLM-as-a-Judge            |

Capturing raw prompt and response content adds real diagnostic value but also real cost and privacy exposure, so most teams redact or truncate content fields while keeping token counts and metadata intact.

## Health checks and diagnostics for model endpoints

Once telemetry is flowing, diagnostics come down to a short set of repeatable checks:

1. Run [synthetic probes](https://yonderly.com/ai-can-build-your-content-in-minutes-testing-it-still-takes-real-devices) against the endpoint on a fixed interval and compare responsiveness against baseline latency.
2. Check PSI and per-pod pressure metrics to catch stalls and GPU saturation that utilization graphs miss.
3. Pull token-level spans and finish reasons from traces to catch silent failures like truncated outputs or hallucinated responses that never trip an error code.
4. Inspect queue depth, GPU memory, and per-component latency to isolate whether the bottleneck sits in preprocessing, inference, or postprocessing.

**Pro Tip:** _Treat a missing PSI value as unknown, not zero. On mixed-OS clusters, Windows nodes won't report PSI at all, and [Kubernetes v1.36 changed kubelet behavior](https://kubernetes.io/docs/reference/instrumentation/understand-psi-metrics/) specifically to stop emitting misleading zero values on unsupported kernels._

Trace-based diagnostics matter most for GenAI endpoints because a request can return a 200 status code and still be wrong. A model that produces a fluent but hallucinated answer looks healthy on every infrastructure dashboard, which is exactly why token-level spans and offline evaluation belong in the standard health check rotation, not just in post-incident review.

## Alerting, SLOs, and thresholds for model endpoints

Define service level indicators around availability, latency percentiles, and model-quality signals such as drift scores or LLM-as-a-Judge ratings, then set service level objectives against each.

- Use anomaly detection for drift and quality metrics, since fixed thresholds tend to miss slow-moving degradation.
- Use hard threshold alerts for infrastructure events like queue depth spikes or GPU memory exhaustion, where fast, deterministic response matters more than nuance.
- Auto-trigger investigation or retraining jobs for well-understood drift patterns, but escalate to a human whenever quality signals move outside historical range without a clear cause.
- Set an error budget per model release so a rollout can be paused automatically once it burns through its allotted risk.

[Azure Machine Learning's model monitoring documentation](https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-monitoring?view=azureml-api-2) shows how event-driven workflows can launch a retraining or rollback job automatically once a drift or quality threshold is breached, removing the delay of waiting on a person to notice the dashboard.

## Operational playbook: triage steps and runbook for common endpoint incidents

When an endpoint degrades, a fixed sequence beats improvisation.

1. Verify synthetic tests are actually failing, and scope the blast radius by region, version, or traffic segment.
2. Check golden signal dashboards and traces to determine whether the issue is infrastructure, model behavior, or upstream data quality.
3. Contain the incident: scale out, shape traffic away from the affected version, or roll back, while capturing evidence (traces, logs, and a sample of affected requests) before it rotates out of retention.
4. After containment, label the affected data, run LLM-as-a-Judge evaluation against the suspect traces, and schedule a retrain if the evaluation confirms a real quality regression.

**Pro Tip:** _Archive prediction logs in object storage and roll up drift metrics on a fixed cadence, such as daily computation with a rolling visualization window, so [seasonal noise doesn't get mistaken for a real incident](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/)._

This sequence works because it separates detection from diagnosis from containment. Skipping straight to a rollback without checking whether the root cause is infrastructure or model quality tends to mask the real problem and invites a repeat incident within days.

## How MLflow supports these observability patterns

The platform's tracing captures the request and response spans described above, and its evaluation tools can score those traces asynchronously, so quality checks never sit on the inference path.

- MLflow can [receive OpenTelemetry GenAI-conformant traces](https://mlflow.org/articles/otel-for-llm) directly, so teams already instrumented with OTel don't need a separate export path.
- The [AI observability cookbooks](https://mlflow.org/ai-observability) walk through wiring traces, token usage, and evaluation into a single view.
- For drift-specific work, MLflow's [drift detection](https://mlflow.org/articles/tags/detecting-model-drift) and [drift tracking](https://mlflow.org/articles/tags/how-to-track-model-drift) articles pair naturally with the PSI and JS divergence checks covered earlier.
- This platform can work well as a central observability store for GenAI and agentic workloads; teams with heavy classical infrastructure monitoring needs will likely pair it with an existing metrics backend.

## Why OTel-first GenAI observability is the pragmatic path forward

OpenTelemetry's GenAI semantic conventions matter because they let you compare token usage and latency across model providers on one dashboard instead of building bespoke parsers for each. As agentic workloads scale, PSI and token-level metrics become the signals that actually predict trouble, while cost, privacy, and sampling decisions become the real engineering trade-offs, not afterthoughts.

> _— Kevin_

## Try MLflow's observability cookbooks for your next deployment

MLflow supports the full loop this article describes: tracing for every request, LLM-as-a-Judge evaluation for silent failures, and an AI Gateway for governing prompts and providers in one place.

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

If you're ready to send OpenTelemetry GenAI traces into a system built for them, start with the AI observability for LLMs and agents page, then explore [MLflow](https://mlflow.org) for the broader lifecycle toolkit.

## Where to go next for implementation details

![Where to go next for implementation details — overview diagram](https://media.babylovegrowth.ai/blog-images/organization-30814/1790445255253_Where-to-go-next-for-implementation-details-overview-diagram.jpeg)

Consult the OpenTelemetry GenAI conventions, Kubernetes PSI docs, the Gateway API inference extension, and Azure ML monitoring guidance for hands-on implementation steps.

## Sources

- [Understand Pressure Stall Information (PSI) Metrics | Kubernetes](https://kubernetes.io/docs/reference/instrumentation/understand-psi-metrics/)
- [Model monitoring | Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-monitoring?view=azureml-api-2)

## FAQ

### What are the top observability tools for GenAI endpoints?

There's no single canonical top three, since tooling choices depend on stack and provider mix, but teams commonly combine an OpenTelemetry collector, a tracing and metrics backend, and an evaluation layer like MLflow's LLM-as-a-Judge for quality scoring. The OpenTelemetry project itself underpins most modern GenAI observability stacks as the instrumentation layer.

### What are the four pillars of observability?

Classic observability rests on logs, metrics, and traces, with a fourth pillar for GenAI workloads covering model-specific telemetry like token usage and finish reasons. Each pillar answers a different question: what happened, how much, in what sequence, and what did the model actually produce.

### What is a SageMaker endpoint?

A SageMaker endpoint is a hosted HTTPS interface that serves real-time predictions from a deployed model on managed infrastructure. It handles the serving layer, but production teams still need to add their own metrics, traces, and drift monitoring on top of it for full observability.

### What is end-to-end observability for model endpoints?

End-to-end observability means tracking a request across the entire path, from the initial API call through model inference to the final response, correlating latency, token usage, and quality signals at every hop. It combines infrastructure metrics like PSI with model-level traces so an engineer can pinpoint whether a slow or wrong response originated in infrastructure, the model, or the input data.

## Recommended

- [Full OpenTelemetry Support in MLflow Tracing](https://mlflow.org/blog/opentelemetry-tracing-support)
- [Sampling Strategies for Distributed Systems Observability](https://mlflow.org/articles/observability-sampling-strategies)
