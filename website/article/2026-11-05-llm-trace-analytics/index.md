---
title: "LLM Trace Analytics: Where to Find It and What to Track"
description: "Where to find LLM trace analytics (MLflow, Langfuse, LangSmith, OpenTelemetry backends) and the seven metrics worth tracking, with setup steps for MLflow."
slug: llm-trace-analytics
tags:
  [
    trace analytics,
    LLM trace analytics,
    LLM analytics,
    LLM trace dashboards,
    LLM observability metrics,
    AI agent monitoring,
    token usage and cost tracking,
  ]
date: 2026-11-05
---

LLM trace analytics turns the traces your AI app already records into charts you can act on: request volume, latency, errors, token use, cost, tool calls, and quality scores. You find it in the dashboard of an LLM tracing tool such as MLflow, Langfuse, or LangSmith, or in an OpenTelemetry backend you already run. This guide covers where to look and which seven metrics to track first.

It is for engineers who already log traces, or are about to, and now need to answer "is it healthy, what does it cost, and is it any good?" Not for chemical or laboratory trace analysis, which is a different field that shares the name.

We work on MLflow, so MLflow gets the most space here. Disclosure: the MLflow team wrote this guide. Each tool section says what the tool's own documentation shows. MLflow details below come from the 3.17 documentation and release notes, and the latest release is 3.18.0 (2026-11-04); other tools' details come from their public docs on 2026-11-05.

Our view: you do not need a twenty-chart dashboard. Pick one place to look, record user, session, and app version on every trace, and watch a few numbers that you will actually act on. The rest of this guide is how to get there.

## TL;DR

The short version:

- **Where to find it:** the dashboard or Overview view in your tracing tool, or a metrics view in an OpenTelemetry backend you already run.
- **What to track first:** request count, latency, error rate, token usage, cost, tool-call success, and quality scores.
- **MLflow:** the experiment **Overview** tab has Usage, Quality, and Tool Calls views, and needs a SQL-backed tracking store.
- **Large trace tables:** MLflow 3.17.0 added opt-in daily rollups that make analytics queries faster without changing results.
- **Slice by something:** a metric with no breakdown by user, session, model, or version tells you something is wrong but not what.

## What is LLM trace analytics?

LLM trace analytics is the aggregate view over many traces. A single trace shows you one request: which model was called, which tools ran, how long each step took. Trace analytics adds up thousands of those requests so you can see rates, trends, and outliers.

It sits between two things you may already have. Debugging one trace tells you why a request failed. Trace analytics tells you how often requests fail, and whether last Tuesday's prompt change made it worse.

Its inputs are the fields every trace already carries: start time, duration, status, token counts, model name, tool calls, and any scores attached later. If your traces do not record those, no dashboard can invent them. Instrument first, then chart.

## Why does LLM trace analytics matter for agents?

An agent makes many model and tool calls per request, so cost and latency are the sum of a chain, not a single call. One slow tool or one retry loop can dominate a request, and a per-call average hides it.

Three problems come up again and again:

1. **Cost drifts quietly.** A longer system prompt or an extra agent step raises every request. You see it on the invoice before you see it in a log.
2. **Failures hide in successful responses.** A request can return HTTP 200 while the answer is wrong or a tool failed and the agent worked around it.
3. **Averages mislead.** A mean latency of two seconds can hide a tail that takes thirty. Percentiles show the tail.

Analytics will not tell you why. It tells you where to open a trace and start reading.

## Where can you find LLM trace analytics?

You can find it in three places: the dashboard of an LLM tracing tool, a metrics view in an OpenTelemetry-compatible observability backend, or a query you write yourself over exported traces. The right one depends on whether you already run an observability stack and how much you want to build.

The sections below cover what each tool's own documentation describes. Features change, so check the linked pages before you decide.

### MLflow: the Overview tab

MLflow is an Apache-2.0 open-source platform for agents, LLMs, and ML models, governed by the Linux Foundation's LF AI & Data. In an LLM or agent experiment, the **Overview** tab holds the analytics. The [Overview documentation](https://mlflow.org/docs/latest/genai/tracing/observe-with-traces/dashboard/) describes three views:

- **Usage:** requests over time with an average reference line, latency, errors, and token usage and token stats.
- **Quality:** a summary of scorer results and a chart for each assessment type found on your traces.
- **Tool Calls:** total tool calls, average latency, success rate, failed calls, and per-tool usage, latency, and error rate.

Every view has a time range selector and a time unit selector. Cost charts sit in the same tab: MLflow records token usage and estimated USD cost per LLM call, and the [token and cost documentation](https://mlflow.org/docs/latest/genai/tracing/token-usage-cost/) says the aggregated charts are in the Overview tab.

Two requirements trip people up. The dashboard needs a SQL-backed tracking store (SQLite, PostgreSQL, or MySQL); with the file store (file-based tracking, such as `./mlruns`) the charts do not render. And cost tracking needs MLflow 3.10.0 or later with the `[genai]` extra installed on the server.

Because MLflow is one open-source server, the same process that shows these charts also stores the traces and runs evaluation. If you want the wider picture, see [MLflow's LLM tracing page](https://mlflow.org/llm-tracing) and [AI monitoring](https://mlflow.org/ai-monitoring). The tradeoff is that the dashboard is tied to the MLflow experiment. If your analytics need to join traces with business data in a warehouse, you will export traces and query them yourself.

### Langfuse: custom dashboards and a metrics API

Langfuse documents [custom dashboards](https://langfuse.com/docs/metrics/features/custom-dashboards) built on a query engine that aggregates across traces, observations, users, sessions, and scores. Its [metrics overview](https://langfuse.com/docs/metrics/overview) lists cost, latency, volume, and quality as the core metrics, and trace name, user, tags, and release or version as the dimensions to slice by. It also ships curated dashboards for latency and cost, and a metrics API for pulling the numbers elsewhere.

This is a good fit if you want to build your own charts without writing queries.

### LangSmith: prebuilt and custom dashboards

LangSmith's [dashboard documentation](https://docs.langchain.com/langsmith/dashboards) says prebuilt dashboards are created automatically for each tracing project and cover essential metrics such as trace count, error rates, and token usage. Custom dashboards let you configure your own charts. The same page notes that which custom-dashboard experience you get depends on your deployment, so confirm that your setup supports the one you need.

### An OpenTelemetry backend you already run

If your platform team already runs Grafana, Datadog, or another OpenTelemetry-compatible backend, you can send LLM telemetry there. OpenTelemetry's [GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/model/gen-ai/metrics.yaml) define shared names, such as the `gen_ai.client.operation.duration` metric and `gen_ai.operation.name` values like `chat`, `execute_tool`, and `invoke_agent`. The metric definition marks that metric's refinements as `development` stability, so names may still change.

The upside is one place for application and LLM telemetry. The downside is that generic backends do not know about LLM-specific ideas such as sessions, assessments, or judge scores, so you build those views yourself. MLflow accepts traces over OTLP, so the two approaches can run side by side.

### Quick comparison

| Where                 | What you get out of the box                                  | Best when                                                    |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MLflow Overview       | Usage, Quality, and Tool Calls views; cost charts            | You want traces, analytics, and evaluation in one OSS server |
| Langfuse dashboards   | Curated cost and latency dashboards, custom widgets, API     | You want to build your own charts without queries            |
| LangSmith dashboards  | Prebuilt per-project dashboard, custom dashboards            | You already use LangChain tooling                            |
| OpenTelemetry backend | Whatever your backend supports, on shared GenAI metric names | Your platform team already runs one                          |

For a ranked view of the tracing tools themselves, read [Best Open-Source LLM Tracing Tools in 2026](https://mlflow.org/articles/best-open-source-llm-tracing-tools-2026).

## What should you track in LLM trace analytics?

Track seven things first: request count, latency percentiles, error rate, token usage, cost, tool-call success, and quality scores. Together they answer three questions: is it up, what does it cost, and is it any good. Add more only when one of these points at a problem.

| Metric                  | Question it answers                     | Slice it by                    |
| ----------------------- | --------------------------------------- | ------------------------------ |
| Request count           | Is traffic normal?                      | Time, app version              |
| Latency (p50, p95, p99) | How slow is the tail, not the mean?     | Model, agent step, tool        |
| Error rate              | How often does a request fail outright? | Model, tool, app version       |
| Token usage             | Did prompts or outputs get longer?      | Model, prompt version, user    |
| Cost                    | What does a request or a user cost?     | Model, provider, user, session |
| Tool-call success       | Which tool is failing or slow?          | Tool name                      |
| Quality scores          | Are answers getting better or worse?    | Scorer, app version            |

### Request count and error rate

These two are the health check. Request count tells you whether traffic matches what you expect, and a sudden drop is often a deploy problem rather than a demand problem. Error rate tells you how many requests ended in an error status.

Error rate has a blind spot: it only counts failures the code noticed. An agent that catches a tool error and answers anyway will look healthy. That is why the next two metrics matter.

### Latency percentiles

Track p50 and p95 at least. The median shows the typical request; p95 shows what your slowest one in twenty users feels. For agents, break latency down by step, since a single slow tool call usually explains a slow request.

On large tables, percentile queries can be the slow ones. See the rollup section below for which percentiles MLflow can serve from precomputed summaries.

### Token usage and cost

Token counts are the leading indicator; cost is the lagging one. MLflow records input, output, and total tokens per LLM call, plus an estimated cost in USD, at both span and trace level. The [token usage tracking guide](https://mlflow.org/articles/token-usage-tracking) goes further on what to do with them.

Watch tokens per request over time. A step up after a release usually means a longer prompt or an extra agent step. Remember that cost here is an estimate from model pricing, so reconcile it against your provider's invoice before using it for billing.

### Tool-call success

For agents, tool calls are where requests quietly break. A tool-level view shows which tool fails, which is slow, and how often the agent calls it. MLflow's Tool Calls view reports total calls, average latency, success rate, and failed calls, with error rate per tool.

### Quality scores

Quality is the metric that latency and cost cannot stand in for. In MLflow, [automatic evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/automatic-evaluations/) runs LLM judges on traces as they are logged, and the Quality view charts the results. Pick two or three judges that map to failures your users would notice, such as groundedness or whether the answer addresses the question, rather than a long list you will not read.

Judges are models too, and they make mistakes. Sample some scored traces by hand before you trust a trend. For the offline side of this, see [Continuous LLM Evaluation](https://mlflow.org/articles/continuous-llm-evaluation).

## How do you set up trace analytics in MLflow?

Set it up in five steps: run a SQL-backed server, trace your app, tag traces with user and session, check the Overview tab, and add judges. Each step takes minutes. Steps 3 and 5 are optional but make the charts far more useful.

1. **Start a SQL-backed server with the GenAI extra.** This gives you charts and cost tracking.

   ```bash
   pip install 'mlflow[genai]>=3.12.0'
   mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
   ```

2. **Trace your app.** Point the client at the server and turn on autologging for your provider or framework. This example uses OpenAI.

   ```python
   import mlflow

   mlflow.set_tracking_uri("http://localhost:5000")
   mlflow.set_experiment("support-agent")
   mlflow.openai.autolog()
   ```

3. **Tag traces with a user and a session.** This lets you group and filter by who and which conversation. The `session_id` and `user` parameters need MLflow 3.12.0 or later. The [users and sessions documentation](https://mlflow.org/docs/latest/genai/tracing/track-users-sessions/) says 3.11.0, but the keyword arguments first shipped in the 3.12.0 release.

   ```python
   with mlflow.tracing.context(session_id="session-123", user="user-456"):
       agent.invoke("What is the capital of France?")
   ```

4. **Open the Overview tab.** In the MLflow UI, open the experiment and select **Overview**. Send a few requests first: the dashboard stays empty until traces exist.

5. **Add automatic evaluation.** Configure judges so the Quality view fills in. See the [automatic evaluation documentation](https://mlflow.org/docs/latest/genai/eval-monitor/automatic-evaluations/).

### Can you query trace data yourself?

Yes. When a chart does not answer your question, `mlflow.search_traces` returns traces that match a filter, and each trace carries its duration, status, and token usage. This sketch finds the slowest successful traces and totals their tokens.

```python
import mlflow

experiment = mlflow.set_experiment("support-agent")  # creates it if missing

traces = mlflow.search_traces(
    locations=[experiment.experiment_id],
    filter_string="trace.status = 'OK' AND trace.execution_time_ms > 1000",
    order_by=["trace.execution_time_ms DESC"],
    max_results=50,
    return_type="list",
)

for t in traces[:5]:
    usage = t.info.token_usage or {}
    print(t.info.trace_id, t.info.execution_time_ms, usage.get("total_tokens"))
```

The filter syntax is in the [search traces documentation](https://mlflow.org/docs/latest/genai/tracing/search-traces/).

## How do you keep trace analytics fast on large trace tables?

On a large SQL-backed server, analytics queries scan many rows, and charts get slow. MLflow 3.17.0 added opt-in daily rollups: precomputed daily summaries that serve whole-day portions of analytics queries. The [3.17.0 release notes](https://github.com/mlflow/mlflow/releases/tag/v3.17.0) list it under "Faster Analytics for Large Trace Datasets."

What the [backend store documentation](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/) says about rollups:

- **Off by default.** Set `MLFLOW_SQL_TRACE_ROLLUPS_ENABLED=true` on the tracking server to turn them on.
- **Results do not change.** Days that are not yet rolled up, and partial days at the edges of a range, are served from the raw path. Rollups only change speed.
- **SQL only.** The feature needs a database backend, not the file store.
- **Maintenance runs on a schedule.** The default is a daily cron at 02:00 UTC. The first run after you enable rollups may take several scheduled runs to cover historical data.
- **Percentiles depend on the database.** The docs say p50, p90, and p99 are served from rollups only on PostgreSQL. Other percentiles, such as p95, and whole-range percentiles use the raw path.

Plan the upgrade before you enable anything. The release notes warn that SQL-backed servers need a coordinated schema upgrade with all writers stopped and a restorable backup, even if you leave rollups disabled. Mixed-version rolling upgrades are not supported. Run `mlflow db upgrade <database-url>` before restarting all replicas.

Most teams do not need rollups on day one. If your charts load fine, leave them off.

## What mistakes make trace analytics misleading?

The common mistakes are charting averages, skipping breakdowns, trusting error rate alone, and treating estimated cost as billing. Each one produces a dashboard that looks fine while something is wrong.

- **Averages instead of percentiles.** A good mean can hide a slow tail. Use p95.
- **No breakdowns.** Without model, version, user, or session, you know something changed but not what. Add metadata at trace time, because you cannot slice by a field you never recorded.
- **Error rate as the only health signal.** Agents recover from tool errors silently. Pair it with tool-call success and quality scores.
- **Estimated cost as an invoice.** Cost comes from model pricing tables. Use it to spot changes, and reconcile against the provider bill.
- **Too many judges.** Ten scores nobody reads are worse than three people trust. Start with the failures your users report.
- **Sampling away the evidence.** If you sample traces to save cost, make sure errors and slow requests are kept, or your tail metrics undercount.

Once the numbers are stable, turn the ones you care about into targets, such as a p95 latency ceiling per agent.

## FAQ

### Where do I find trace analytics for my LLM app?

Look in the dashboard of your LLM tracing tool: the Overview tab in MLflow, custom dashboards in Langfuse, or the project dashboard in LangSmith. If you run an OpenTelemetry-compatible observability backend, you can chart GenAI metrics there instead. Pick the place that already holds your traces.

### What metrics should I track for LLM applications?

Start with request count, latency percentiles, error rate, token usage, cost, tool-call success, and quality scores. They cover availability, spend, and answer quality. Slice each by model, version, user, or session so a change points to a cause.

### Does MLflow have a trace analytics dashboard?

Yes. The experiment Overview tab has Usage, Quality, and Tool Calls views, plus token and cost charts. It needs a SQL-backed tracking store, and cost tracking needs MLflow 3.10.0 or later with the `[genai]` extra on the server.

### Why is my MLflow Overview tab empty or not showing charts?

The usual causes are that no traces have been logged to the experiment yet, or that the server uses the file store (file-based tracking, such as `./mlruns`). The dashboard documentation says charts need a SQL-backed store such as SQLite, PostgreSQL, or MySQL. See [migrating from the file store](https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store).

### Do I need OpenTelemetry for trace analytics?

No. Tracing tools such as MLflow, Langfuse, and LangSmith ship their own dashboards. OpenTelemetry helps when you want LLM telemetry in the same backend as the rest of your services, or want to avoid re-instrumenting if you change tools.

### How is trace analytics different from LLM evaluation?

Trace analytics describes what happened across many requests: volume, speed, errors, and cost. Evaluation judges whether the outputs were good. They meet in the Quality view, where judge scores on live traces become a chart over time.

## Start with one dashboard and three slices

Pick one place to look, and make sure each trace records the user, session, and app version. Then watch latency percentiles, tokens per request, and one quality score for a week. You will learn more from that than from a twenty-chart dashboard.

To try it in MLflow, follow the [LLM tracing overview](https://mlflow.org/llm-tracing), start a server, and open the Overview tab on your first experiment.
