---
title: Introducing GenAI Experiment Overview
description: Monitor your GenAI applications with comprehensive analytics and visualizations in MLflow's new Overview tab
slug: mlflow-genai-overview
authors: [mlflow-maintainers]
tags: [tracing, genai, observability, charts]
thumbnail: /img/blog/genai-overview.png
---

We're excited to introduce the **GenAI Experiment Overview tab**, a new analytics dashboard in MLflow that gives you comprehensive visibility into your GenAI application's health at a glance.

<video src={require("./overview_demo.mp4").default} controls loop autoPlay muted width="100%" />

## Challenges

Building GenAI applications is one thing, but keeping them running well in production is another. Teams often struggle to answer basic questions:

- Why is my application slow today?
- Are my outputs maintaining quality over time?
- Which tools in my agent are causing failures?
- How much am I spending on tokens?

Previously, answering these questions required digging through individual traces or building custom dashboards. The Overview tab changes that.

## One Dashboard, Complete Visibility

The Overview tab consolidates your GenAI application metrics into three focused views:

- **Usage**: Track requests, latency, errors, and token consumption over time
- **Quality**: Monitor agent quality based on your [MLflow scorers](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/)
- **Tool Calls**: Analyze agent tool performance, success rates, and error patterns

Each view includes time range and granularity controls, making it easy to zoom in on issues or spot long-term trends.

## Get Started

The Overview tab is available now in MLflow's GenAI experiments. Simply [enable tracing](https://mlflow.org/docs/latest/genai/tracing/) in your application, and MLflow will automatically populate the dashboard with your metrics.

For full details on each sub-tab and available metrics, check out the [MLflow Tracing UI documentation](https://mlflow.org/docs/latest/genai/tracing/observe-with-traces/ui/).

We'd love to hear your [feedback](https://github.com/mlflow/mlflow/issues) as you explore this new feature!
