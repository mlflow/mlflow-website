---
title: Tracing Quickstart
slug: tracing-quickstart
description: Get started with MLflow Tracing to observe and debug your LLM applications.
tags: [tracing, quickstart, genai]
---

This cookbook walks you through setting up MLflow Tracing for your LLM applications.

<!-- truncate -->

## Prerequisites

- Python 3.9+
- MLflow 3.0+

## Install Dependencies

```bash
pip install mlflow openai
```

## Enable Tracing

```python
import mlflow

mlflow.openai.autolog()
```

## Run a Simple Example

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello, world!"}],
)
print(response.choices[0].message.content)
```

After running this code, navigate to the MLflow UI to see the trace of your LLM call, including latency, token usage, and the full request/response payloads.

## Next Steps

- [Evaluate LLM outputs](/cookbook/evaluation-quickstart) with MLflow scorers
