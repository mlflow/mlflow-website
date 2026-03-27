---
title: Prompt Engineering with MLflow
slug: prompt-engineering
description: Manage, version, and optimize prompts using MLflow Prompt Registry.
tags: [prompts, prompt-registry, genai]
---

This cookbook demonstrates how to use MLflow Prompt Registry to manage and version your prompts.

<!-- truncate -->

## Prerequisites

- Python 3.9+
- MLflow 3.0+

## Install Dependencies

```bash
pip install mlflow openai
```

## Register a Prompt

```python
import mlflow

prompt = mlflow.register_prompt(
    name="qa-system-prompt",
    template="Answer the following question concisely: {{question}}",
)
print(f"Registered prompt version: {prompt.version}")
```

## Load and Use a Prompt

```python
prompt = mlflow.load_prompt("qa-system-prompt")

from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt.format(question="What is MLflow?")},
    ],
)
print(response.choices[0].message.content)
```

## Version Management

Every call to `mlflow.register_prompt` with updated content creates a new version, giving you full audit history and the ability to roll back.

## Next Steps

- [Optimize prompts automatically](/blog/mlflow-prompt-optimization) with GEPA
- [Evaluate prompt quality](/cookbook/evaluation-quickstart) with MLflow scorers
