---
title: Evaluation Quickstart
slug: evaluation-quickstart
description: Learn how to evaluate LLM outputs using MLflow's built-in scorers.
tags: [evaluation, quickstart, genai]
---

This cookbook shows you how to use `mlflow.genai.evaluate` to assess the quality of LLM outputs.

<!-- truncate -->

## Prerequisites

- Python 3.9+
- MLflow 3.0+

## Install Dependencies

```bash
pip install mlflow openai
```

## Define an Evaluation Dataset

```python
eval_data = [
    {
        "inputs": {"question": "What is MLflow?"},
        "expected_output": "MLflow is an open-source platform for the machine learning lifecycle.",
    },
    {
        "inputs": {"question": "What is tracing?"},
        "expected_output": "Tracing captures detailed execution data from LLM applications.",
    },
]
```

## Run Evaluation

```python
import mlflow

results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=my_predict_function,
    scorers=["correctness"],
)
print(results.tables["eval_results"])
```

## View Results

Navigate to the MLflow Experiment UI to see evaluation results, including per-row scores and aggregated metrics.

## Next Steps

- [Optimize prompts](/cookbook/prompt-engineering) with evaluation feedback loops
