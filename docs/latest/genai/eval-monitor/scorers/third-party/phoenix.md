# Arize Phoenix

[Arize Phoenix](https://docs.arize.com/phoenix/) is an open-source LLM observability and evaluation framework from Arize AI. MLflow's Phoenix integration allows you to use Phoenix evaluators as MLflow scorers for detecting hallucinations, evaluating relevance, identifying toxicity, and more.

## Prerequisites[​](#prerequisites "Direct link to Prerequisites")

Phoenix scorers require the `arize-phoenix-evals` package:

bash

```bash
pip install arize-phoenix-evals

```

## Quick Start[​](#quick-start "Direct link to Quick Start")

You can call Phoenix scorers directly:

python

```python
from mlflow.genai.scorers.phoenix import Hallucination

scorer = Hallucination(model="openai:/gpt-4")
feedback = scorer(
    inputs="What is the capital of France?",
    outputs="Paris is the capital of France.",
    expectations={"context": "France is a country in Europe. Its capital is Paris."},
)

print(feedback.value)  # "factual" or "hallucinated"
print(feedback.metadata["score"])  # Numeric score

```

Or use them in [mlflow.genai.evaluate](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.evaluate):

python

```python
import mlflow
from mlflow.genai.scorers.phoenix import Hallucination, Relevance

eval_dataset = [
    {
        "inputs": {"query": "What is MLflow?"},
        "outputs": "MLflow is an open-source platform for managing machine learning workflows.",
        "expectations": {
            "context": "MLflow is an ML platform for experiment tracking and model deployment."
        },
    },
    {
        "inputs": {"query": "How do I track experiments?"},
        "outputs": "You can use mlflow.start_run() to begin tracking experiments.",
        "expectations": {
            "context": "MLflow provides APIs like mlflow.start_run() for experiment tracking."
        },
    },
]

results = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[
        Hallucination(model="openai:/gpt-4"),
        Relevance(model="openai:/gpt-4"),
    ],
)

```

## Available Phoenix Scorers[​](#available-phoenix-scorers "Direct link to Available Phoenix Scorers")

Phoenix scorers evaluate different aspects of LLM outputs:

| Scorer                                                                                                                             | What does it evaluate?                                              | Phoenix Docs                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| [Hallucination](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.phoenix.Hallucination) | Does the output contain fabricated information not in the context?  | [Link](https://docs.arize.com/phoenix/evaluation/how-to-evals/running-pre-tested-evals/hallucinations)            |
| [Relevance](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.phoenix.Relevance)         | Is the retrieved context relevant to the input query?               | [Link](https://docs.arize.com/phoenix/evaluation/how-to-evals/running-pre-tested-evals/relevance)                 |
| [Toxicity](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.phoenix.Toxicity)           | Does the output contain toxic or harmful content?                   | [Link](https://docs.arize.com/phoenix/evaluation/how-to-evals/running-pre-tested-evals/toxicity)                  |
| [QA](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.phoenix.QA)                       | Does the answer correctly address the question based on reference?  | [Link](https://docs.arize.com/phoenix/evaluation/how-to-evals/running-pre-tested-evals/q-and-a-on-retrieved-data) |
| [Summarization](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.phoenix.Summarization) | Is the summary accurate and complete relative to the original text? | [Link](https://docs.arize.com/phoenix/evaluation/how-to-evals/running-pre-tested-evals/summarization)             |

## Creating Scorers by Name[​](#creating-scorers-by-name "Direct link to Creating Scorers by Name")

You can also create Phoenix scorers dynamically using [get\_scorer](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.phoenix.get_scorer):

python

```python
from mlflow.genai.scorers.phoenix import get_scorer

# Create scorer by name
scorer = get_scorer(
    metric_name="Hallucination",
    model="openai:/gpt-4",
)

feedback = scorer(
    inputs="What is MLflow?",
    outputs="MLflow is a platform for ML workflows.",
    expectations={"context": "MLflow is an ML platform."},
)

```

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [Evaluate Agents](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

[Learn specialized techniques for evaluating AI agents with tool usage](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

### [Evaluate Traces](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

[Evaluate production traces to understand application behavior](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

### [Built-in Judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md)

[Explore MLflow's built-in evaluation judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md)
