# GenAI Evaluation Quickstart

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/genai/eval-monitor/notebooks/quickstart-eval.ipynb)

This notebook will walk you through evaluating your GenAI applications with MLflow's comprehensive evaluation framework. In less than 5 minutes, you'll learn how to evaluate LLM outputs, use built-in and custom evaluation criteria, and analyze results in the MLflow UI.

## Prerequisites[​](#prerequisites "Direct link to Prerequisites")

Install the required packages by running:

python

```python
pip install 'mlflow[genai]' openai

```

## Step 1: Set up your environment[​](#step-1-set-up-your-environment "Direct link to Step 1: Set up your environment")

### Connect to MLflow[​](#connect-to-mlflow "Direct link to Connect to MLflow")

Before running evaluation, start the MLflow tracking server:

bash

```bash
mlflow server

```

This starts MLflow at <http://localhost:5000> with a SQLite backend (default).

Then configure your environment in the notebook:

python

```python
import os

import mlflow

# Configure environment
os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace with your API key
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment
mlflow.set_experiment("GenAI Evaluation Quickstart")

```

## Step 2: Define your mock agent's prediction function[​](#step-2-define-your-mock-agents-prediction-function "Direct link to Step 2: Define your mock agent's prediction function")

Create a prediction function that takes a question and returns an answer using OpenAI's gpt-4o-mini model.

python

```python
from openai import OpenAI

client = OpenAI()


def my_agent(question: str) -> str:
  response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
          {
              "role": "system",
              "content": "You are a helpful assistant. Answer questions concisely.",
          },
          {"role": "user", "content": question},
      ],
  )
  return response.choices[0].message.content


# Wrapper function for evaluation
def qa_predict_fn(question: str) -> str:
  return my_agent(question)

```

## Step 3: Prepare an evaluation dataset[​](#step-3-prepare-an-evaluation-dataset "Direct link to Step 3: Prepare an evaluation dataset")

The evaluation dataset is a list of samples, each with an `inputs` and `expectations` field.

* `inputs`: The input to the `predict_fn` function. **The key(s) must match the parameter name of the `predict_fn` function**.
* `expectations`: The expected output from the `predict_fn` function, namely, ground truth for the answer.

python

```python
# Define a simple Q&A dataset with questions and expected answers
eval_dataset = [
  {
      "inputs": {"question": "What is the capital of France?"},
      "expectations": {"expected_response": "Paris"},
  },
  {
      "inputs": {"question": "Who was the first person to build an airplane?"},
      "expectations": {"expected_response": "Wright Brothers"},
  },
  {
      "inputs": {"question": "Who wrote Romeo and Juliet?"},
      "expectations": {"expected_response": "William Shakespeare"},
  },
]

```

## Step 4: Define evaluation criteria using Scorers[​](#step-4-define-evaluation-criteria-using-scorers "Direct link to Step 4: Define evaluation criteria using Scorers")

**Scorer** is a function that computes a score for a given input-output pair against various evaluation criteria. You can use built-in scorers provided by MLflow for common evaluation criteria, as well as create your own custom scorers.

Here we use three scorers:

* **Correctness**: Evaluates if the answer is factually correct, using the "expected\_response" field in the dataset.
* **Guidelines**: Evaluates if the answer meets the given guidelines.
* **is\_concise**: A custom scorer to judge if the answer is concise (less than 5 words).

The first two scorers use LLMs to evaluate the response, so-called **LLM-as-a-Judge**.

python

```python
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines


@scorer
def is_concise(outputs: str) -> bool:
  """Evaluate if the answer is concise (less than 5 words)"""
  return len(outputs.split()) <= 5


scorers = [
  Correctness(),
  Guidelines(name="is_english", guidelines="The answer must be in English"),
  is_concise,
]

```

## Step 5: Run the evaluation[​](#step-5-run-the-evaluation "Direct link to Step 5: Run the evaluation")

Now we have all three components of the evaluation: dataset, prediction function, and scorers. Let's run the evaluation!

python

```python
# Run evaluation
results = mlflow.genai.evaluate(
  data=eval_dataset,
  predict_fn=qa_predict_fn,
  scorers=scorers,
)

```

## View Results[​](#view-results "Direct link to View Results")

After running the evaluation, go to the MLflow UI and navigate to your experiment. You'll see the evaluation results with detailed metrics for each scorer.

By clicking on each row in the table, you can see the detailed rationale behind the score and the trace of the prediction.

## Summary[​](#summary "Direct link to Summary")

Congratulations! You've successfully:

* ✅ Set up MLflow GenAI Evaluation for your applications
* ✅ Evaluated a Q\&A application with built-in scorers
* ✅ Created custom evaluation guidelines
* ✅ Learned to analyze results in the MLflow UI

MLflow's evaluation framework provides comprehensive tools for assessing GenAI application quality, helping you build more reliable and effective AI systems.
