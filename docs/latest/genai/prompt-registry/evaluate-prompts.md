# Evaluating Prompts

Combining [MLflow Prompt Registry](/mlflow-website/docs/latest/genai/prompt-registry.md) with [MLflow LLM Evaluation](/mlflow-website/docs/latest/genai/eval-monitor.md) enables you to evaluate prompt performance across different models and datasets, and track the evaluation results in a centralized registry. You can also inspect model outputs from the **traces** logged during evaluation to understand how the model responds to different prompts.

Key Benefits of MLflow Prompt Evaluation

* **Effective Evaluation**: \`MLflow's LLM Evaluation API provides a simple and consistent way to evaluate prompts across different models and datasets without writing boilerplate code.
* **Compare Results**: Compare evaluation results with ease in the MLflow UI.
* **Tracking Results**: Track evaluation results in MLflow Experiment to maintain the history of prompt performance and different evaluation settings.
* **Tracing**: Inspect model behavior during inference deeply with traces generated during evaluation.

## Quickstart[​](#quickstart "Direct link to Quickstart")

### 1. Install Required Libraries[​](#1-install-required-libraries "Direct link to 1. Install Required Libraries")

First install MLflow and OpenAI SDK. If you use different LLM providers, install the corresponding SDK instead.

bash

```bash
pip install 'mlflow[genai]>=2.21.0' openai -qU

```

Also set OpenAI API key (or any other LLM providers e.g. Anthropic).

python

```python
import os
from getpass import getpass

os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")

```

### 1. Create a Prompt[​](#1-create-a-prompt "Direct link to 1. Create a Prompt")

* UI
* Python

![Create Prompt UI](/mlflow-website/docs/latest/assets/images/create-prompt-ui-03c88144e65d28eb7847b2ae5d8dd49a.png)

1. Run `mlflow server` in your terminal to start the MLflow UI.
2. Navigate to the **Prompts** tab in the MLflow UI.
3. Click on the **Create Prompt** button.
4. Fill in the prompt details such as name, prompt template text, and commit message (optional).
5. Click **Create** to register the prompt.

To create a new prompt using the Python API, use [`mlflow.register_prompt()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.register_prompt) API:

python

```python
import mlflow

# Use double curly braces for variables in the template
initial_template = """\
Summarize content you are provided with in {{ num_sentences }} sentences.

Sentences: {{ sentences }}
"""

# Register a new prompt
prompt = mlflow.genai.register_prompt(
    name="summarization-prompt",
    template=initial_template,
    # Optional: Provide a commit message to describe the changes
    commit_message="Initial commit",
)

# The prompt object contains information about the registered prompt
print(f"Created prompt '{prompt.name}' (version {prompt.version})")

```

### 2. Prepare Evaluation Data[​](#2-prepare-evaluation-data "Direct link to 2. Prepare Evaluation Data")

Below, we create a small summarization dataset for demonstration purposes.

python

```python
import pandas as pd

eval_data = [
    {
        "inputs": {
            "sentences": "Artificial intelligence has transformed how businesses operate in the 21st century. Companies are leveraging AI for everything from customer service to supply chain optimization. The technology enables automation of routine tasks, freeing human workers for more creative endeavors. However, concerns about job displacement and ethical implications remain significant. Many experts argue that AI will ultimately create more jobs than it eliminates, though the transition may be challenging.",
        },
        "expectations": {
            "summary": "AI has revolutionized business operations through automation and optimization, though ethical concerns about job displacement persist alongside predictions that AI will ultimately create more employment opportunities than it eliminates.",
        },
    },
    {
        "inputs": {
            "sentences": "Climate change continues to affect ecosystems worldwide at an alarming rate. Rising global temperatures have led to more frequent extreme weather events including hurricanes, floods, and wildfires. Polar ice caps are melting faster than predicted, contributing to sea level rise that threatens coastal communities. Scientists warn that without immediate and dramatic reductions in greenhouse gas emissions, many of these changes may become irreversible. International cooperation remains essential but politically challenging.",
        },
        "expectations": {
            "summary": "Climate change is causing accelerating environmental damage through extreme weather events and melting ice caps, with scientists warning that without immediate reduction in greenhouse gas emissions, many changes may become irreversible.",
        },
    },
    {
        "inputs": {
            "sentences": "The human genome project was completed in 2003 after 13 years of international collaborative research. It successfully mapped all of the genes of the human genome, approximately 20,000-25,000 genes in total. The project cost nearly $3 billion but has enabled countless medical advances and spawned new fields like pharmacogenomics. The knowledge gained has dramatically improved our understanding of genetic diseases and opened pathways to personalized medicine. Today, a complete human genome can be sequenced in under a day for about $1,000.",
        },
        "expectations": {
            "summary": "The Human Genome Project, completed in 2003, mapped approximately 20,000-25,000 human genes at a cost of $3 billion, enabling medical advances, improving understanding of genetic diseases, and establishing the foundation for personalized medicine.",
        },
    },
    {
        "inputs": {
            "sentences": "Remote work adoption accelerated dramatically during the COVID-19 pandemic. Organizations that had previously resisted flexible work arrangements were forced to implement digital collaboration tools and virtual workflows. Many companies reported surprising productivity gains, though concerns about company culture and collaboration persisted. After the pandemic, a hybrid model emerged as the preferred approach for many businesses, combining in-office and remote work. This shift has profound implications for urban planning, commercial real estate, and work-life balance.",
        },
        "expectations": {
            "summary": "The COVID-19 pandemic forced widespread adoption of remote work, revealing unexpected productivity benefits despite collaboration challenges, and resulting in a hybrid work model that impacts urban planning, real estate, and work-life balance.",
        },
    },
]

```

### 3. Define Prediction Function[​](#3-define-prediction-function "Direct link to 3. Define Prediction Function")

Define a function that takes a DataFrame of inputs and returns a list of predictions.

MLflow will pass the input columns (`inputs` only in this example) to the function. The output string will be compared with the `targets` column to evaluate the model.

python

```python
import mlflow
import openai


def predict_fn(sentences: str) -> str:
    # Load the latest version of the registered prompt
    prompt = mlflow.genai.load_prompt("prompts:/summarization-prompt@latest")
    completion = openai.OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt.format(sentences=sentences, num_sentences=1),
            }
        ],
    )
    return completion.choices[0].message.content

```

### 4. Run Evaluation[​](#4-run-evaluation "Direct link to 4. Run Evaluation")

Run the [`mlflow.genai.evaluate()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.evaluate) API to evaluate the model with the prepared data and prompt. In this example, we will use the following two built-in metrics.

python

```python
from typing import Literal
from mlflow.genai.judges import make_judge

answer_similarity = make_judge(
    name="answer_similarity",
    instructions=(
        "Evaluated on the degree of semantic similarity of the provided output to the expected answer.\n\n"
        "Output: {{ outputs }}\n\n"
        "Expected: {{ expectations }}"
        "Return 'yes' if the output is similar to the expected answer, otherwise return 'no'."
    ),
    model="openai:/gpt-5-mini",
    feedback_value_type=Literal["yes", "no"],
)

results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=[answer_similarity],
)

```

### 5. View Results[​](#5-view-results "Direct link to 5. View Results")

You can view the evaluation results in the MLflow UI. Navigate to the **Experiments** tab, select the **Evaluations** tab, and click on the evaluation run to view the evaluation result.

![Evaluation Results](/mlflow-website/docs/latest/assets/images/prompt-evaluation-result-2a8eb8bdd0d27413488af07919dd844b.png)
