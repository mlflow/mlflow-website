# RetrievalSufficiency judge

The `RetrievalSufficiency` judge evaluates whether the retrieved context (from RAG applications, agents, or any system that retrieves documents) contains enough information to adequately answer the user's request based on the ground truth label provided as `expected_facts` or an `expected_response`.

This built-in LLM judge is designed for evaluating RAG systems where you need to ensure that your retrieval process is providing all necessary information.

## Prerequisites for running the examples[​](#prerequisites-for-running-the-examples "Direct link to Prerequisites for running the examples")

1. Install MLflow and required packages

   bash

   ```bash
   pip install --upgrade mlflow

   ```

2. Create an MLflow experiment by following the [setup your environment quickstart](/mlflow-website/docs/latest/genai/getting-started/connect-environment.md).

3. (Optional, if using OpenAI models) Use the native OpenAI SDK to connect to OpenAI-hosted models. Select a model from the [available OpenAI models](https://platform.openai.com/docs/models).

   python

   ```python
   import mlflow
   import os
   import openai

   # Ensure your OPENAI_API_KEY is set in your environment
   # os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>" # Uncomment and set if not globally configured

   # Enable auto-tracing for OpenAI
   mlflow.openai.autolog()

   # Create an OpenAI client
   client = openai.OpenAI()

   # Select an LLM
   model_name = "gpt-4o-mini"

   ```

## Usage examples[​](#usage-examples "Direct link to Usage examples")

The `RetrievalSufficiency` judge can be invoked directly for single trace assessment or used with MLflow's evaluation framework for batch evaluation.

**Requirements:**

* **Trace requirements**:

  * The MLflow Trace must contain at least one span with `span_type` set to `RETRIEVER`
  * `inputs` and `outputs` must be on the Trace's root span

* **Ground-truth labels**: Required - must provide either `expected_facts` or `expected_response` in the expectations dictionary

- Invoke directly
- Invoke with evaluate()

python

```python
from mlflow.genai.scorers import RetrievalSufficiency
import mlflow

# Get a trace from a previous run
trace = mlflow.get_trace("<your-trace-id>")

# Assess if the retrieved context is sufficient for the expected facts
feedback = RetrievalSufficiency()(
    trace=trace,
    expectations={
        "expected_facts": [
            "MLflow has four main components",
            "Components include Tracking",
            "Components include Projects",
            "Components include Models",
            "Components include Registry",
        ]
    },
)
print(feedback)

```

python

```python
import mlflow
from mlflow.genai.scorers import RetrievalSufficiency

# Evaluate traces from previous runs with ground truth expectations
results = mlflow.genai.evaluate(
    data=eval_dataset,  # Dataset with trace data and expected_facts
    scorers=[RetrievalSufficiency()],
)

```

tip

For a complete RAG application example with this judge, see the [RAG Evaluation guide](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/rag.md).

## Interpret results[​](#interpret-results "Direct link to Interpret results")

The RetrievalSufficiency judge evaluates each retriever span separately and returns a separate Feedback object for each retriever span in your trace. Each [`Feedback`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.Feedback) object contains:

* **value**: "yes" if the retrieved documents contain all the information needed to generate the expected facts, "no" if the retrieved documents are missing critical information
* **rationale**: Explanation of which expected facts the context covers or lacks

This helps you identify when your retrieval system is failing to fetch all necessary information, which is a common cause of incomplete or incorrect responses in RAG applications.

## Select the LLM that powers the judge[​](#select-the-llm-that-powers-the-judge "Direct link to Select the LLM that powers the judge")

You can change the judge model by using the `model` argument in the judge definition. The model must be specified in the format `<provider>:/<model-name>`, where `<provider>` is a LiteLLM-compatible model provider.

For a list of supported models, see [selecting judge models](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges.md#selecting-judge-models).

## Next steps[​](#next-steps "Direct link to Next steps")

### [Evaluate context relevance](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/relevance.md#retrievalrelevance-judge)

[Ensure retrieved documents are relevant before checking sufficiency](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/relevance.md#retrievalrelevance-judge)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/relevance.md#retrievalrelevance-judge)

### [Evaluate groundedness](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/groundedness.md)

[Verify that responses use only the provided context](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/groundedness.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/groundedness.md)

### [Build evaluation datasets](/mlflow-website/docs/latest/genai/datasets.md)

[Create ground truth datasets with expected facts for testing](/mlflow-website/docs/latest/genai/datasets.md)

[Learn more →](/mlflow-website/docs/latest/genai/datasets.md)
