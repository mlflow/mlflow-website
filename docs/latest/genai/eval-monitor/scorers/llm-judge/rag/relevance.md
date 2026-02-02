# Answer and Context Relevance Judges

MLflow provides two built-in LLM judges to assess relevance in your GenAI applications. These judges help diagnose quality issues - if context isn't relevant, the generation step cannot produce a helpful response.

* `RelevanceToQuery`: Evaluates if your app's response directly addresses the user's input
* `RetrievalRelevance`: Evaluates if each document returned by your app's retriever(s) is relevant

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

## Usage Examples[​](#usage-examples "Direct link to Usage Examples")

### RelevanceToQuery Judge[​](#relevancetoquery-judge "Direct link to RelevanceToQuery Judge")

This judge evaluates if your app's response directly addresses the user's input without deviating into unrelated topics.

You can invoke the judge directly with a single input for testing, or pass it to [mlflow.genai.evaluate](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.evaluate) for running full evaluation on a dataset.

**Requirements:**

* **Trace requirements**: `inputs` and `outputs` must be on the Trace's root span

- Invoke directly
- Invoke with evaluate()

python

```python
import mlflow
from mlflow.genai.scorers import RelevanceToQuery

assessment = RelevanceToQuery(name="my_relevance_to_query")(
    inputs={"question": "What is the capital of France?"},
    outputs="The capital of France is Paris.",
)
print(assessment)

```

python

```python
import mlflow
from mlflow.genai.scorers import RelevanceToQuery

data = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "outputs": "The capital of France is Paris.",
    }
]
result = mlflow.genai.evaluate(data=data, scorers=[RelevanceToQuery()])

```

### RetrievalRelevance Judge[​](#retrievalrelevance-judge "Direct link to RetrievalRelevance Judge")

This judge evaluates if each document returned by your app's retriever(s) is relevant to the input request. It evaluates each retriever span separately and returns a separate [`Feedback`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.Feedback) object for each retriever span in your trace.

**Requirements:**

* **Trace requirements**: The MLflow Trace must contain at least one span with `span_type` set to `RETRIEVER`

- Invoke directly
- Invoke with evaluate()

python

```python
from mlflow.genai.scorers import RetrievalRelevance
import mlflow

# Get a trace from a previous run
trace = mlflow.get_trace("<your-trace-id>")

# Assess if each retrieved document is relevant
feedbacks = RetrievalRelevance()(trace=trace)
print(feedbacks)

```

python

```python
import mlflow
from mlflow.genai.scorers import RetrievalRelevance

# Evaluate traces from previous runs
results = mlflow.genai.evaluate(
    data=traces,  # DataFrame or list containing trace data
    scorers=[RetrievalRelevance()],
)

```

tip

For a complete RAG application example with these judges, see the [RAG Evaluation guide](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/rag.md).

## Select the LLM that powers the judge[​](#select-the-llm-that-powers-the-judge "Direct link to Select the LLM that powers the judge")

You can change the judge model by using the `model` argument in the judge definition. The model must be specified in the format `<provider>:/<model-name>`, where `<provider>` is a LiteLLM-compatible model provider.

For a list of supported models, see [selecting judge models](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges.md#selecting-judge-models).

## Interpret results[​](#interpret-results "Direct link to Interpret results")

The judge returns a [`Feedback`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.Feedback) object containing:

* **value**: "yes" if context is relevant, "no" if not
* **rationale**: Explanation of why the judge found the context relevant or irrelevant

## Next steps[​](#next-steps "Direct link to Next steps")

### [Explore other built-in judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md#available-judges)

[Learn about groundedness, safety, and correctness judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md#available-judges)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md#available-judges)

### [Create custom judges](#)

[Build specialized judges for your use case](#)

[Learn more →](#)

### [Evaluate RAG applications](/mlflow-website/docs/latest/genai/eval-monitor/quickstart.md)

[Apply relevance judges in comprehensive RAG evaluation](/mlflow-website/docs/latest/genai/eval-monitor/quickstart.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/quickstart.md)
