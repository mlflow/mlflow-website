# ToolCallEfficiency Judge

The `ToolCallEfficiency` judge evaluates the agent's trajectory for redundancy in tool usage, such as tool calls with the same or similar arguments.

This built-in LLM judge is designed for evaluating AI agents and tool-calling applications where you need to ensure the agent operates efficiently without making unnecessary or duplicate tool calls.

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

The `ToolCallEfficiency` judge can be invoked directly for single trace assessment or used with MLflow's evaluation framework for batch evaluation.

**Requirements:**

* **Trace requirements**: - The MLflow Trace must contain at least one span with `span_type` set to `TOOL`

- Invoke directly
- Invoke with evaluate()

python

```python
from mlflow.genai.scorers import ToolCallEfficiency
import mlflow

# Get a trace from a previous run
trace = mlflow.get_trace("<your-trace-id>")

# Assess if tool calls are efficient
feedback = ToolCallEfficiency(name="my_tool_call_efficiency")(trace=trace)
print(feedback)

```

python

```python
import mlflow
from mlflow.genai.scorers import ToolCallEfficiency

# Evaluate traces from previous runs
results = mlflow.genai.evaluate(
    data=traces,  # DataFrame or list containing trace data
    scorers=[ToolCallEfficiency()],
)

```

tip

For a complete agent example with this judge, see the [Tool Call Evaluation guide](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call.md).

## Select the LLM that powers the judge[​](#select-the-llm-that-powers-the-judge "Direct link to Select the LLM that powers the judge")

You can change the judge model by using the `model` argument in the judge definition. The model must be specified in the format `<provider>:/<model-name>`, where `<provider>` is a LiteLLM-compatible model provider.

For a list of supported models, see [selecting judge models](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges.md#selecting-judge-models).

## Interpret results[​](#interpret-results "Direct link to Interpret results")

The judge returns a Feedback object with:

* **value**: "yes" if tool calls are efficient, "no" if otherwise

* **rationale**: Detailed explanation identifying:

  <!-- -->

  * Which specific tool calls are redundant (if any)
  * Why certain calls are considered duplicates or could be consolidated
  * Why the tool usage is efficient

## Next steps[​](#next-steps "Direct link to Next steps")

### [Evaluate tool call correctness](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call/correctness.md)

[Check if tools are called with correct arguments](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call/correctness.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call/correctness.md)

### [Evaluate agents](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

[Learn comprehensive agent evaluation techniques](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

### [Build evaluation datasets](/mlflow-website/docs/latest/genai/datasets.md)

[Create test cases for testing agent efficiency](/mlflow-website/docs/latest/genai/datasets.md)

[Learn more →](/mlflow-website/docs/latest/genai/datasets.md)
