# ToolCallCorrectness Judge

The `ToolCallCorrectness` judge evaluates whether the tools called by an agent and the arguments they are called with are correct given the user request.

This built-in LLM judge is designed for evaluating AI agents and tool-calling applications where you need to ensure the agent selects appropriate tools and provides correct arguments to fulfill the user's request.

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

## Evaluation modes[​](#evaluation-modes "Direct link to Evaluation modes")

The `ToolCallCorrectness` judge supports three modes of evaluation:

1. **Ground-truth free (default)**: When no expectations are provided, uses an LLM to judge whether tool calls are reasonable given the user request and available tools.

2. **With expectations (fuzzy match)**: When expectations are provided and `should_exact_match=False`, uses an LLM to semantically compare actual tool calls against expected tool calls.

3. **With expectations (exact match)**: When expectations are provided and `should_exact_match=True`, performs direct comparison of tool names and arguments.

## Usage examples[​](#usage-examples "Direct link to Usage examples")

The `ToolCallCorrectness` judge can be invoked directly for single trace assessment or used with MLflow's evaluation framework for batch evaluation.

**Requirements:**

* **Trace requirements**: - The MLflow Trace must contain at least one span with `span_type` set to `TOOL`
* **Ground-truth labels**: Optional - can provide `expected_tool_calls` in the expectations dictionary for comparison

- Invoke directly
- Invoke with evaluate()

python

```python
from mlflow.genai.scorers import ToolCallCorrectness
import mlflow

# Get a trace from a previous run
trace = mlflow.get_trace("<your-trace-id>")

# Assess if tool calls are correct (ground-truth free mode)
feedback = ToolCallCorrectness(name="my_tool_call_correctness")(trace=trace)
print(feedback)

```

python

```python
import mlflow
from mlflow.genai.scorers import ToolCallCorrectness

# Evaluate traces from previous runs
results = mlflow.genai.evaluate(
    data=traces,  # DataFrame or list containing trace data
    scorers=[ToolCallCorrectness()],
)

```

tip

For a complete agent example with this judge, see the [Tool Call Evaluation guide](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call.md).

## Using expectations for comparison[​](#using-expectations-for-comparison "Direct link to Using expectations for comparison")

You can provide expected tool calls to compare against the actual tool calls made by the agent.

### Parameters[​](#parameters "Direct link to Parameters")

| Parameter                  | Type   | Default | Description                                                                                                                                                                                    |
| -------------------------- | ------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `should_exact_match`       | `bool` | `False` | Controls comparison mode when expectations are provided. If `False`, uses LLM for semantic comparison of tool calls. If `True`, performs direct string comparison of tool names and arguments. |
| `should_consider_ordering` | `bool` | `False` | Whether to enforce the order of tool calls when comparing against expectations. If `True`, tool calls must match the expected order. If `False`, order is ignored.                             |

### Fuzzy matching (default)[​](#fuzzy-matching-default "Direct link to Fuzzy matching (default)")

With fuzzy matching, the LLM semantically compares actual tool calls against expected ones:

python

```python
from mlflow.genai.scorers import ToolCallCorrectness

# Define expected tool calls
eval_dataset = [
    {
        "inputs": {"query": "What's the weather in San Francisco?"},
        "expectations": {
            "expected_tool_calls": [
                {"name": "get_weather", "arguments": {"location": "San Francisco, CA"}},
            ]
        },
    },
    {
        "inputs": {"query": "What's the weather in Tokyo?"},
        "expectations": {
            "expected_tool_calls": [
                {"name": "get_weather", "arguments": {"location": "Tokyo, Japan"}},
            ]
        },
    },
]

# Evaluate with fuzzy matching (default)
eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=weather_agent,
    scorers=[ToolCallCorrectness()],  # should_exact_match=False by default
)

```

### Exact matching[​](#exact-matching "Direct link to Exact matching")

With exact matching, tool names and arguments are compared directly:

python

```python
from mlflow.genai.scorers import ToolCallCorrectness

# Define expected tool calls
eval_dataset = [
    {
        "inputs": {"query": "What's the weather in San Francisco?"},
        "expectations": {
            "expected_tool_calls": [
                {"name": "get_weather", "arguments": {"location": "San Francisco, CA"}},
            ]
        },
    },
]

# Use exact matching for stricter comparison
scorer = ToolCallCorrectness(should_exact_match=True)

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=weather_agent,
    scorers=[scorer],
)

```

### Partial expectations (names only)[​](#partial-expectations-names-only "Direct link to Partial expectations (names only)")

You can provide only tool names without arguments to check that the correct tools are called:

python

```python
from mlflow.genai.scorers import ToolCallCorrectness

eval_dataset = [
    {
        "inputs": {"query": "What's the weather in Tokyo?"},
        "expectations": {
            "expected_tool_calls": [
                {"name": "get_weather"},  # Only check tool name
            ]
        },
    },
]

scorer = ToolCallCorrectness(should_exact_match=True)
eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=weather_agent,
    scorers=[scorer],
)

```

## Considering tool call ordering[​](#considering-tool-call-ordering "Direct link to Considering tool call ordering")

By default, the judge ignores the order of tool calls. To enforce ordering:

python

```python
from mlflow.genai.scorers import ToolCallCorrectness

# Enforce that tools are called in the expected order
scorer = ToolCallCorrectness(
    should_exact_match=True,
    should_consider_ordering=True,
)

# Example with multiple expected tool calls
eval_dataset = [
    {
        "inputs": {"query": "Get weather for Paris and then for London"},
        "expectations": {
            "expected_tool_calls": [
                {"name": "get_weather", "arguments": {"location": "Paris"}},
                {"name": "get_weather", "arguments": {"location": "London"}},
            ]
        },
    },
]

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=weather_agent,
    scorers=[scorer],
)

```

## Select the LLM that powers the judge[​](#select-the-llm-that-powers-the-judge "Direct link to Select the LLM that powers the judge")

You can change the judge model by using the `model` argument in the judge definition. The model must be specified in the format `<provider>:/<model-name>`, where `<provider>` is a LiteLLM-compatible model provider.

For a list of supported models, see [selecting judge models](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges.md#selecting-judge-models).

## Interpret results[​](#interpret-results "Direct link to Interpret results")

The judge returns a Feedback object with:

* **value**: "yes" if tool calls are correct, "no" if incorrect

* **rationale**: Detailed explanation identifying:

  <!-- -->

  * Which tool calls are correct or problematic
  * Whether arguments match expectations or are reasonable
  * Why certain tool choices were appropriate or inappropriate

## Next steps[​](#next-steps "Direct link to Next steps")

### [Evaluate tool call efficiency](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call/efficiency.md)

[Check if tool calls are efficient without redundancy](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call/efficiency.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call/efficiency.md)

### [Evaluate agents](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

[Learn comprehensive agent evaluation techniques](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

### [Build evaluation datasets](/mlflow-website/docs/latest/genai/datasets.md)

[Create test cases with expected tool calls for testing](/mlflow-website/docs/latest/genai/datasets.md)

[Learn more →](/mlflow-website/docs/latest/genai/datasets.md)
