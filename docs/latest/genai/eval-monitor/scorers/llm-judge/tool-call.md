# Tool Call Evaluation with Built-in Judges

AI agents often use tools (functions) to complete tasks - from fetching data to performing calculations. Evaluating tool-calling applications requires assessing whether agents select appropriate tools and provide correct arguments to fulfill user requests.

MLflow provides built-in judges designed specifically for evaluating tool-calling agents:

## Available Tool Call Judges[​](#available-tool-call-judges "Direct link to Available Tool Call Judges")

| Judge                                                                                                            | What does it evaluate?                                       | Requires ground-truth? | Requires traces?      |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ---------------------- | --------------------- |
| [ToolCallCorrectness](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call/correctness.md) | Are the tool calls and arguments correct for the user query? | No                     | ⚠️ **Trace Required** |
| [ToolCallEfficiency](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call/efficiency.md)   | Are the tool calls efficient without redundancy?             | No                     | ⚠️ **Trace Required** |

tip

All tool call judges require MLflow Traces with at least one span marked as `span_type="TOOL"`. Use the [@mlflow.trace](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.trace) decorator with `span_type="TOOL"` on your tool functions.

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

## Complete Agent Example[​](#complete-agent-example "Direct link to Complete Agent Example")

Here's a complete example showing how to build a tool-calling agent and evaluate it with the judges:

python

```python
import json
import mlflow
import openai
from mlflow.genai.scorers import ToolCallCorrectness, ToolCallEfficiency

mlflow.openai.autolog()
client = openai.OpenAI()

# Define the tool schema for the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and country"},
                },
                "required": ["location"],
            },
        },
    },
]


# Define the tool function with proper span type
@mlflow.trace(span_type="TOOL")
def get_weather(location: str) -> dict:
    # Simulated weather data - in practice, this would call a weather API
    return {"temperature": 72, "condition": "sunny", "location": location}


# Define your agent
@mlflow.trace
def agent(query: str):
    # Call the LLM with tools
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}],
        tools=tools,
    )
    message = response.choices[0].message
    responses = []
    if message.tool_calls:
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = get_weather(**args)
            responses.append(
                {
                    "response": f"Weather in {result['location']}: {result['condition']}, {result['temperature']}°F"
                }
            )

    return {"response": responses if responses else message.content}


# Create evaluation dataset
eval_dataset = [
    {"inputs": {"query": "What's the weather like in Paris?"}},
    {"inputs": {"query": "How's the weather in Tokyo?"}},
]

# Run evaluation with tool call judges
eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=agent,
    scorers=[
        ToolCallCorrectness(model="openai:/gpt-4o-mini"),
        ToolCallEfficiency(model="openai:/gpt-4o-mini"),
    ],
)

```

### Understanding the Results[​](#understanding-the-results "Direct link to Understanding the Results")

Each tool call judge evaluates tool spans separately:

* **ToolCallCorrectness**: Assesses whether the agent selected appropriate tools and provided correct arguments
* **ToolCallEfficiency**: Evaluates whether the agent made redundant or unnecessary tool calls

## Select the LLM that powers the judge[​](#select-the-llm-that-powers-the-judge "Direct link to Select the LLM that powers the judge")

You can change the judge model by using the `model` argument in the judge definition. The model must be specified in the format `<provider>:/<model-name>`, where `<provider>` is a LiteLLM-compatible model provider.

For a list of supported models, see [selecting judge models](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges.md#selecting-judge-models).

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [ToolCallCorrectness](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call/correctness.md)

[Evaluate if tool calls and arguments are correct](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call/correctness.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call/correctness.md)

### [ToolCallEfficiency](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call/efficiency.md)

[Check for redundant or unnecessary tool calls](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call/efficiency.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call/efficiency.md)

### [Build evaluation datasets](/mlflow-website/docs/latest/genai/datasets.md)

[Create ground truth datasets for testing agents](/mlflow-website/docs/latest/genai/datasets.md)

[Learn more →](/mlflow-website/docs/latest/genai/datasets.md)
