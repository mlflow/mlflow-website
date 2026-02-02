# Tracing Amazon Bedrock with MLflow

MLflow supports automatic tracing for Amazon Bedrock, a fully managed service on AWS that provides high-performing foundations from leading AI providers such as Anthropic, Cohere, Meta, Mistral AI, and more. By enabling auto tracing for Amazon Bedrock by calling the [`mlflow.bedrock.autolog()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.bedrock.html#mlflow.bedrock.autolog) function, MLflow will capture traces for LLM invocation and log them to the active MLflow Experiment.

![Bedrock DIY Agent Tracing](/mlflow-website/docs/latest/images/llms/tracing/bedrock-tracing-agent.png)

MLflow trace automatically captures the following information about Amazon Bedrock calls:

* Prompts and completion responses
* Latencies
* Model name
* Additional metadata such as temperature, max\_tokens, if specified.
* Function calling if returned in the response
* Any exception if raised
* and more...

## Getting Started[​](#getting-started "Direct link to Getting Started")

1

### Install Dependencies

bash

```bash
pip install 'mlflow[genai]' boto3

```

2

### Start MLflow Server

* Local (pip)
* Local (docker)

If you have a local Python environment >= 3.10, you can start the MLflow server locally using the `mlflow` CLI command.

bash

```bash
mlflow server

```

MLflow also provides a Docker Compose file to start a local MLflow server with a postgres database and a minio server.

bash

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/mlflow/mlflow.git
cd mlflow
git sparse-checkout set docker-compose
cd docker-compose
cp .env.dev.example .env
docker compose up -d

```

Refer to the [instruction](https://github.com/mlflow/mlflow/tree/master/docker-compose/README.md) for more details, e.g., overriding the default environment variables.

3

### Enable Tracing and Make API Calls

Enable tracing with `mlflow.bedrock.autolog()` and invoke Bedrock as usual using the boto3 runtime client. Ensure your AWS credentials and region are configured (e.g., `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`).

python

```python
import boto3
import mlflow

# Enable auto-tracing for Amazon Bedrock
mlflow.bedrock.autolog()
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Bedrock")

# Create a boto3 client for invoking the Bedrock API
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="<REPLACE_WITH_YOUR_AWS_REGION>",
)

# Make a standard Bedrock call
response = bedrock.converse(
    modelId="anthropic.claude-3-7-sonnet-20250219-v1:0",
    messages=[
        {
            "role": "user",
            "content": "Describe the purpose of a 'hello world' program in one line.",
        },
    ],
    inferenceConfig={
        "maxTokens": 512,
        "temperature": 0.1,
        "topP": 0.9,
    },
)

```

4

### View Traces in MLflow UI

Open the MLflow UI at <http://localhost:5000> (or your MLflow server URL) to see the trace for your Bedrock API calls.

![Bedrock Trace](/mlflow-website/docs/latest/images/llms/tracing/bedrock-tracing-stream.png)

## Supported APIs[​](#supported-apis "Direct link to Supported APIs")

MLflow supports automatic tracing for the following Amazon Bedrock APIs:

* [converse](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html)
* [converse\_stream](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse_stream.html)
* [invoke\_model](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/invoke_model.html)
* [invoke\_model\_with\_response\_stream](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/invoke_model_with_response_stream.html)

## Raw Inputs and Outputs[​](#raw-inputs-and-outputs "Direct link to Raw Inputs and Outputs")

By default, MLflow renders the rich chat-like UI for input and output messages in the `Chat` tab. To view the raw input and output payload, including configuration parameters, click on the `Inputs / Outputs` tab in the UI.

note

The `Chat` panel is only supported for the `converse` and `converse_stream` APIs. For the other APIs, MLflow only displays the `Inputs / Outputs` tab.

## Token Usage[​](#token-usage "Direct link to Token Usage")

MLflow automatically captures token usage statistics for supported Bedrock models and APIs. The token usage for each LLM call will be logged in the `mlflow.chat.tokenUsage` attribute. The total token usage throughout the trace will be available in the `token_usage` field of the trace info object.

Token usage includes:

* **Input tokens** (prompt tokens)
* **Output tokens** (completion/generation tokens)
* **Total tokens** (sum of input and output)

Token usage is extracted from the response for all major Bedrock providers, including:

* Anthropic (Claude)
* AI21 (Jamba)
* Amazon Titan/Nova
* Meta Llama

#### Supported APIs[​](#supported-apis-1 "Direct link to Supported APIs")

Token usage is logged for:

* `invoke_model`
* `invoke_model_with_response_stream`
* `converse`
* `converse_stream`

python

```python
import boto3
import mlflow

mlflow.bedrock.autolog()

# Create a boto3 client for invoking the Bedrock API
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="<REPLACE_WITH_YOUR_AWS_REGION>",
)

# Use the converse method to create a new message
response = bedrock.converse(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of low latency LLMs.",
        }
    ],
    inferenceConfig={
        "maxTokens": 512,
        "temperature": 0.1,
        "topP": 0.9,
    },
)

# Get the trace object just created
last_trace_id = mlflow.get_last_active_trace_id()
trace = mlflow.get_trace(trace_id=last_trace_id)

# Print the token usage
total_usage = trace.info.token_usage
print("== Total token usage: ==")
print(f" Input tokens: {total_usage['input_tokens']}")
print(f" Output tokens: {total_usage['output_tokens']}")
print(f" Total tokens: {total_usage['total_tokens']}")

# Print the token usage for each LLM call
print("\n== Detailed usage for each LLM call: ==")
for span in trace.data.spans:
    if usage := span.get_attribute("mlflow.chat.tokenUsage"):
        print(f"{span.name}:")
        print(f" Input tokens: {usage['input_tokens']}")
        print(f" Output tokens: {usage['output_tokens']}")
        print(f" Total tokens: {usage['total_tokens']}")

```

If a provider or model does not return usage information, this attribute will be omitted.

## Streaming[​](#streaming "Direct link to Streaming")

MLflow supports tracing streaming calls to Amazon Bedrock APIs. The generated trace shows the aggregated output message in the `Chat` tab, while the individual chunks are displayed in the `Events` tab.

python

```python
response = bedrock.converse_stream(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[
        {
            "role": "user",
            "content": [
                {"text": "Describe the purpose of a 'hello world' program in one line."}
            ],
        }
    ],
    inferenceConfig={
        "maxTokens": 300,
        "temperature": 0.1,
        "topP": 0.9,
    },
)

for chunk in response["stream"]:
    print(chunk)

```

![Bedrock Stream Tracing](/mlflow-website/docs/latest/images/llms/tracing/bedrock-tracing-stream.png)

warning

MLflow does not create a span immediately when the streaming response is returned. Instead, it creates a span when the streaming chunks are **consumed**, for example, the for-loop in the code snippet above.

## Function Calling Agent[​](#function-calling-agent "Direct link to Function Calling Agent")

MLflow Tracing automatically captures function calling metadata when calling Amazon Bedrock APIs. The function definition and instruction in the response will be highlighted in the `Chat` tab on trace UI.

Combining this with the manual tracing feature, you can define a function-calling agent (ReAct) and trace its execution. The entire agent implementation might look complicated, but the tracing part is pretty straightforward: (1) add the `@mlflow.trace` decorator to functions to trace and (2) enable auto-tracing for Amazon Bedrock with `mlflow.bedrock.autolog()`. MLflow will take care of the complexity such as resolving call chains and recording execution metadata.

python

```python
import boto3
import mlflow
from mlflow.entities import SpanType

# Enable auto-tracing for Amazon Bedrock
mlflow.bedrock.autolog()
mlflow.set_experiment("Bedrock")
# Create a boto3 client for invoking the Bedrock API
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="<REPLACE_WITH_YOUR_AWS_REGION>",
)
model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"


# Define the tool function. Decorate it with `@mlflow.trace` to create a span for its execution.
@mlflow.trace(span_type=SpanType.TOOL)
def get_weather(city: str) -> str:
    """ "Get the current weather in a given location"""
    return "sunny" if city == "San Francisco, CA" else "unknown"


# Define the tool configuration passed to Bedrock
tools = [
    {
        "toolSpec": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city and state, e.g., San Francisco, CA",
                        },
                    },
                    "required": ["city"],
                }
            },
        }
    }
]
tool_functions = {"get_weather": get_weather}


# Define a simple tool calling agent
@mlflow.trace(span_type=SpanType.AGENT)
def run_tool_agent(question: str) -> str:
    messages = [{"role": "user", "content": [{"text": question}]}]
    # Invoke the model with the given question and available tools
    response = bedrock.converse(
        modelId=model_id,
        messages=messages,
        toolConfig={"tools": tools},
    )
    assistant_message = response["output"]["message"]
    messages.append(assistant_message)
    # If the model requests tool call(s), invoke the function with the specified arguments
    tool_use = next(
        (c["toolUse"] for c in assistant_message["content"] if "toolUse" in c), None
    )
    if tool_use:
        tool_func = tool_functions[tool_use["name"]]
        tool_result = tool_func(**tool_use["input"])
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": tool_use["toolUseId"],
                            "content": [{"text": tool_result}],
                        }
                    }
                ],
            }
        )
        # Send the tool results to the model and get a new response
        response = bedrock.converse(
            modelId=model_id,
            messages=messages,
            toolConfig={"tools": tools},
        )
    return response["output"]["message"]["content"][0]["text"]


# Run the tool calling agent
question = "What's the weather like in San Francisco today?"
answer = run_tool_agent(question)

```

Executing the code above will create a single trace that involves all LLM invocations and the tool calls.

![Bedrock DIY Agent Tracing](/mlflow-website/docs/latest/assets/images/bedrock-tracing-agent-cae1bcf40457074afa5bfde0b05b292e.png)

## Disable auto-tracing[​](#disable-auto-tracing "Direct link to Disable auto-tracing")

Auto tracing for Amazon Bedrock can be disabled globally by calling `mlflow.bedrock.autolog(disable=True)` or `mlflow.autolog(disable=True)`.

## Next steps[​](#next-steps "Direct link to Next steps")

### [Track User Feedback](/mlflow-website/docs/latest/genai/tracing/collect-user-feedback.md)

[Record user feedback on traces for tracking user satisfaction.](/mlflow-website/docs/latest/genai/tracing/collect-user-feedback.md)

[Learn about feedback →](/mlflow-website/docs/latest/genai/tracing/collect-user-feedback.md)

### [Manage Prompts](/mlflow-website/docs/latest/genai/prompt-registry.md)

[Learn how to manage prompts with MLflow's prompt registry.](/mlflow-website/docs/latest/genai/prompt-registry.md)

[Manage prompts →](/mlflow-website/docs/latest/genai/prompt-registry.md)

### [Evaluate Traces](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

[Evaluate traces with LLM judges to understand and improve your AI application's behavior.](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

[Evaluate traces →](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)
