# Query Endpoints

Once you've created an endpoint, you can call it through several different API styles depending on your needs.

## Viewing Usage Examples[​](#viewing-usage-examples "Direct link to Viewing Usage Examples")

To see code examples for your endpoint, navigate to the Endpoints list and click either the Use button or the endpoint name itself. This opens a modal with comprehensive usage examples tailored to your specific endpoint.

![Usage Modal](/mlflow-website/docs/latest/assets/images/usage-modal-0e1b25b064247119b5494f8641452c88.png)

The usage modal organizes examples into two categories: unified APIs that work across any provider, and passthrough APIs that expose provider-specific features.

## Unified APIs[​](#unified-apis "Direct link to Unified APIs")

Unified APIs provide a consistent interface regardless of the underlying model provider. These APIs make it easy to switch between different models or providers without changing your application code.

### MLflow Invocations API[​](#mlflow-invocations-api "Direct link to MLflow Invocations API")

The MLflow Invocations API is the native interface for calling gateway endpoints. This API seamlessly handles model switching and advanced routing features like traffic splitting and fallbacks:

* cURL
* Python

bash

```bash
curl -X POST http://localhost:5000/gateway/my-endpoint/mlflow/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }'

```

python

```python
import requests

response = requests.post(
    "http://localhost:5000/gateway/my-endpoint/mlflow/invocations",
    json={"messages": [{"role": "user", "content": "Hello!"}], "temperature": 0.7},
)
print(response.json())

```

#### API Specification[​](#api-specification "Direct link to API Specification")

The MLflow Invocations API supports both OpenAI-style chat completions and embeddings endpoints.

**Endpoint URL Pattern:**

text

```text
POST /gateway/{endpoint_name}/mlflow/invocations

```

**Chat Completions Request Body:**

The request body follows the OpenAI chat completions format with these supported parameters. See [OpenAI Chat Completions API Reference](https://platform.openai.com/docs/api-reference/chat) for complete documentation.

| Parameter           | Type    | Required | Description                                                                                                           |
| ------------------- | ------- | -------- | --------------------------------------------------------------------------------------------------------------------- |
| `messages`          | array   | Yes      | Array of message objects with `role` and `content` fields                                                             |
| `temperature`       | number  | No       | Sampling temperature between 0 and 2. Higher values make output more random.                                          |
| `max_tokens`        | integer | No       | Maximum number of tokens to generate.                                                                                 |
| `top_p`             | number  | No       | Nucleus sampling parameter between 0 and 1. Alternative to temperature.                                               |
| `n`                 | integer | No       | Number of completions to generate. Default is 1.                                                                      |
| `stream`            | boolean | No       | Whether to stream responses.                                                                                          |
| `stream_options`    | object  | No       | Options for streaming responses.                                                                                      |
| `stop`              | array   | No       | List of sequences where the API will stop generating tokens.                                                          |
| `presence_penalty`  | number  | No       | Penalizes new tokens based on presence in text so far. Range: -2.0 to 2.0.                                            |
| `frequency_penalty` | number  | No       | Penalizes new tokens based on frequency in text so far. Range: -2.0 to 2.0.                                           |
| `tools`             | array   | No       | List of tools the model can call. Each tool includes `type`, `function` with `name`, `description`, and `parameters`. |
| `response_format`   | object  | No       | Format for the model output. Can specify "text", "json\_object", or "json\_schema" with schema definition.            |

**Response Format:**

json

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-5",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I assist you today?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}

```

**Streaming Responses:**

When `stream: true` is set, the response is sent as Server-Sent Events (SSE):

text

```text
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-5","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-5","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: [DONE]

```

**Embeddings Request Body:**

For embeddings endpoints, the request body follows the OpenAI embeddings format. See [OpenAI Embeddings API Reference](https://platform.openai.com/docs/api-reference/embeddings) for complete documentation.

| Parameter         | Type            | Required | Description                                                          |
| ----------------- | --------------- | -------- | -------------------------------------------------------------------- |
| `input`           | string or array | Yes      | Input text(s) to embed. Can be a single string or array of strings.  |
| `encoding_format` | string          | No       | Format to return embeddings. Options: "float" (default) or "base64". |

**Embeddings Response Format:**

json

```json
{
  "object": "list",
  "data": [{
    "object": "embedding",
    "embedding": [0.0023064255, -0.009327292, ...],
    "index": 0
  }],
  "model": "text-embedding-ada-002",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}

```

### OpenAI-Compatible Chat Completions API[​](#openai-compatible-chat-completions-api "Direct link to OpenAI-Compatible Chat Completions API")

For teams already using the OpenAI chat completion style APIs, the gateway provides an OpenAI-compatible interface. Simply point your OpenAI client to the gateway's base URL and use your endpoint name as the model parameter. This lets you leverage existing OpenAI-based code while gaining the gateway's routing capabilities.

See [OpenAI Chat Completions API Reference](https://platform.openai.com/docs/api-reference/chat) for complete documentation.

* cURL
* Python

bash

```bash
curl -X POST http://localhost:5000/gateway/mlflow/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-endpoint",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }'

```

python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5000/gateway/mlflow/v1",
    api_key="",  # API key not needed, configured server-side
)

response = client.chat.completions.create(
    model="my-endpoint",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
)
print(response.choices[0].message.content)

```

## Passthrough APIs[​](#passthrough-apis "Direct link to Passthrough APIs")

The Passthrough API relays requests to the provider's LLM endpoint using its native formats, allowing you to use their native client SDKs with the MLflow Gateway. While unified APIs work for most use cases, passthrough APIs give you full access to provider-specific features that may not be available through the unified interface.

For detailed information on passthrough APIs for each provider, see [Model Providers](/mlflow-website/docs/latest/genai/governance/ai-gateway/endpoints/model-providers.md).

### OpenAI Passthrough[​](#openai-passthrough "Direct link to OpenAI Passthrough")

The OpenAI passthrough API exposes the full OpenAI API including Chat Completions, Embeddings, and Responses endpoints. See [OpenAI API Reference](https://platform.openai.com/docs/api-reference) for complete documentation.

* cURL
* Python

bash

```bash
curl -X POST http://localhost:5000/gateway/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-endpoint",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

```

python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5000/gateway/openai/v1",
    api_key="dummy",  # API key not needed, configured server-side
)

response = client.chat.completions.create(
    model="my-endpoint", messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

```

### Anthropic Passthrough[​](#anthropic-passthrough "Direct link to Anthropic Passthrough")

Access Anthropic's Messages API directly through the gateway. See [Anthropic API Reference](https://docs.anthropic.com/en/api) for complete documentation.

* cURL
* Python

bash

```bash
curl -X POST http://localhost:5000/gateway/anthropic/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-endpoint",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

```

python

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:5000/gateway/anthropic",
    api_key="dummy",  # API key not needed, configured server-side
)

response = client.messages.create(
    model="my-endpoint",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.content[0].text)

```

### Google Gemini Passthrough[​](#google-gemini-passthrough "Direct link to Google Gemini Passthrough")

The Gemini passthrough API follows Google's API structure. See [Google Gemini API Reference](https://ai.google.dev/gemini-api/docs) for complete documentation.

* cURL
* Python

bash

```bash
curl -X POST http://localhost:5000/gateway/gemini/v1beta/models/my-endpoint:generateContent \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{
      "parts": [{"text": "Hello!"}]
    }]
  }'

```

python

```python
from google import genai

client = genai.Client(
    api_key="dummy",
    http_options={
        "base_url": "http://localhost:5000/gateway/gemini",
    },
)

response = client.models.generate_content(
    model="my-endpoint",
    contents={"text": "Hello!"},
)
client.close()
print(response.candidates[0].content.parts[0].text)

```

## Framework Integrations[​](#framework-integrations "Direct link to Framework Integrations")

The MLflow AI Gateway's OpenAI-compatible API makes it easy to integrate with popular LLM frameworks. Simply point your framework to the gateway's base URL and use your endpoint name as the model.

* LiteLLM
* LangChain
* LangGraph
* DSPy
* OpenAI Agents SDK

python

```python
import litellm

response = litellm.completion(
    model="openai/my-endpoint",
    messages=[{"role": "user", "content": "Hello!"}],
    api_base="http://localhost:5000/gateway/mlflow/v1",
    api_key="not-needed",
)
print(response.choices[0].message.content)

```

python

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="my-endpoint",
    base_url="http://localhost:5000/gateway/mlflow/v1",
    api_key="not-needed",
)
response = llm.invoke("Hello!")
print(response.content)

```

python

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(
    model="my-endpoint",
    base_url="http://localhost:5000/gateway/mlflow/v1",
    api_key="not-needed",
)
graph = create_react_agent(llm, tools=[])
result = graph.invoke({"messages": [{"role": "user", "content": "Hello!"}]})
print(result["messages"][-1].content)

```

python

```python
import dspy

lm = dspy.LM(
    model="openai/my-endpoint",
    api_base="http://localhost:5000/gateway/mlflow/v1",
    api_key="not-needed",
)
dspy.configure(lm=lm)
program = dspy.Predict("question -> answer")
print(program(question="What is MLflow?").answer)

```

python

```python
import openai
from agents import Agent, Runner, set_default_openai_client

client = openai.AsyncOpenAI(
    base_url="http://localhost:5000/gateway/openai/v1",
    api_key="not-needed",
)
set_default_openai_client(client)

agent = Agent(name="Assistant", instructions="You are helpful.", model="my-endpoint")
result = await Runner.run(agent, input="Hello!")
print(result.final_output)

```

## Using Gateway Endpoints with MLflow Judges[​](#using-gateway-endpoints-with-mlflow-judges "Direct link to Using Gateway Endpoints with MLflow Judges")

AI Gateway endpoints can be used as the backing LLM for MLflow's [LLM Judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md#llms-as-judges). This allows you to run judge evaluations through the gateway, benefiting from centralized API key management and cost tracking.

To use a gateway endpoint as a judge model, use the `gateway:/` prefix followed by your endpoint name:

* Built-in Judges
* Custom Judges

python

```python
from mlflow.genai.scorers import Correctness

# Use a gateway endpoint for the Correctness judge
scorer = Correctness(model="gateway:/my-chat-endpoint")

```

python

```python
from mlflow.genai.judges import make_judge
from typing import Literal

# Create a custom judge using a gateway endpoint
coherence_judge = make_judge(
    name="coherence",
    instructions=(
        "Evaluate if the response is coherent and maintains a clear flow.\n"
        "Question: {{ inputs }}\n"
        "Response: {{ outputs }}\n"
    ),
    feedback_value_type=Literal["coherent", "somewhat coherent", "incoherent"],
    model="gateway:/my-chat-endpoint",
)

```

For more details on creating and using LLM judges, see the [LLM Judges documentation](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md#llms-as-judges).
