# AI Gateway Server Usage

<!-- -->

Learn how to query your AI Gateway endpoints, integrate with applications, and leverage different APIs and tools.

## Basic Querying[​](#basic-querying "Direct link to Basic Querying")

### REST API Requests[​](#rest-api-requests "Direct link to REST API Requests")

The gateway exposes REST endpoints that follow OpenAI-compatible patterns. Each endpoint / route accepts JSON payloads and returns structured responses. Use these when integrating with applications that don't have MLflow client libraries:

bash

```bash
# Chat completions
curl -X POST http://localhost:5000/gateway/chat/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'

# Text completions
curl -X POST http://localhost:5000/gateway/completions/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_tokens": 100
  }'

# Embeddings
curl -X POST http://localhost:5000/gateway/embeddings/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Text to embed"
  }'

```

### Query Parameters[​](#query-parameters "Direct link to Query Parameters")

These parameters control model behavior and are supported across most providers. Different models may support different subsets of these parameters:

#### Chat Completions[​](#chat-completions "Direct link to Chat Completions")

json

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
  ],
  "temperature": 0.7,
  "max_tokens": 150,
  "top_p": 0.9,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stop": ["\n\n"],
  "stream": false
}

```

#### Text Completions[​](#text-completions "Direct link to Text Completions")

json

```json
{
  "prompt": "Once upon a time",
  "temperature": 0.8,
  "max_tokens": 100,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stop": [".", "!"],
  "stream": false
}

```

#### Embeddings[​](#embeddings "Direct link to Embeddings")

json

```json
{
  "input": ["Text to embed", "Another text"],
  "encoding_format": "float"
}

```

### Streaming Responses[​](#streaming-responses "Direct link to Streaming Responses")

Enable streaming for real-time response generation:

bash

```bash
curl -X POST http://localhost:5000/gateway/chat/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a story"}],
    "stream": true
  }'

```

## Python Client Integration[​](#python-client-integration "Direct link to Python Client Integration")

### OpenAI python SDK client (Recommended)[​](#openai-python-sdk-client-recommended "Direct link to OpenAI python SDK client (Recommended)")

MLflow gateway allows developers to use serving models through OpenAI's SDK.

python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:5000/v1",
    # API key is not needed, it is configured in gateway server side.
    api_key="",
)

messages = [{"role": "user", "content": "How are you ?"}]

response = client.chat.completions.create(
    # The model name must be set to either endpoint name or route name
    # that is configured in gateway YAML file.
    model="chat",
    messages=messages,
)
print(response.choices[0].message)

```

Streaming API is also supported:

python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:5000/v1",
    # API key is not needed, it is configured in gateway server side.
    api_key="",
)

messages = [{"role": "user", "content": "How are you ?"}]

response = client.chat.completions.create(
    # The model name must be set to either endpoint name or route name
    # that is configured in gateway YAML file.
    model="chat",
    messages=messages,
    stream=True,
)

for chunk in stream:
    print(chunk)
    print(chunk.choices[0].delta)
    print("****************")

```

### MLflow Deployments Client[​](#mlflow-deployments-client "Direct link to MLflow Deployments Client")

The MLflow deployments client provides a Python interface that handles authentication, error handling, and response parsing. Use this when building Python applications:

python

```python
from mlflow.deployments import get_deploy_client

# Create a client for the gateway
client = get_deploy_client("http://localhost:5000")

# Query a chat endpoint
response = client.predict(
    endpoint="chat",
    inputs={"messages": [{"role": "user", "content": "What is MLflow?"}]},
)

print(response["choices"][0]["message"]["content"])

```

### Advanced Client Usage[​](#advanced-client-usage "Direct link to Advanced Client Usage")

Build reusable functions for common operations like streaming responses and batch embedding generation:

python

```python
from mlflow.deployments import get_deploy_client

# Initialize client
client = get_deploy_client("http://localhost:5000")


# Chat with streaming
def stream_chat(prompt):
    response = client.predict(
        endpoint="chat",
        inputs={
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": 0.7,
        },
    )

    for chunk in response:
        if chunk["choices"][0]["delta"].get("content"):
            print(chunk["choices"][0]["delta"]["content"], end="")


# Generate embeddings
def get_embeddings(texts):
    response = client.predict(endpoint="embeddings", inputs={"input": texts})
    return [item["embedding"] for item in response["data"]]


# Example usage
stream_chat("Explain quantum computing")
embeddings = get_embeddings(["Hello world", "MLflow AI Gateway"])

```

### Error Handling[​](#error-handling "Direct link to Error Handling")

Proper error handling helps you distinguish between network issues, authentication problems, and model-specific errors:

python

```python
from mlflow.deployments import get_deploy_client
from mlflow.exceptions import MlflowException

client = get_deploy_client("http://localhost:5000")

try:
    response = client.predict(
        endpoint="chat", inputs={"messages": [{"role": "user", "content": "Hello"}]}
    )
    print(response)
except MlflowException as e:
    print(f"MLflow error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

```

## Streaming Responses[​](#streaming-responses-1 "Direct link to Streaming Responses")

For long-form content generation, enable streaming to receive partial responses as they're generated instead of waiting for the complete response:

bash

```bash
curl -X POST http://localhost:5000/gateway/chat/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a story"}],
    "stream": true
  }'

```

## API Reference[​](#api-reference "Direct link to API Reference")

### Gateway Management[​](#gateway-management "Direct link to Gateway Management")

Query the gateway's current configuration and available endpoints programmatically:

python

```python
from mlflow.deployments import get_deploy_client

client = get_deploy_client("http://localhost:5000")

# List available endpoints
endpoints = client.list_endpoints()
for endpoint in endpoints:
    print(f"Endpoint: {endpoint['name']}")

# Get endpoint details
endpoint_info = client.get_endpoint("chat")
print(f"Model: {endpoint_info.get('model', {}).get('name', 'N/A')}")
print(f"Provider: {endpoint_info.get('model', {}).get('provider', 'N/A')}")

# Note: Route creation, updates, and deletion are typically done
# through configuration file changes, not programmatically

```

### Health Monitoring[​](#health-monitoring "Direct link to Health Monitoring")

Monitor gateway availability and responsiveness for production deployments:

python

```python
import requests

try:
    response = requests.get("http://localhost:5000/health")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("Gateway is healthy")
except requests.RequestException as e:
    print(f"Health check failed: {e}")

```

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [Gateway Overview](/mlflow-website/docs/latest/genai/governance/ai-gateway.md)

[Return to the AI Gateway overview page](/mlflow-website/docs/latest/genai/governance/ai-gateway.md)

[View overview →](/mlflow-website/docs/latest/genai/governance/ai-gateway.md)

### [Configuration Guide](/mlflow-website/docs/latest/genai/governance/ai-gateway/legacy/configuration.md)

[Learn how to configure providers and advanced settings](/mlflow-website/docs/latest/genai/governance/ai-gateway/legacy/configuration.md)

[Configure providers →](/mlflow-website/docs/latest/genai/governance/ai-gateway/legacy/configuration.md)

### [Setup Guide](/mlflow-website/docs/latest/genai/governance/ai-gateway/legacy/setup.md)

[Get started with installation and environment setup](/mlflow-website/docs/latest/genai/governance/ai-gateway/legacy/setup.md)

[View setup →](/mlflow-website/docs/latest/genai/governance/ai-gateway/legacy/setup.md)
