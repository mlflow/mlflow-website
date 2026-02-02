# Tracing Pydantic AI Gateway

[Pydantic AI Gateway](https://ai.pydantic.dev/gateway/) is a unified interface for accessing multiple AI providers with a single key. It supports models from OpenAI, Anthropic, Google Vertex, Groq, AWS Bedrock, and more. Key features include spending limits, failover management, and zero translation—requests flow through directly in each provider's native format, giving you immediate access to new model features as soon as they are released.

Since Pydantic AI Gateway exposes OpenAI and Anthropic-compatible APIs, you can use MLflow's automatic tracing integrations to capture detailed traces of your LLM interactions.

![Pydantic AI Gateway Tracing](/mlflow-website/docs/latest/images/llms/pydantic-ai/pydanticai-gateway-tracing.png)

Looking for PydanticAI Agent Framework?

This guide covers tracing LLM calls through **Pydantic AI Gateway**. If you're building agents using the Pydantic AI framework directly, see the [PydanticAI Integration](/mlflow-website/docs/latest/genai/tracing/integrations/listing/pydantic_ai.md) guide instead.

## Prerequisite[​](#prerequisite "Direct link to Prerequisite")

### Start MLflow Server[​](#start-mlflow-server "Direct link to Start MLflow Server")

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

### Get Pydantic AI Gateway API Key[​](#get-pydantic-ai-gateway-api-key "Direct link to Get Pydantic AI Gateway API Key")

Create an account on [Pydantic AI Gateway](https://gateway.pydantic.dev/) to get your API key, or bring your own API key from a supported LLM provider.

## Query Gateway[​](#query-gateway "Direct link to Query Gateway")

You can trace LLM calls through Pydantic AI Gateway using any of the following approaches:

* OpenAI SDK
* Anthropic SDK
* Pydantic AI Agents

Since Pydantic AI Gateway exposes an OpenAI-compatible API, you can use MLflow's [OpenAI automatic tracing](/mlflow-website/docs/latest/genai/tracing/integrations/listing/openai.md) integration to trace calls.

python

```python
import mlflow
from openai import OpenAI

# Enable auto-tracing for OpenAI
mlflow.openai.autolog()

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Pydantic AI Gateway")

# Point OpenAI client to Pydantic AI Gateway
client = OpenAI(
    base_url="https://gateway.pydantic.dev/proxy/chat/",
    api_key="<PYDANTIC_AI_GATEWAY_API_KEY>",
)

# Make API calls - traces will be captured automatically
response = client.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content": "Hello world"}],
)
print(response.choices[0].message.content)

```

If your Pydantic AI Gateway is configured to expose an Anthropic-compatible API, you can use MLflow's [Anthropic automatic tracing](/mlflow-website/docs/latest/genai/tracing/integrations/listing/anthropic.md) integration to trace calls.

python

```python
import mlflow
import anthropic

# Enable auto-tracing for Anthropic
mlflow.anthropic.autolog()

# Set tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Pydantic AI Gateway")

# Point Anthropic client to Pydantic AI Gateway
client = anthropic.Anthropic(
    base_url="https://gateway.pydantic.dev/proxy/chat/",
    api_key="<PYDANTIC_AI_GATEWAY_API_KEY>",
)

# Make API calls - traces will be captured automatically
response = client.messages.create(
    max_tokens=1000,
    model="claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Hello world"}],
)
print(response.content[0].text)

```

You can also use the PydanticAI agent framework to interact with models through the gateway, with full tracing support.

export

```export
PYDANTIC_AI_GATEWAY_API_KEY=your-pydantic-ai-gateway-api-key
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=Pydantic AI Gateway

```

python

```python
import mlflow
from pydantic_ai import Agent

# Enable auto-tracing for PydanticAI
mlflow.pydantic_ai.autolog()

# Create an agent with the gateway model
# Use the appropriate model identifier for your gateway configuration
agent = Agent(
    "gateway/openai:gpt-4o",  # or your gateway model identifier
    system_prompt="You are a helpful assistant.",
    instrument=True,
)


# Run the agent - traces will be captured automatically
async def main():
    result = await agent.run("What is the capital of France?")
    print(result.output)


# If running in a notebook
await main()

# If running as a script
# import asyncio
# asyncio.run(main())

```

For more advanced PydanticAI features like tool calling, MCP servers, and streaming, see the [PydanticAI Integration](/mlflow-website/docs/latest/genai/tracing/integrations/listing/pydantic_ai.md) guide.

## View Traces in MLflow UI[​](#view-traces-in-mlflow-ui "Direct link to View Traces in MLflow UI")

Open the MLflow UI at <http://localhost:5000> (or your custom MLflow server URL) to see the traces from your Pydantic AI Gateway calls.

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [Track User Feedback](/mlflow-website/docs/latest/genai/tracing/collect-user-feedback.md)

[Record user feedback on traces for tracking user satisfaction.](/mlflow-website/docs/latest/genai/tracing/collect-user-feedback.md)

[Learn about feedback →](/mlflow-website/docs/latest/genai/tracing/collect-user-feedback.md)

### [Manage Prompts](/mlflow-website/docs/latest/genai/prompt-registry.md)

[Learn how to manage prompts with MLflow's prompt registry.](/mlflow-website/docs/latest/genai/prompt-registry.md)

[Manage prompts →](/mlflow-website/docs/latest/genai/prompt-registry.md)

### [Evaluate Traces](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

[Evaluate traces with LLM judges to understand and improve your AI application's behavior.](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

[Evaluate traces →](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)
