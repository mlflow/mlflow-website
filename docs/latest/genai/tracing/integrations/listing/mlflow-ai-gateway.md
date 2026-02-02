# Tracing MLflow AI Gateway

[MLflow AI Gateway](/mlflow-website/docs/latest/genai/governance/ai-gateway.md) is a unified, centralized interface for accessing multiple LLM providers. It simplifies API key management, provides a consistent API across providers, and enables seamless switching between models from OpenAI, Anthropic, Google, and other providers.

Since MLflow AI Gateway exposes an OpenAI-compatible API, you can use MLflow's automatic tracing integrations to capture detailed traces of your LLM interactions.

![MLflow AI Gateway Tracing](/mlflow-website/docs/latest/images/llms/tracing/basic-openai-trace.png)

## Integration Options[​](#integration-options "Direct link to Integration Options")

There are two ways to trace LLM calls through MLflow AI Gateway:

| Approach                              | Description                             | Best For                                                   |
| ------------------------------------- | --------------------------------------- | ---------------------------------------------------------- |
| **Server-side Tracing** (Coming Soon) | Gateway automatically logs all requests | Centralized tracing for all requests through the gateway   |
| **Client-side Tracing**               | Use OpenAI SDK with MLflow autolog      | Combining LLM traces with your agent or application traces |

Coming Soon

**Server-side tracing** for MLflow AI Gateway is not available yet. Stay tuned for updates!

## Prerequisite[​](#prerequisite "Direct link to Prerequisite")

### Start MLflow Server with AI Gateway[​](#start-mlflow-server-with-ai-gateway "Direct link to Start MLflow Server with AI Gateway")

To start MLflow server with AI Gateway, you need to install the `mlflow[genai]` package.

bash

```bash
pip install mlflow[genai]

```

Then start the MLflow server as usual, no additional configuration is needed.

bash

```bash
mlflow server

```

### Create Endpoint[​](#create-endpoint "Direct link to Create Endpoint")

Create an endpoint in MLflow AI Gateway to route requests to your LLM provider. See the [AI Gateway Quickstart](/mlflow-website/docs/latest/genai/governance/ai-gateway/quickstart.md) for detailed setup instructions.

## Query Gateway[​](#query-gateway "Direct link to Query Gateway")

You can trace LLM calls through MLflow AI Gateway using any of the following approaches:

* OpenAI SDK
* Anthropic SDK
* Gemini SDK
* LangChain

Since MLflow AI Gateway exposes an OpenAI-compatible API, you can use MLflow's [OpenAI automatic tracing](/mlflow-website/docs/latest/genai/tracing/integrations/listing/openai.md) integration to trace calls.

python

```python
import mlflow
from openai import OpenAI

# Enable auto-tracing for OpenAI
mlflow.openai.autolog()

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLflow AI Gateway")

# Point OpenAI client to MLflow AI Gateway
client = OpenAI(
    base_url="http://localhost:5000/gateway/openai/v1",
    api_key="dummy",  # API key not needed, configured server-side
)

response = client.chat.completions.create(
    model="my-endpoint", messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

```

You can use MLflow's [Anthropic automatic tracing](/mlflow-website/docs/latest/genai/tracing/integrations/listing/anthropic.md) integration to trace calls through MLflow AI Gateway.

python

```python
import mlflow
import anthropic

# Enable auto-tracing for Anthropic
mlflow.anthropic.autolog()

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLflow AI Gateway")

# Point Anthropic client to MLflow AI Gateway
client = anthropic.Anthropic(
    base_url="http://localhost:5000/gateway/anthropic",
    api_key="dummy",  # API key not needed, configured server-side
)

# Make API calls - traces will be captured automatically
response = client.messages.create(
    max_tokens=1004,
    model="<your-endpoint-name>",  # Use your endpoint name
    messages=[{"role": "user", "content": "Hello world"}],
)
print(response.content[0].text)

```

You can use MLflow's [Gemini automatic tracing](/mlflow-website/docs/latest/genai/tracing/integrations/listing/gemini.md) integration to trace calls through MLflow AI Gateway.

python

```python
import mlflow
from google import genai

# Enable auto-tracing for Gemini
mlflow.gemini.autolog()

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLflow AI Gateway")

# Point Gemini client to MLflow AI Gateway
client = genai.Client(
    http_options={"base_url": "http://localhost:5000/gateway/gemini"},
    api_key="dummy",  # API key not needed, configured server-side
)

# Make API calls - traces will be captured automatically
response = client.models.generate_content(
    model="<your-endpoint-name>",  # Use your endpoint name
    contents={"text": "Hello!"},
)
print(response.text)

```

You can use MLflow's [LangChain automatic tracing](/mlflow-website/docs/latest/genai/tracing/integrations/listing/langchain.md) integration to trace calls through MLflow AI Gateway.

python

```python
import mlflow
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# Enable auto-tracing for LangChain
mlflow.langchain.autolog()

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLflow AI Gateway")

# Point LangChain to MLflow AI Gateway
llm = ChatOpenAI(
    base_url="http://localhost:5000/gateway/mlflow/v1",
    model="<your-endpoint-name>",  # Use your endpoint name
    api_key="dummy",  # API key not needed, configured server-side
)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# Run the agent as usual
agent = create_agent(
    llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)
agent.invoke({"input": "What's the weather in San Francisco?"})

```

## View Traces in MLflow UI[​](#view-traces-in-mlflow-ui "Direct link to View Traces in MLflow UI")

Open the MLflow UI at <http://localhost:5000> (or your custom MLflow server URL) to see the traces from your MLflow AI Gateway calls.

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
