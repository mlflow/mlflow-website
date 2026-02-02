# Tracing Microsoft Agent Framework

![Microsoft Agent Framework Tracing](/mlflow-website/docs/latest/images/llms/tracing/microsoft-agent-framework-tracing.png)

[MLflow Tracing](/mlflow-website/docs/latest/genai/tracing.md) provides automatic tracing capability for [Microsoft Agent Framework](https://github.com/microsoft/agent-framework?tab=readme-ov-file), a flexible and modular AI agents framework developed by Microsoft. MLflow supports tracing for Microsoft Agent Framework through the [OpenTelemetry](/mlflow-website/docs/latest/genai/tracing/opentelemetry.md) integration.

## Step 1: Install libraries[​](#step-1-install-libraries "Direct link to Step 1: Install libraries")

bash

```bash
pip install 'mlflow[genai]>=3.6.0' agent-framework opentelemetry-exporter-otlp-proto-http

```

## Step 2: Start the MLflow Tracking Server[​](#step-2-start-the-mlflow-tracking-server "Direct link to Step 2: Start the MLflow Tracking Server")

Start the MLflow Tracking Server with a SQL-based backend store:

bash

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

```

This example uses SQLite as the backend store. To use other types of SQL databases such as PostgreSQL, MySQL, and MSSQL, change the store URI as described in the [backend store documentation](/mlflow-website/docs/latest/self-hosting/architecture/backend-store.md). OpenTelemetry ingestion is not supported with file-based backend stores.

## Step 3: Configure OpenTelemetry[​](#step-3-configure-opentelemetry "Direct link to Step 3: Configure OpenTelemetry")

Configure the OpenTelemetry tracer to export traces to the MLflow Tracking Server endpoint.

* Set the endpoint to the MLflow Tracking Server's `/v1/traces` endpoint (OTLP).
* Set the `x-mlflow-experiment-id` header to the MLflow experiment ID. If you don't have an experiment ID, create it from Python SDK or the MLflow UI.

python

```python
from agent_framework.observability import setup_observability
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Create the OTLP span exporter with endpoint and headers
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_ID = "1234567890"
OTEL_EXPORTER_OTLP_ENDPOINT = f"{MLFLOW_TRACKING_URI}/v1/traces"
OTEL_EXPORTER_OTLP_HEADERS = {"x-mlflow-experiment-id": MLFLOW_EXPERIMENT_ID}

exporter = OTLPSpanExporter(
    endpoint=OTEL_EXPORTER_OTLP_ENDPOINT, headers=OTEL_EXPORTER_OTLP_HEADERS
)
# enable_sensitive_data=True is required for recording LLM inputs and outputs.
setup_observability(enable_sensitive_data=True, exporters=[exporter])

```

## Step 4: Run the Agent[​](#step-4-run-the-agent "Direct link to Step 4: Run the Agent")

Define and invoke the agent in a Python script like `agent.py` as usual. Microsoft Agent Framework will generate traces for your agent and send them to the MLflow Tracking Server endpoint.

python

```python
import asyncio
from pydantic import Field
from random import randint
from typing import Annotated

from agent_framework.openai import OpenAIAssistantsClient


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def main():
    async with OpenAIAssistantsClient(model_id="gpt-4o-mini").create_agent(
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    ) as agent:
        query = "What's the weather like in Seattle?"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Agent: {result}\n")


# Comment this out if you are using notebook.
if __name__ == "__main__":
    asyncio.run(main())

```

Run the script to invoke the agent.

bash

```bash
python agent.py

```

Open the MLflow UI at `http://localhost:5000` and navigate to the experiment to see the traces.

## Next Steps[​](#next-steps "Direct link to Next Steps")

* [Evaluate the Agent](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md): Learn how to evaluate the agent's performance.
* [Manage Prompts](/mlflow-website/docs/latest/genai/prompt-registry.md): Learn how to manage prompts for the agent.
* [Automatic Agent Optimization](/mlflow-website/docs/latest/genai/prompt-registry/optimize-prompts.md): Learn how to automatically optimize the agent end-to-end with state-of-the-art optimization algorithms.
