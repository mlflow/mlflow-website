# Tracing Google Agent Development Kit (ADK)

![Google ADK Tracing](/mlflow-website/docs/latest/images/llms/tracing/google-adk-tracing.png)

[MLflow Tracing](/mlflow-website/docs/latest/genai/tracing.md) provides automatic tracing capability for [Google ADK](https://google.github.io/adk-docs/), a flexible and modular AI agents framework developed by Google. MLflow supports tracing for Google ADK through the [OpenTelemetry](/mlflow-website/docs/latest/genai/tracing/opentelemetry.md) integration.

## Step 1: Install libraries[​](#step-1-install-libraries "Direct link to Step 1: Install libraries")

bash

```bash
pip install mlflow>=3.6.0 google-adk opentelemetry-exporter-otlp-proto-http

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

bash

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5000/v1/traces
export OTEL_EXPORTER_OTLP_HEADERS=x-mlflow-experiment-id=123

```

## Step 4: Run the Agent[​](#step-4-run-the-agent "Direct link to Step 4: Run the Agent")

Define and invoke the agent in a Python script like `my_agent/agent.py` as usual. Google ADK will generate traces for your agent and send them to the MLflow Tracking Server endpoint. To enable tracing for Google ADK and send traces to MLflow, set up the OpenTelemetry tracer provider with the `OTLPSpanExporter` before running the agent.

python

```python
# my_agent/agent.py
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Configure the tracer provider and add the exporter
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(tracer_provider)


def calculator(a: float, b: float) -> str:
    """Add two numbers and return the result.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    return str(a + b)


calculator_tool = FunctionTool(func=calculator)

root_agent = LlmAgent(
    name="MathAgent",
    model="gemini-2.0-flash-exp",
    instruction=(
        "You are a helpful assistant that can do math. "
        "When asked a math problem, use the calculator tool to solve it."
    ),
    tools=[calculator_tool],
)

```

Run the agent with the `adk run` command or the web UI.

bash

```bash
adk run my_agent

```

Open the MLflow UI at `http://localhost:5000` and navigate to the experiment to see the traces.

## Next Steps[​](#next-steps "Direct link to Next Steps")

* [Evaluate the Agent](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md): Learn how to evaluate the agent's performance.
* [Manage Prompts](/mlflow-website/docs/latest/genai/prompt-registry.md): Learn how to manage prompts for the agent.
* [Automatic Agent Optimization](/mlflow-website/docs/latest/genai/prompt-registry/optimize-prompts.md): Learn how to automatically optimize the agent end-to-end with state-of-the-art optimization algorithms.
