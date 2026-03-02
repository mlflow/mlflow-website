---
title: Full OpenTelemetry Support in MLflow Tracing
description: MLflow 3.6.0 brings comprehensive OpenTelemetry integration for unified observability
slug: opentelemetry-tracing-support
authors: [mlflow-maintainers]
tags: [tracing, opentelemetry, genai, mlops]
thumbnail: /img/blog/mlflow-opentelemetry-thumbnail.png
image: /img/blog/mlflow-opentelemetry-thumbnail.png
---

We're excited to announce that MLflow 3.6.0 brings **full OpenTelemetry support** to the open-source MLflow server. This integration represents a major step forward in making MLflow a **vendor-neutral observability platform** for GenAI applications.

<img
src={require("./hero.png").default}
alt="MLflow OpenTelemetry Hero"
width="100%"
/>

## What's New?

MLflow has always been committed to providing powerful tracing capabilities for GenAI applications. With the addition of comprehensive OpenTelemetry integration, you can now:

- **Create unified traces** that combine MLflow SDK instrumentation with OpenTelemetry auto-instrumentation from third-party libraries
- **Ingest OpenTelemetry spans** directly into the MLflow tracking server
- **Seamlessly integrate** with existing applications that are instrumented with OpenTelemetry
- **Choose Arbitrary Languages** for your AI applications and trace them, including Java, Go, Rust, and more.

## Send OpenTelemetry traces to MLflow server

MLflow 3.6.0 supports ingesting OpenTelemetry traces directly through the OTLP (OpenTelemetry Protocol) endpoint. This allows you to send traces from any OpenTelemetry-instrumented application to the MLflow tracking server.

### Using MLflow Tracing SDK

The MLflow Tracing SDK is built on top of the OpenTelemetry SDK and designed for minimal instrumentation effort. It automatically integrates with OpenTelemetry's default tracer provider, allowing spans from different instrumentation sources to combine into cohesive traces.

**Auto-logging for GenAI frameworks:**

```python
import mlflow
from openai import OpenAI

# Enable auto-logging for OpenAI
mlflow.openai.autolog()

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
```

**Manual instrumentation with @trace decorator:**

```python
import mlflow

@mlflow.trace
def process_request(data):
    # Your application logic
    return result

# Call the traced function
process_request({"input": "data"})
```

### Using the OpenTelemetry SDK with OTLP Exporter

For applications that don't use the MLflow SDK or are written in languages other than Python/TypeScript, you can send traces directly to MLflow using the standard OpenTelemetry OTLP exporter. MLflow 3.6.0 exposes an OTLP-compliant endpoint that accepts traces from any OpenTelemetry-instrumented application.

#### Programmatic Configuration

Configure the OTLPSpanExporter in your application code to send traces to MLflow's OTLP endpoint:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Replace with your own MLflow server URI
mlflow_server_uri = "http://localhost:5000"
experiment_id = "123"  # Your MLflow experiment ID

# Configure OTLP exporter to send to MLflow
otlp_exporter = OTLPSpanExporter(
    endpoint=f"{mlflow_server_uri}/v1/traces",
    headers={
        "Content-Type": "application/x-protobuf",
        "x-mlflow-experiment-id": experiment_id,
    }
)

# Set up the tracer provider
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(provider)

# Now any OpenTelemetry instrumentation will send traces to MLflow
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("my_operation"):
    # Your application code here
    pass
```

#### Environment Variable Configuration

For applications written in other languages (Java, Go, Rust, etc.) or when you prefer configuration via environment variables, set the following:

```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:5000/v1/traces
export OTEL_EXPORTER_OTLP_TRACES_HEADERS=x-mlflow-experiment-id=123
```

This approach works with any OpenTelemetry SDK and allows you to trace applications across your entire technology stack, sending all traces to MLflow for unified observability.

<img
src={require("./otel_trace.png").default}
alt="OpenTelemetry and MLflow Trace"
width="100%"
/>

## Unified Observability Across Your Stack

With MLflow 3.6.0, you can now seamlessly combine:

- **OpenTelemetry auto-instrumentation** (FastAPI, Django, Flask, etc.)
- **MLflow auto-logging** (OpenAI, LangChain, DSPy, etc.)
- **MLflow @trace decorator** for custom spans

They can produce a **single, unified trace** across your application, providing end-to-end visibility from incoming requests to LLM calls and final outputs.

```python
# These now create a single unified trace
from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import mlflow

app = FastAPI()
FastAPIInstrumentor.instrument_app(app)  # OTel auto-instrumentation

@mlflow.trace  # MLflow tracing
def my_llm_function(input_text):
    # Your logic here
    return result
```

<img
src={require("./otel_and_mlflow_trace.png").default}
alt="OpenTelemetry and MLflow Trace"
width="100%"
/>

## Benefits of OpenTelemetry Integration

### ⚖️ Vendor Neutrality

OpenTelemetry is an open standard supported by the Cloud Native Computing Foundation (CNCF). By fully supporting OTel, MLflow ensures you're never locked into a proprietary observability solution.

### 🌐 Ecosystem Compatibility

Leverage the extensive OpenTelemetry ecosystem, including:

- Auto-instrumentation for popular frameworks (FastAPI, Django, Flask, etc.)
- Language support beyond Python (TypeScript, Java, Go, and more)
- Integration with existing observability tools and platforms

### 🔭 Unified Visibility

See the complete picture of your application's behavior in a single trace, from HTTP requests through your business logic to external API calls.

## Learn More

Ready to get started with OpenTelemetry in MLflow? Check out these resources:

- [MLflow Tracing Overview](https://mlflow.org/docs/latest/genai/tracing/)
- [OpenTelemetry Tracing Documentation](https://mlflow.org/docs/latest/genai/tracing/opentelemetry/)
- [MLflow Typescript SDK for Tracing](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/typescript-sdk/)
- [MLflow Auto Tracing integrations](https://mlflow.org/docs/latest/genai/tracing/integrations/)
- [OpenTelemetry Project](https://opentelemetry.io/)

## Join the Community

We're excited about the possibilities this integration opens up and would love to hear your [feedback](https://github.com/mlflow/mlflow/issues) and [contributions](https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md).

For those interested in sharing knowledge, we invite you to [collaborate on the MLflow website](https://github.com/mlflow/mlflow-website/blob/main/CONTRIBUTING.md). Whether it's writing tutorials, sharing use cases, or providing feedback, every contribution enriches the MLflow community.

Stay tuned for more updates as we continue to enhance MLflow's observability capabilities!
