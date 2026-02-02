# Tracing Groq

![Groq tracing via autolog](/mlflow-website/docs/latest/images/llms/groq/groq-tracing.png)

[MLflow Tracing](/mlflow-website/docs/latest/genai/tracing.md) provides automatic tracing capability when using Groq. When Groq auto-tracing is enabled by calling the [`mlflow.groq.autolog()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.groq.html#mlflow.groq.autolog) function, usage of the Groq SDK will automatically record generated traces during interactive development.

MLflow automatically captures the following information about Groq calls:

* Prompts and completion responses
* Latencies
* Model name
* Token usage (input, output, and total tokens)
* Additional metadata such as `temperature`, `max_tokens`, if specified
* Any exception if raised

## Getting Started[​](#getting-started "Direct link to Getting Started")

1

### Install Dependencies

bash

```bash
pip install 'mlflow[genai]' groq

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

Enable tracing with `mlflow.groq.autolog()` and make API calls as usual.

python

```python
import groq
import mlflow
import os

# Enable auto-tracing for Groq
mlflow.groq.autolog()

# Set a tracking URI and an experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Groq")

# Initialize Groq client
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

# Use the create method to create new message
message = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of low latency LLMs.",
        }
    ],
)

print(message.choices[0].message.content)

```

4

### View Traces in MLflow UI

Browse to the MLflow UI at <http://localhost:5000> (or your MLflow server URL) and you should see the traces for the Groq API calls.

## Supported APIs[​](#supported-apis "Direct link to Supported APIs")

Note that only synchronous calls are supported, and that asynchronous API and streaming methods are not traced.

| Normal | Streaming | Async |
| ------ | --------- | ----- |
| ✅     | -         | -     |

## Token usage[​](#token-usage "Direct link to Token usage")

MLflow >= 3.2.0 supports token usage tracking for Groq. The token usage for each LLM call will be logged in the `mlflow.chat.tokenUsage` attribute. The total token usage throughout the trace will be available in the `token_usage` field of the trace info object.

python

```python
import json
import mlflow

mlflow.groq.autolog()

client = groq.Groq()

# Use the create method to create new message
message = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of low latency LLMs.",
        }
    ],
)

# Get the trace object just created
last_trace_id = mlflow.get_last_active_trace_id()
trace = mlflow.get_trace(trace_id=last_trace_id)

# Print the token usage
total_usage = trace.info.token_usage
print("== Total token usage: ==")
print(f"  Input tokens: {total_usage['input_tokens']}")
print(f"  Output tokens: {total_usage['output_tokens']}")
print(f"  Total tokens: {total_usage['total_tokens']}")

# Print the token usage for each LLM call
print("\n== Detailed usage for each LLM call: ==")
for span in trace.data.spans:
    if usage := span.get_attribute("mlflow.chat.tokenUsage"):
        print(f"{span.name}:")
        print(f"  Input tokens: {usage['input_tokens']}")
        print(f"  Output tokens: {usage['output_tokens']}")
        print(f"  Total tokens: {usage['total_tokens']}")

```

bash

```bash
== Total token usage: ==
  Input tokens: 21
  Output tokens: 628
  Total tokens: 649

== Detailed usage for each LLM call: ==
Completions:
  Input tokens: 21
  Output tokens: 628

```

Currently, groq token usage doesn't support token usage tracking for Audio transcription and Audio translation.

## Disable auto-tracing[​](#disable-auto-tracing "Direct link to Disable auto-tracing")

Auto tracing for Groq can be disabled globally by calling `mlflow.groq.autolog(disable=True)` or `mlflow.autolog(disable=True)`.
