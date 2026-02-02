# Tracing Portkey

[Portkey](https://portkey.ai/) is an enterprise-grade AI gateway that provides unified access to 1600+ LLMs through a single OpenAI-compatible API. It offers built-in guardrails, observability, caching, load balancing, and fallback mechanisms for production AI applications.

![Portkey Tracing](/mlflow-website/docs/latest/images/llms/tracing/basic-openai-trace.png)

Since <!-- -->Portkey<!-- --> exposes an OpenAI-compatible API, you can use MLflow's OpenAI autolog integration to automatically trace all your LLM calls through the gateway.

## Getting Started

Prerequisites

Create a [Portkey account](https://portkey.ai/) and generate an API key from the [API Keys page](https://app.portkey.ai/api-keys). Configure your virtual keys for the LLM providers you want to use.

1

### Install Dependencies

* Python
* TypeScript

bash

```bash
pip install mlflow openai

```

bash

```bash
npm install mlflow-openai openai

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

* Python
* TypeScript

Enable tracing with `mlflow.openai.autolog()` and configure the OpenAI client to use<!-- --> <!-- -->Portkey<!-- -->'s base URL.

python

```python
import mlflow
from openai import OpenAI

# Enable auto-tracing for OpenAI
mlflow.openai.autolog()

# Set tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Portkey")

# Create OpenAI client pointing to Portkey
client = OpenAI(
    base_url="https://api.portkey.ai/v1",
    api_key="<YOUR_PORTKEY_API_KEY>",
    default_headers={
        "x-portkey-provider": "openai",  # or "anthropic", "google", etc.
    },
)

# Make API calls - traces will be captured automatically
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
)
print(response.choices[0].message.content)

```

Initialize MLflow tracing with `init()` and wrap the OpenAI client with the<!-- --> `tracedOpenAI` function.

typescript

```typescript
import { init } from "mlflow-tracing";
import { tracedOpenAI } from "mlflow-openai";
import { OpenAI } from "openai";

// Initialize MLflow tracing
init({
  trackingUri: "http://localhost:5000",
  experimentId: "<experiment-id>",
});

// Wrap the OpenAI client pointing to Portkey
const client = tracedOpenAI(
  new OpenAI({
    baseURL: "https://api.portkey.ai/v1",
    apiKey: "<YOUR_PORTKEY_API_KEY>",
    defaultHeaders: {
      "x-portkey-provider": "openai", // or "anthropic", "google", etc.
    },
  })
);

// Make API calls - traces will be captured automatically
const response = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "What is the capital of France?" },
  ],
});
console.log(response.choices[0].message.content);

```

4

### View Traces in MLflow UI

Open the MLflow UI at http\://localhost:5000 to see the traces from your <!-- -->Portkey<!-- --> API calls.

## Combining with Manual Tracing

You can combine auto-tracing with MLflow's manual tracing to create comprehensive traces that include your application logic:

* Python
* TypeScript

python

```python
import mlflow
from mlflow.entities import SpanType
from openai import OpenAI

mlflow.openai.autolog()

client = OpenAI(
    base_url="https://api.portkey.ai/v1",
    api_key="<YOUR_PORTKEY_API_KEY>", default_headers={"x-portkey-provider": "openai"},
)


@mlflow.trace(span_type=SpanType.CHAIN)
def ask_question(question: str) -> str:
    """A traced function that calls the LLM through Portkey."""
    response = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content


# The entire function call and nested LLM call will be traced
answer = ask_question("What is machine learning?")
print(answer)

```

typescript

```typescript
import { init, trace, SpanType } from "mlflow-tracing";
import { tracedOpenAI } from "mlflow-openai";
import { OpenAI } from "openai";

init({
  trackingUri: "http://localhost:5000",
  experimentId: "<experiment-id>",
});

const client = tracedOpenAI(
  new OpenAI({
    baseURL: "https://api.portkey.ai/v1",
    apiKey: "<YOUR_PORTKEY_API_KEY>", defaultHeaders: { "x-portkey-provider": "openai" },
  })
);

// Wrap your function with trace() to create a span
const askQuestion = trace(
  { name: "askQuestion", spanType: SpanType.CHAIN },
  async (question: string): Promise<string> => {
    const response = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: question }],
    });
    return response.choices[0].message.content ?? "";
  }
);

// The entire function call and nested LLM call will be traced
const answer = await askQuestion("What is machine learning?");
console.log(answer);

```

## Streaming Support

MLflow supports tracing streaming responses from <!-- -->Portkey<!-- -->:

* Python
* TypeScript

python

```python
import mlflow
from openai import OpenAI

mlflow.openai.autolog()

client = OpenAI(
    base_url="https://api.portkey.ai/v1",
    api_key="<YOUR_PORTKEY_API_KEY>", default_headers={"x-portkey-provider": "openai"},
)

stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a haiku about machine learning."}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

```

typescript

```typescript
import { init } from "mlflow-tracing";
import { tracedOpenAI } from "mlflow-openai";
import { OpenAI } from "openai";

init({
  trackingUri: "http://localhost:5000",
  experimentId: "<experiment-id>",
});

const client = tracedOpenAI(
  new OpenAI({
    baseURL: "https://api.portkey.ai/v1",
    apiKey: "<YOUR_PORTKEY_API_KEY>", defaultHeaders: { "x-portkey-provider": "openai" },
  })
);

const stream = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Write a haiku about machine learning." }],
  stream: true,
});

for await (const chunk of stream) {
  if (chunk.choices[0].delta.content) {
    process.stdout.write(chunk.choices[0].delta.content);
  }
}

```

MLflow will automatically capture the complete streamed response in the trace.

## Next Steps

### [Track User Feedback](/mlflow-website/docs/latest/genai/tracing/collect-user-feedback.md)

[Record user feedback on traces for tracking user satisfaction.](/mlflow-website/docs/latest/genai/tracing/collect-user-feedback.md)

[Learn about feedback →](/mlflow-website/docs/latest/genai/tracing/collect-user-feedback.md)

### [Manage Prompts](/mlflow-website/docs/latest/genai/prompt-registry.md)

[Learn how to manage prompts with MLflow's prompt registry.](/mlflow-website/docs/latest/genai/prompt-registry.md)

[Manage prompts →](/mlflow-website/docs/latest/genai/prompt-registry.md)

### [Evaluate Traces](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

[Evaluate traces with LLM judges to understand and improve your AI application's behavior.](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

[Evaluate traces →](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)
