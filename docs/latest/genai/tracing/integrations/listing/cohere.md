# Tracing Cohere

[MLflow Tracing](/mlflow-website/docs/latest/genai/tracing/integrations.md) provides automatic tracing capability for <!-- -->Cohere<!-- --> models through the OpenAI SDK integration. Since <!-- -->Cohere<!-- --> offers an OpenAI-compatible API format, you can use<!-- --> `mlflow.openai.autolog()` to trace interactions with <!-- -->Cohere<!-- --> models.

![Tracing via autolog](/mlflow-website/docs/latest/images/llms/tracing/openai-function-calling.png)

MLflow trace automatically captures the following information about <!-- -->Cohere<!-- --> calls:

* Prompts and completion responses
* Latencies
* Token usage
* Model name
* Additional metadata such as `temperature`, `max_completion_tokens`, if specified.
* Function calling if returned in the response
* Built-in tools such as web search, file search, computer use, etc.
* Any exception if raised

## Getting Started

1

### Install dependencies

* Python
* JS / TS

bash

```bash
pip install 'mlflow[genai]' openai

```

bash

```bash
npm install mlflow-openai openai

```

2

### Start MLflow server

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

### Enable tracing and call Cohere

* Python
* JS / TS

python

```python
import openai
import mlflow

# Enable auto-tracing for OpenAI (works with Cohere)
mlflow.openai.autolog()

# Optional: Set a tracking URI and an experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Cohere")

# Initialize the OpenAI client with Cohere API endpoint
client = openai.OpenAI(
    base_url="https://api.cohere.ai/compatibility/v1",
    api_key="<your_cohere_api_key>",
)

response = client.chat.completions.create(
    model="command-a-03-2025",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
)

```

typescript

```typescript
import { OpenAI } from "openai";
import { tracedOpenAI } from "mlflow-openai";

// Wrap the OpenAI client and point to Cohere endpoint
const client = tracedOpenAI(
  new OpenAI({
    baseURL: "https://api.cohere.com/v1",
    apiKey: "<your_cohere_api_key>",
  })
);

const response = await client.chat.completions.create({
  model: "<your_cohere_model>",
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "What is the capital of France?" },
  ],
  temperature: 0.1,
  max_tokens: 100,
});

```

4

### View traces in MLflow UI

Browse to your MLflow UI (for example, http\://localhost:5000) and open the `Cohere` experiment to see traces for the calls above.

![Cohere Tracing](/mlflow-website/docs/latest/images/llms/tracing/basic-openai-trace.png)

-> View<!-- --> *[Next Steps](#next-steps)* <!-- -->for learning about more MLflow features like user feedback tracking, prompt management, and evaluation.

## Streaming and Async Support

MLflow supports tracing for streaming and async <!-- -->Cohere<!-- --> APIs. Visit the<!-- --> [OpenAI Tracing documentation](/mlflow-website/docs/latest/genai/tracing/integrations/listing/openai.md) for example code snippets for tracing streaming and async calls through OpenAI SDK.

## Combine with frameworks or manual tracing

The automatic tracing capability in MLflow is designed to work seamlessly with the<!-- --> [Manual Tracing SDK](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/manual-tracing.md) or multi-framework integrations. The examples below show Python (manual span) and JS/TS (manual span) at the same level of complexity.

* Python
* JS / TS

python

```python
import json
from openai import OpenAI
import mlflow
from mlflow.entities import SpanType

# Initialize the OpenAI client with Cohere API endpoint
client = OpenAI(
    base_url="https://api.cohere.ai/compatibility/v1",
    api_key="<your_cohere_api_key>",
)


# Create a parent span for the Cohere call
@mlflow.trace(span_type=SpanType.CHAIN)
def answer_question(question: str):
    messages = [{"role": "user", "content": question}]
    response = client.chat.completions.create(
        model="command-a-03-2025",
        messages=messages,
    )

    # Attach session/user metadata to the trace
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.session": "session-12345",
            "mlflow.trace.user": "user-a",
        }
    )
    return response.choices[0].message.content


answer = answer_question("What is the capital of France?")

```

typescript

```typescript
import * as mlflow from "mlflow-tracing";
import { OpenAI } from "openai";
import { tracedOpenAI } from "mlflow-openai";

mlflow.init({
  trackingUri: "http://localhost:5000",
  experimentId: "<your_experiment_id>",
});

// Wrap the OpenAI client and point to Cohere endpoint
const client = tracedOpenAI(
  new OpenAI({
    baseURL: "https://api.cohere.com/v1",
    apiKey: "<your_cohere_api_key>",
  })
);

// Create a traced function that wraps the Cohere call
const answerQuestion = mlflow.trace(
  async (question: string) => {
    const resp = await client.chat.completions.create({
      model: "<your_cohere_model>",
      messages: [{ role: "user", content: question }],
    });
    return resp.choices[0].message?.content;
  },
  { name: "answer-question" }
);

await answerQuestion("What is the capital of France?");

```

Running either example will produce a trace that includes the <!-- -->Cohere<!-- --> LLM span; the traced function creates the parent span automatically.

![Cohere Tracing with Manual Tracing](/mlflow-website/docs/latest/images/llms/tracing/openai-trace-with-manual-span.png)

## Next steps

### [Track User Feedback](/mlflow-website/docs/latest/genai/tracing/collect-user-feedback.md)

[Record user feedback on traces for tracking user satisfaction.](/mlflow-website/docs/latest/genai/tracing/collect-user-feedback.md)

[Learn about feedback ->](/mlflow-website/docs/latest/genai/tracing/collect-user-feedback.md)

### [Manage Prompts](/mlflow-website/docs/latest/genai/prompt-registry.md)

[Learn how to manage prompts with MLflow's prompt registry.](/mlflow-website/docs/latest/genai/prompt-registry.md)

[Manage prompts ->](/mlflow-website/docs/latest/genai/prompt-registry.md)

### [Evaluate Traces](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

[Evaluate traces with LLM judges to understand and improve your AI application's behavior.](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

[Evaluate traces ->](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)
