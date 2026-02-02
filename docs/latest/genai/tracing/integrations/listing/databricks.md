# Tracing Databricks

[Databricks](https://www.databricks.com/) offers a unified platform for data, analytics and AI. Databricks Foundation Model APIs provide an OpenAI-compatible API format for accessing state-of-the-art models such as OpenAI GPT, Anthropic Claude, Google Gemini, and more, through a single platform. Since Databricks Foundation Model APIs are OpenAI-compatible, you can use MLflow tracing to trace your interactions with Databricks Foundation Model APIs.

![Tracing via autolog](/mlflow-website/docs/latest/images/llms/tracing/openai-function-calling.png)

## Managed MLflow on Databricks[​](#managed-mlflow-on-databricks "Direct link to Managed MLflow on Databricks")

Databricks offers a fully managed MLflow service as a part of their platform. This is the easiest way to get started with MLflow tracing, without having to set up any infrastructure. If you are using Databricks Foundation Model APIs, it is no brainer to use the managed MLflow for end-to-end LLMOps including tracing.

Visit Databricks documentation

This guide only covers how to trace Databricks Foundation Model APIs using MLflow tracing. For more details on how to get started with MLflow tracing on Databricks (e.g., tracing agent deployed on Databricks), please refer to the [Databricks documentation](https://docs.databricks.com/aws/en/mlflow3/genai/).

## Getting Started[​](#getting-started "Direct link to Getting Started")

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

### Enable tracing and call Databricks

* Python
* JS / TS

python

```python
import openai
import mlflow

# Enable auto-tracing for OpenAI (works with Databricks)
mlflow.openai.autolog()

# Initialize the OpenAI client with Databricks API endpoint
client = openai.OpenAI(
    base_url="https://example.staging.cloud.databricks.com/serving-endpoints",
    api_key="<your_databricks_token>",
)

response = client.chat.completions.create(
    model="databricks-gemini-3-pro",
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

// Wrap the OpenAI client and point to Databricks endpoint
const client = tracedOpenAI(
  new OpenAI({
    baseURL: "https://example.staging.cloud.databricks.com/serving-endpoints",
    apiKey: "<your_databricks_token>",
  })
);

const response = await client.chat.completions.create({
  model: "databricks-gemini-3-pro",
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "What is the capital of France?" },
  ],
});

```

3

### View traces in MLflow UI

Browse to your MLflow UI (for example, <http://localhost:5000>) and open the `Databricks` experiment to see traces for the calls above.

![Databricks Tracing](/mlflow-website/docs/latest/images/llms/tracing/basic-openai-trace.png)

-> View *[Next Steps](#next-steps)* for learning about more MLflow features like user feedback tracking, prompt management, and evaluation.

## Streaming and Async Support[​](#streaming-and-async-support "Direct link to Streaming and Async Support")

MLflow supports tracing for streaming and async Databricks APIs. Visit the [OpenAI Tracing documentation](/mlflow-website/docs/latest/genai/tracing/integrations/listing/openai.md) for example code snippets for tracing streaming and async calls through OpenAI SDK.

## Combine with frameworks or manual tracing[​](#combine-with-frameworks-or-manual-tracing "Direct link to Combine with frameworks or manual tracing")

The automatic tracing capability in MLflow is designed to work seamlessly with the [Manual Tracing SDK](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/manual-tracing.md) or multi-framework integrations. Please refer to the [Combining with frameworks or manual tracing](/mlflow-website/docs/latest/genai/tracing/integrations/listing/openai.md#combine-with-manual-tracing) for example code snippets.

![Databricks Tracing with Manual Tracing](/mlflow-website/docs/latest/images/llms/tracing/openai-trace-with-manual-span.png)

## Next steps[​](#next-steps "Direct link to Next steps")

### [Track User Feedback](/mlflow-website/docs/latest/genai/tracing/collect-user-feedback.md)

[Record user feedback on traces for tracking user satisfaction.](/mlflow-website/docs/latest/genai/tracing/collect-user-feedback.md)

[Learn about feedback ->](/mlflow-website/docs/latest/genai/tracing/collect-user-feedback.md)

### [Manage Prompts](/mlflow-website/docs/latest/genai/prompt-registry.md)

[Learn how to manage prompts with MLflow's prompt registry.](/mlflow-website/docs/latest/genai/prompt-registry.md)

[Manage prompts ->](/mlflow-website/docs/latest/genai/prompt-registry.md)

### [Evaluate Traces](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

[Evaluate traces with LLM judges to understand and improve your AI application's behavior.](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

[Evaluate traces ->](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)
