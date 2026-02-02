# Tracing Databricks AI Gateway

[Databricks AI Gateway](https://docs.databricks.com/en/ai-gateway/index.html) (formerly Mosaic AI Gateway) is the Databricks solution for governing and monitoring access to generative AI models and their associated model serving endpoints. It is a centralized service that brings governance, monitoring, and production readiness to model serving endpoints.

Looking for Databricks Foundation Model APIs?

This guide covers tracing LLM calls through **Databricks AI Gateway**. If you're using Databricks Foundation Model APIs directly, see the [Databricks Integration](/mlflow-website/docs/latest/genai/tracing/integrations/listing/databricks.md) guide instead.

## What is Databricks AI Gateway?[​](#what-is-databricks-ai-gateway "Direct link to What is Databricks AI Gateway?")

Databricks AI Gateway streamlines the usage and management of generative AI models and agents within an organization. Key capabilities include:

* **Governance**: Control access with permissions and rate limiting
* **Monitoring**: Track usage and costs with system tables, audit requests with payload logging
* **Safety**: Configure AI guardrails to prevent harmful content and detect PII
* **Reliability**: Minimize outages with fallbacks and load balance traffic across models

All data is logged into Delta tables in Unity Catalog.

## Integration Options[​](#integration-options "Direct link to Integration Options")

There are two ways to trace LLM calls through Databricks AI Gateway:

| Approach                           | Description                                                    | Best For                                                   |
| ---------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------- |
| **Server-side (Inference Tables)** | Databricks automatically logs all requests to inference tables | Centralized tracing for all requests through the gateway   |
| **Client-side Tracing**            | Use OpenAI SDK with MLflow autolog                             | Combining LLM traces with your agent or application traces |

note

With **server-side tracing**, all requests through the gateway are captured in inference tables, regardless of which client made them. For application-specific tracing where you want to combine LLM calls with your application logic, use **client-side tracing**.

## Option 1: Server-side Tracing (Inference Tables)[​](#option-1-server-side-tracing-inference-tables "Direct link to Option 1: Server-side Tracing (Inference Tables)")

Databricks AI Gateway can automatically log all requests and responses to inference tables in Unity Catalog. This provides centralized monitoring without any client-side code changes.

See the [Databricks Inference Tables documentation](https://docs.databricks.com/aws/en/ai-gateway/inference-tables) for setup instructions.

## Option 2: Client-side Tracing[​](#option-2-client-side-tracing "Direct link to Option 2: Client-side Tracing")

If you want to trace LLM calls as part of your application traces, you can use MLflow's automatic tracing with the OpenAI SDK or other supported SDKs.

1

### Install Dependencies

* Python
* TypeScript

bash

```bash
pip install 'mlflow[genai]' openai

```

bash

```bash
npm install mlflow-openai openai

```

2

### Set tracking URI and experiment

To save traces to Databricks-managed MLflow, set the tracking URI to `databricks` and the experiment to the name of your Databricks workspace.

* Python
* TypeScript

python

```python
import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("<your-databricks-workspace>")

```

typescript

```typescript
import * as mlflow from "mlflow-tracing";

mlflow.init({
  trackingUri: "databricks",
  experimentId: "<your-databricks-workspace>",
});

```

When you want to self-host MLflow outside Databricks, follow the [Self-hosting](/mlflow-website/docs/latest/self-hosting.md) guide to set up your MLflow server and set the tracking URI accordingly.

3

### Enable Tracing and Make API Calls

* Python
* TypeScript

Since Databricks AI Gateway exposes an OpenAI-compatible API, enable tracing with `mlflow.openai.autolog()` and configure the OpenAI client to use your Databricks serving endpoint.

python

```python
import mlflow
from openai import OpenAI

# Enable auto-tracing for OpenAI
mlflow.openai.autolog()

# Point OpenAI client to Databricks AI Gateway
client = OpenAI(
    base_url="https://<databricks-workspace>/serving-endpoints",
    api_key="<DATABRICKS_TOKEN>",
)

# Make API calls - traces will be captured automatically
response = client.chat.completions.create(
    model="<your-endpoint-name>",  # Your Databricks serving endpoint name
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
)
print(response.choices[0].message.content)

```

Initialize MLflow tracing with `init()` and wrap the OpenAI client with the `tracedOpenAI` function.

typescript

```typescript
import { tracedOpenAI } from "mlflow-openai";
import { OpenAI } from "openai";

// Wrap the OpenAI client pointing to Databricks AI Gateway
const client = tracedOpenAI(
  new OpenAI({
    baseURL: "https://<databricks-workspace>/serving-endpoints",
    apiKey: "<DATABRICKS_TOKEN>",
  })
);

// Make API calls - traces will be captured automatically
const response = await client.chat.completions.create({
  model: "<your-endpoint-name>",
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "What is the capital of France?" },
  ],
});
console.log(response.choices[0].message.content);

```

4

### View Traces

* **On Databricks**: Navigate to the MLflow Experiments page in your workspace
* **Local MLflow**: Open the MLflow UI at <http://localhost:5000>

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
