# MLflow AI Gateway

warning

MLflow AI Gateway does not support Windows.

MLflow AI Gateway provides a unified interface for deploying and managing multiple LLM providers within your organization. It simplifies interactions with services like OpenAI, Anthropic, and others through a single, secure endpoint.

The gateway server excels in production environments where organizations need to manage multiple LLM providers securely while maintaining operational flexibility and developer productivity.

#### Unified Interface

Access multiple LLM providers through a single endpoint, eliminating the need to integrate with each provider individually.

#### Centralized Security

Store API keys in one secure location with request/response logging for audit trails and compliance.

#### Provider Abstraction

Switch between OpenAI, Anthropic, Azure OpenAI, and other providers without changing your application code.

#### Zero-Downtime Updates

Add, remove, or modify endpoints dynamically without restarting the server or disrupting running applications.

#### Cost Optimization

Monitor usage across providers and optimize costs by routing requests to the most efficient models.

#### Team Collaboration

Shared endpoint configurations and standardized access patterns across development teams.

## Getting Started[​](#getting-started "Direct link to Getting Started")

Choose your path to get up and running with MLflow AI Gateway:

### [Setup](/mlflow-website/docs/latest/genai/governance/ai-gateway/setup.md)

[Install MLflow, configure environment, and start your gateway server](/mlflow-website/docs/latest/genai/governance/ai-gateway/setup.md)

[Start setup →](/mlflow-website/docs/latest/genai/governance/ai-gateway/setup.md)

### [Configuration](/mlflow-website/docs/latest/genai/governance/ai-gateway/configuration.md)

[Configure providers, endpoints, and advanced gateway settings](/mlflow-website/docs/latest/genai/governance/ai-gateway/configuration.md)

[Configure providers →](/mlflow-website/docs/latest/genai/governance/ai-gateway/configuration.md)

### [Usage](/mlflow-website/docs/latest/genai/governance/ai-gateway/usage.md)

[Query endpoints with Python client and REST APIs](/mlflow-website/docs/latest/genai/governance/ai-gateway/usage.md)

[Start using →](/mlflow-website/docs/latest/genai/governance/ai-gateway/usage.md)

### [Integration](/mlflow-website/docs/latest/genai/governance/ai-gateway/integration.md)

[Integrate with applications, frameworks, and production systems](/mlflow-website/docs/latest/genai/governance/ai-gateway/integration.md)

[Learn integrations →](/mlflow-website/docs/latest/genai/governance/ai-gateway/integration.md)

## Quick Start[​](#quick-start "Direct link to Quick Start")

Get your AI Gateway running with OpenAI in under 5 minutes:

* 1\. Install
* 2\. Configure
* 3\. Start Server
* 4\. Test

Install MLflow with gateway dependencies:

bash

```bash
pip install 'mlflow[gateway]'

```

Set your OpenAI API key:

bash

```bash
export OPENAI_API_KEY=your_api_key_here

```

Create a simple configuration file `config.yaml`:

yaml

```yaml
endpoints:
  - name: chat
    endpoint_type: llm/v1/chat
    model:
      provider: openai
      name: gpt-3.5-turbo
      config:
        openai_api_key: $OPENAI_API_KEY

```

Start the gateway server:

bash

```bash
mlflow gateway start --config-path config.yaml --port 5000

```

Your gateway is now running at `http://localhost:5000`

Test your endpoint:

bash

```bash
curl -X POST http://localhost:5000/gateway/chat/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

```

## Supported Providers[​](#supported-providers "Direct link to Supported Providers")

MLflow AI Gateway supports a comprehensive range of LLM providers:

| Provider              | Chat | Chat function calling | Completions | Embeddings | Notes                                    |
| --------------------- | ---- | --------------------- | ----------- | ---------- | ---------------------------------------- |
| OpenAI                | ✅   | ✅                    | ✅          | ✅         | GPT-4, GPT-5, text-embedding models      |
| Azure OpenAI          | ✅   | ✅                    | ✅          | ✅         | Enterprise OpenAI with Azure integration |
| Anthropic             | ✅   | ✅                    | ✅          | ❌         | Claude models via Anthropic API          |
| Gemini                | ✅   | ✅                    | ✅          | ✅         | Gemini models via Gemini API             |
| AWS Bedrock Claude    | ✅   | ✅                    | ✅          | ✅         | Claude models provided by AWS Bedrock    |
| AWS Bedrock Titan     | ❌   | ❌                    | ✅          | ❌         | Titan models provided by AWS Bedrock     |
| AWS Bedrock AI21      | ❌   | ❌                    | ✅          | ❌         | AI21 models provided by AWS Bedrock      |
| MLflow Models         | ✅   | ❌                    | ✅          | ✅         | Your own deployed MLflow models          |
| Cohere (deprecated)   | ✅   | ❌                    | ✅          | ✅         | Command and embedding models             |
| PaLM (deprecated)     | ✅   | ❌                    | ✅          | ✅         | Google's PaLM models                     |
| MosaicML (deprecated) | ✅   | ❌                    | ✅          | ❌         | MPT models and custom deployments        |

## Core Concepts[​](#core-concepts "Direct link to Core Concepts")

Understanding these key concepts will help you effectively use the AI Gateway:

### Endpoints[​](#endpoints "Direct link to Endpoints")

Endpoints are named configurations that define how to access a specific model from a provider. Each endpoint specifies the model, provider settings, and access parameters.

### Providers[​](#providers "Direct link to Providers")

Providers are the underlying LLM services (OpenAI, Anthropic, etc.) that actually serve the models. The gateway abstracts away provider-specific details.

### Routes[​](#routes "Direct link to Routes")

Routes define the URL structure for accessing endpoints. The gateway automatically creates routes based on your endpoint configurations.

### Dynamic Updates[​](#dynamic-updates "Direct link to Dynamic Updates")

The gateway supports hot-reloading of configurations, allowing you to add, modify, or remove endpoints without restarting the server.

## Next Steps[​](#next-steps "Direct link to Next Steps")

Ready to dive deeper? Explore these resources:

### [Setup Guide](/mlflow-website/docs/latest/genai/governance/ai-gateway/setup.md)

[Get started with installation, environment setup, and server configuration](/mlflow-website/docs/latest/genai/governance/ai-gateway/setup.md)

[Start setup →](/mlflow-website/docs/latest/genai/governance/ai-gateway/setup.md)

### [Usage Guide](/mlflow-website/docs/latest/genai/governance/ai-gateway/usage.md)

[Learn basic querying patterns with Python client and REST APIs](/mlflow-website/docs/latest/genai/governance/ai-gateway/usage.md)

[Learn usage →](/mlflow-website/docs/latest/genai/governance/ai-gateway/usage.md)

### [Integration Guide](/mlflow-website/docs/latest/genai/governance/ai-gateway/integration.md)

[Integrate with applications, frameworks, and production systems](/mlflow-website/docs/latest/genai/governance/ai-gateway/integration.md)

[View integrations →](/mlflow-website/docs/latest/genai/governance/ai-gateway/integration.md)
