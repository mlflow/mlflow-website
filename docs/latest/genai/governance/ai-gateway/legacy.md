# Gateway Server (Legacy)

<!-- -->

The Gateway Server provides a YAML-based configuration approach for deploying and managing LLM endpoints. This legacy method offers flexibility for users who prefer file-based configuration and command-line server management.

note

For new deployments, we recommend using the [Gateway Quickstart](/mlflow-website/docs/latest/genai/governance/ai-gateway/quickstart.md) which provides a modern web interface for managing endpoints, API keys, and routing configurations with zero-downtime updates.

## Supported Providers[​](#supported-providers "Direct link to Supported Providers")

The Gateway Server supports a comprehensive range of LLM providers through YAML configuration:

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

Understanding these key concepts will help you effectively configure the Gateway Server:

### Endpoints[​](#endpoints "Direct link to Endpoints")

Endpoints are named configurations defined in YAML that specify how to access a specific model from a provider. Each endpoint includes the model name, provider settings, and authentication parameters. Endpoints are configured in your YAML file and loaded when the server starts.

### Providers[​](#providers "Direct link to Providers")

Providers are the underlying LLM services (OpenAI, Anthropic, etc.) that serve the models. Each provider requires specific configuration parameters and authentication credentials, which you define in the endpoint configuration.

### Routes[​](#routes "Direct link to Routes")

Routes provide advanced request routing capabilities, allowing you to define traffic splitting and fallback strategies across multiple endpoints. Routes are configured in the YAML file under the `routes` section and enable load balancing and high availability patterns.

### Configuration Management[​](#configuration-management "Direct link to Configuration Management")

The Gateway Server uses YAML files for all configuration. To update endpoints or routes, you modify the YAML file and restart the server. This approach provides version control and declarative configuration benefits, though it requires server restarts for changes to take effect.

## Getting Started[​](#getting-started "Direct link to Getting Started")

Choose your next step to configure and use the Gateway Server:

### [Setup](/mlflow-website/docs/latest/genai/governance/ai-gateway/legacy/setup.md)

[Install MLflow, configure environment, and start your gateway server](/mlflow-website/docs/latest/genai/governance/ai-gateway/legacy/setup.md)

[Get started →](/mlflow-website/docs/latest/genai/governance/ai-gateway/legacy/setup.md)

### [Configuration](/mlflow-website/docs/latest/genai/governance/ai-gateway/legacy/configuration.md)

[Configure providers, endpoints, and advanced routing settings](/mlflow-website/docs/latest/genai/governance/ai-gateway/legacy/configuration.md)

[Configure →](/mlflow-website/docs/latest/genai/governance/ai-gateway/legacy/configuration.md)

### [Usage](/mlflow-website/docs/latest/genai/governance/ai-gateway/legacy/usage.md)

[Query endpoints with Python client and REST APIs](/mlflow-website/docs/latest/genai/governance/ai-gateway/legacy/usage.md)

[Learn usage →](/mlflow-website/docs/latest/genai/governance/ai-gateway/legacy/usage.md)
