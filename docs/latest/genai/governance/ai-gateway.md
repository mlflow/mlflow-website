# MLflow AI Gateway

MLflow AI Gateway provides a unified interface for deploying and managing multiple LLM providers within your organization. It simplifies interactions with services like OpenAI, Anthropic, and others through a single, secure endpoint.

The gateway excels in production environments where organizations need to manage multiple LLM providers securely while maintaining operational flexibility. Advanced routing capabilities enable traffic splitting for A/B testing and automatic failover chains for high availability.

MLflow AI Gateway also offers passthrough endpoints, enabling requests to be forwarded in providers' native formats. This feature allows you to access provider-specific capabilities as soon as they become available.

#### Unified Interface

Access multiple LLM providers through a single endpoint, eliminating the need to integrate with each provider individually.

#### Centralized Security

Store API keys in one secure location with request/response logging for audit trails and compliance.

#### Advanced Routing

Traffic splitting for A/B testing and automatic fallbacks ensure high availability across providers.

#### Zero-Downtime Updates

Add, remove, or modify endpoints dynamically without restarting the server or disrupting running applications.

#### Cost Optimization

Monitor usage across providers and optimize costs by routing requests to the most efficient models.

#### Team Collaboration

Shared endpoint configurations and standardized access patterns across development teams.

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [Quickstart](/mlflow-website/docs/latest/genai/governance/ai-gateway/quickstart.md)

[Get your AI Gateway running in minutes with a simple walkthrough](/mlflow-website/docs/latest/genai/governance/ai-gateway/quickstart.md)

[Get started →](/mlflow-website/docs/latest/genai/governance/ai-gateway/quickstart.md)

### [API Keys](/mlflow-website/docs/latest/genai/governance/ai-gateway/api-keys/create-and-manage.md)

[Create and manage API keys for secure credential management](/mlflow-website/docs/latest/genai/governance/ai-gateway/api-keys/create-and-manage.md)

[Manage keys →](/mlflow-website/docs/latest/genai/governance/ai-gateway/api-keys/create-and-manage.md)

### [Endpoints](/mlflow-website/docs/latest/genai/governance/ai-gateway/endpoints/create-and-manage.md)

[Create, configure, and query AI model endpoints](/mlflow-website/docs/latest/genai/governance/ai-gateway/endpoints/create-and-manage.md)

[Configure endpoints →](/mlflow-website/docs/latest/genai/governance/ai-gateway/endpoints/create-and-manage.md)

### [Traffic Routing](/mlflow-website/docs/latest/genai/governance/ai-gateway/traffic-routing-fallbacks.md)

[Configure traffic splitting and fallbacks for high availability](/mlflow-website/docs/latest/genai/governance/ai-gateway/traffic-routing-fallbacks.md)

[Learn routing →](/mlflow-website/docs/latest/genai/governance/ai-gateway/traffic-routing-fallbacks.md)

### [Authentication](/mlflow-website/docs/latest/self-hosting/security/basic-http-auth.md#ai-gateway-permissions)

[Configure HTTP Basic Authentication for AI Gateway resources](/mlflow-website/docs/latest/self-hosting/security/basic-http-auth.md#ai-gateway-permissions)

[Learn more →](/mlflow-website/docs/latest/self-hosting/security/basic-http-auth.md#ai-gateway-permissions)
