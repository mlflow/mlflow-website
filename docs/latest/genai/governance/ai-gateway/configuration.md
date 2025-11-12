# AI Gateway Configuration

<!-- -->

Configure providers, endpoints, and advanced settings for your MLflow AI Gateway.

## Provider Configurations[​](#provider-configurations "Direct link to Provider Configurations")

Configure endpoints for different LLM providers using these YAML examples:

* OpenAI
* Azure OpenAI
* Anthropic
* Gemini
* AWS Bedrock
* Cohere
* MosaicAI
* Databricks
* MLflow Models

yaml

```yaml
endpoints:
  - name: gpt4-chat
    endpoint_type: llm/v1/chat
    model:
      provider: openai
      name: gpt-4
      config:
        openai_api_key: $OPENAI_API_KEY
        openai_api_base: https://api.openai.com/v1  # Optional
        openai_organization: your_org_id  # Optional

```

yaml

```yaml
endpoints:
  - name: azure-chat
    endpoint_type: llm/v1/chat
    model:
      provider: azuread
      name: gpt-35-turbo
      config:
        openai_api_key: $AZURE_OPENAI_API_KEY
        openai_api_base: https://your-resource.openai.azure.com/
        openai_api_version: "2023-05-15"
        openai_deployment_name: your-deployment-name

```

yaml

```yaml
endpoints:
  - name: claude-chat
    endpoint_type: llm/v1/chat
    model:
      provider: anthropic
      name: claude-2
      config:
        anthropic_api_key: $ANTHROPIC_API_KEY

```

yaml

```yaml
endpoints:
  - name: gemini-chat
    endpoint_type: llm/v1/chat
    model:
      provider: gemini
      name: gemini-2.5-flash
      config:
        gemini_api_key: $GEMINI_API_KEY

```

yaml

```yaml
endpoints:
  - name: bedrock-chat
    endpoint_type: llm/v1/chat
    model:
      provider: bedrock
      name: anthropic.claude-instant-v1
      config:
        aws_config:
          aws_access_key_id: $AWS_ACCESS_KEY_ID
          aws_secret_access_key: $AWS_SECRET_ACCESS_KEY
          aws_region: us-east-1

```

yaml

```yaml
endpoints:
  - name: cohere-completions
    endpoint_type: llm/v1/completions
    model:
      provider: cohere
      name: command
      config:
        cohere_api_key: $COHERE_API_KEY

  - name: cohere-embeddings
    endpoint_type: llm/v1/embeddings
    model:
      provider: cohere
      name: embed-english-v2.0
      config:
        cohere_api_key: $COHERE_API_KEY

```

yaml

```yaml
endpoints:
  - name: mosaicai-chat
    endpoint_type: llm/v1/chat
    model:
      provider: mosaicai
      name: llama2-70b-chat
      config:
        mosaicai_api_key: $MOSAICAI_API_KEY

```

Databricks [Foundation Models APIs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/) are compatible with the OpenAI Chat Completions API, so you can use them with `openai` provider in the AI Gateway. Specify the endpoint name (e.g., `databricks-claude-sonnet-4`) in the `name` field and set the host and token as OpenAI API key and base URL respectively.

yaml

```yaml
endpoints:
  - name: databricks-chat
    endpoint_type: llm/v1/chat
    model:
      provider: openai
      name: databricks-claude-sonnet-4
      config:
        openai_api_key: $DATABRICKS_TOKEN
        openai_api_base: https://your-workspace.cloud.databricks.com/serving-endpoints/  # Replace with your Databricks workspace URL

```

yaml

```yaml
endpoints:
  - name: custom-model
    endpoint_type: llm/v1/chat
    model:
      provider: mlflow-model-serving
      name: my-model
      config:
        model_server_url: http://localhost:5001

```

note

MosaicML PaLM, and Cohere providers are deprecated, will be removed in a future MLflow version.

## Environment Variables[​](#environment-variables "Direct link to Environment Variables")

Store API keys as environment variables for security:

bash

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Azure OpenAI
export AZURE_OPENAI_API_KEY=your-azure-key
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# AWS Bedrock
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1

# Cohere
export COHERE_API_KEY=...

```

## Advanced Configuration[​](#advanced-configuration "Direct link to Advanced Configuration")

### Rate Limiting[​](#rate-limiting "Direct link to Rate Limiting")

Configure rate limits per endpoint:

yaml

```yaml
endpoints:
  - name: rate-limited-chat
    endpoint_type: llm/v1/chat
    model:
      provider: openai
      name: gpt-3.5-turbo
      config:
        openai_api_key: $OPENAI_API_KEY
    limit:
      renewal_period: minute
      calls: 100  # max calls per renewal period

```

### Model Parameters[​](#model-parameters "Direct link to Model Parameters")

Set default model parameters:

yaml

```yaml
endpoints:
  - name: configured-chat
    endpoint_type: llm/v1/chat
    model:
      provider: openai
      name: gpt-3.5-turbo
      config:
        openai_api_key: $OPENAI_API_KEY
        temperature: 0.7
        max_tokens: 1000
        top_p: 0.9

```

### Multiple Endpoints[​](#multiple-endpoints "Direct link to Multiple Endpoints")

Configure multiple endpoints for different use cases:

yaml

```yaml
endpoints:
  # Fast, cost-effective endpoint
  - name: fast-chat
    endpoint_type: llm/v1/chat
    model:
      provider: openai
      name: gpt-3.5-turbo
      config:
        openai_api_key: $OPENAI_API_KEY

  # High-quality endpoint
  - name: quality-chat
    endpoint_type: llm/v1/chat
    model:
      provider: openai
      name: gpt-4
      config:
        openai_api_key: $OPENAI_API_KEY

  # Embeddings endpoint
  - name: embeddings
    endpoint_type: llm/v1/embeddings
    model:
      provider: openai
      name: text-embedding-ada-002
      config:
        openai_api_key: $OPENAI_API_KEY

```

### Traffic route[​](#traffic-route "Direct link to Traffic route")

Add the `routes` configuration to split incoming traffic to multiple endpoints:

yaml

```yaml
endpoints:
  - name: chat1
    endpoint_type: llm/v1/chat
    model:
      provider: openai
      name: gpt-5
      config:
        openai_api_key: $OPENAI_API_KEY

  - name: chat2
    endpoint_type: llm/v1/chat
    model:
      provider: openai
      name: gpt-4.1
      config:
        openai_api_key: $OPENAI_API_KEY

routes:
  - name: chat-route
    task_type: llm/v1/chat
    destinations:
      - name: chat1
        traffic_percentage: 80
      - name: chat2
        traffic_percentage: 20
    routing_strategy: TRAFFIC_SPLIT

```

Currently, MLflow only support the `TRAFFIC_SPLIT` strategy which randomly route incoming requests based on the configured percentage.

## Dynamic Configuration Updates[​](#dynamic-configuration-updates "Direct link to Dynamic Configuration Updates")

The AI Gateway supports hot-reloading of configurations without server restart. Simply update your config.yaml file and changes are detected automatically.

## Security Best Practices[​](#security-best-practices "Direct link to Security Best Practices")

### API Key Management[​](#api-key-management "Direct link to API Key Management")

1. **Never commit API keys** to version control
2. **Use environment variables** for all sensitive credentials
3. **Rotate keys regularly** and update environment variables
4. **Use separate keys** for development and production

### Network Security[​](#network-security "Direct link to Network Security")

1. **Use HTTPS** in production with proper TLS certificates
2. **Implement authentication** and authorization layers
3. **Configure firewalls** to restrict access to the gateway
4. **Monitor and log** all gateway requests for audit trails

### Configuration Security[​](#configuration-security "Direct link to Configuration Security")

yaml

```yaml
# Secure configuration example
endpoints:
  - name: production-chat
    endpoint_type: llm/v1/chat
    model:
      provider: openai
      name: gpt-4
      config:
        openai_api_key: $OPENAI_API_KEY  # From environment
    limit:
      renewal_period: minute
      calls: 1000

```

## Next Steps[​](#next-steps "Direct link to Next Steps")

Now that your providers are configured, learn how to use and integrate your gateway:

### [Usage Guide](/mlflow-website/docs/latest/genai/governance/ai-gateway/usage.md)

[Query endpoints with Python client and REST APIs](/mlflow-website/docs/latest/genai/governance/ai-gateway/usage.md)

[Start using →](/mlflow-website/docs/latest/genai/governance/ai-gateway/usage.md)

### [Integration Guide](/mlflow-website/docs/latest/genai/governance/ai-gateway/integration.md)

[Integrate with applications, frameworks, and production systems](/mlflow-website/docs/latest/genai/governance/ai-gateway/integration.md)

[Learn integrations →](/mlflow-website/docs/latest/genai/governance/ai-gateway/integration.md)

### [Tutorial](/mlflow-website/docs/latest/genai/governance/ai-gateway/guides.md)

[Step-by-step walkthrough with examples](/mlflow-website/docs/latest/genai/governance/ai-gateway/guides.md)

[Follow tutorial →](/mlflow-website/docs/latest/genai/governance/ai-gateway/guides.md)
