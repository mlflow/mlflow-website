# Model Providers

MLflow AI Gateway supports 100+ model providers through the LiteLLM integration. This page covers the major providers, their capabilities, and how to use their passthrough APIs.

## Supported Providers[​](#supported-providers "Direct link to Supported Providers")

The AI Gateway supports providers across these categories:

### Major Cloud Providers[​](#major-cloud-providers "Direct link to Major Cloud Providers")

| Provider          | Chat | Embeddings | Passthrough API              |
| ----------------- | ---- | ---------- | ---------------------------- |
| **OpenAI**        | Yes  | Yes        | `/gateway/openai/v1/...`     |
| **Anthropic**     | Yes  | No         | `/gateway/anthropic/v1/...`  |
| **Google Gemini** | Yes  | Yes        | `/gateway/gemini/v1beta/...` |
| **Azure OpenAI**  | Yes  | Yes        | Via OpenAI passthrough       |
| **AWS Bedrock**   | Yes  | Yes        | -                            |
| **Vertex AI**     | Yes  | Yes        | -                            |

### Additional Providers[​](#additional-providers "Direct link to Additional Providers")

| Provider         | Chat | Embeddings | Notes                    |
| ---------------- | ---- | ---------- | ------------------------ |
| **Cohere**       | Yes  | Yes        | Command and Embed models |
| **Mistral**      | Yes  | Yes        | Mistral AI models        |
| **Groq**         | Yes  | No         | Open-source models       |
| **Together AI**  | Yes  | Yes        | Open-source models       |
| **Fireworks AI** | Yes  | Yes        | Open-source models       |
| **Ollama**       | Yes  | Yes        | Local models             |
| **Databricks**   | Yes  | Yes        | Foundation Model APIs    |

For a complete list of supported providers, view the provider dropdown when creating an endpoint or see the [LiteLLM documentation](https://docs.litellm.ai/docs/providers).

## Provider-Specific Passthrough APIs[​](#provider-specific-passthrough-apis "Direct link to Provider-Specific Passthrough APIs")

### OpenAI[​](#openai "Direct link to OpenAI")

The OpenAI passthrough exposes the full OpenAI API:

**Base URL:** `http://localhost:5000/gateway/openai/v1`

**Supported Endpoints:**

* `POST /chat/completions` - Chat completions
* `POST /embeddings` - Text embeddings
* `POST /responses` - Responses API (multi-turn conversations)

- Python SDK
- cURL

python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5000/gateway/openai/v1",
    api_key="dummy",  # Not needed, configured server-side
)

# Chat completion
response = client.chat.completions.create(
    model="my-endpoint",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Embeddings
embeddings = client.embeddings.create(
    model="my-embeddings-endpoint",
    input="Text to embed",
)

# Responses API
response = client.responses.create(
    model="my-endpoint",
    input="Hello!",
)

```

bash

```bash
# Chat completion
curl -X POST http://localhost:5000/gateway/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "my-endpoint", "messages": [{"role": "user", "content": "Hello!"}]}'

# Embeddings
curl -X POST http://localhost:5000/gateway/openai/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "my-embeddings-endpoint", "input": "Text to embed"}'

# Responses API
curl -X POST http://localhost:5000/gateway/openai/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model": "my-endpoint", "input": "Hello!"}'

```

See [OpenAI API Reference](https://platform.openai.com/docs/api-reference) for complete documentation.

### Anthropic[​](#anthropic "Direct link to Anthropic")

Access Claude models through the Anthropic passthrough:

**Base URL:** `http://localhost:5000/gateway/anthropic`

**Supported Endpoints:**

* `POST /v1/messages` - Messages API

- Python SDK
- cURL

python

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:5000/gateway/anthropic",
    api_key="dummy",  # Not needed, configured server-side
)

response = client.messages.create(
    model="my-endpoint",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.content[0].text)

```

bash

```bash
curl -X POST http://localhost:5000/gateway/anthropic/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-endpoint",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

```

See [Anthropic API Reference](https://docs.anthropic.com/en/api) for complete documentation.

### Google Gemini[​](#google-gemini "Direct link to Google Gemini")

Access Gemini models through Google's API format:

**Base URL:** `http://localhost:5000/gateway/gemini`

**Supported Endpoints:**

* `POST /v1beta/models/{model}:generateContent` - Content generation
* `POST /v1beta/models/{model}:streamGenerateContent` - Streaming generation

- Python SDK
- cURL

python

```python
from google import genai

client = genai.Client(
    api_key="dummy",
    http_options={
        "base_url": "http://localhost:5000/gateway/gemini",
    },
)

response = client.models.generate_content(
    model="my-endpoint",
    contents={"text": "Hello!"},
)
client.close()
print(response.candidates[0].content.parts[0].text)

```

bash

```bash
curl -X POST http://localhost:5000/gateway/gemini/v1beta/models/my-endpoint:generateContent \
  -H "Content-Type: application/json" \
  -d '{"contents": [{"parts": [{"text": "Hello!"}]}]}'

```

See [Google Gemini API Reference](https://ai.google.dev/gemini-api/docs) for complete documentation.

### Azure OpenAI[​](#azure-openai "Direct link to Azure OpenAI")

Azure OpenAI uses the same passthrough as OpenAI with additional configuration:

**Base URL:** `http://localhost:5000/gateway/openai/v1`

When creating an Azure OpenAI endpoint:

1. Select **Azure OpenAI** as the provider
2. Enter your Azure endpoint URL
3. Enter your Azure API key
4. Specify your deployment name

python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5000/gateway/openai/v1",
    api_key="dummy",
)

response = client.chat.completions.create(
    model="my-azure-endpoint",
    messages=[{"role": "user", "content": "Hello!"}],
)

```

### Databricks Foundation Models[​](#databricks-foundation-models "Direct link to Databricks Foundation Models")

Databricks Foundation Models APIs are OpenAI-compatible:

**Base URL:** `http://localhost:5000/gateway/openai/v1`

When creating a Databricks endpoint:

1. Select **Databricks** as the provider
2. Enter your Databricks workspace URL
3. Enter your Databricks personal access token
4. Specify the model endpoint name

python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5000/gateway/openai/v1",
    api_key="dummy",
)

response = client.chat.completions.create(
    model="my-databricks-endpoint",
    messages=[{"role": "user", "content": "Hello!"}],
)

```

## Model Capabilities[​](#model-capabilities "Direct link to Model Capabilities")

When creating endpoints, the model selector shows capability badges:

| Badge         | Description                                  |
| ------------- | -------------------------------------------- |
| **Tools**     | Model supports function/tool calling         |
| **Reasoning** | Model has enhanced reasoning capabilities    |
| **Caching**   | Model supports prompt caching for efficiency |
| **Vision**    | Model can process images                     |

Additional information displayed:

* **Context window**: Maximum tokens the model can process
* **Token costs**: Input and output pricing per million tokens
