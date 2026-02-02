# Automatic Tracing

MLflow Tracing is integrated with various GenAI libraries and provides **one-line automatic tracing** experience for each library (and the combination of them!). This page shows detailed examples to integrate MLflow with popular GenAI libraries.

[](/mlflow-website/docs/latest/images/llms/tracing/tracing-top.mp4)

## Supported Integrations[​](#supported-integrations "Direct link to Supported Integrations")

Each integration automatically captures your application's logic and intermediate steps based on your implementation of the authoring framework / SDK. Click on the logo of your library to see the detailed integration guide.

### OpenTelemetry[​](#opentelemetry "Direct link to OpenTelemetry")

[![OpenTelemetry Logo](/mlflow-website/docs/latest/images/logos/opentelemetry-logo-only.png)](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/opentelemetry.md)

[OpenTelemetry](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/opentelemetry.md)

### Agent Frameworks (Python)[​](#agent-frameworks-python "Direct link to Agent Frameworks (Python)")

[![LangChain Logo](/mlflow-website/docs/latest/images/logos/langchain-logo-only.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/langchain.md)

[LangChain](/mlflow-website/docs/latest/genai/tracing/integrations/listing/langchain.md)

[![LangGraph Logo](/mlflow-website/docs/latest/images/logos/langgraph-logo-only.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/langgraph.md)

[LangGraph](/mlflow-website/docs/latest/genai/tracing/integrations/listing/langgraph.md)

[![OpenAI Agent Logo](/mlflow-website/docs/latest/images/logos/openai-logo-only.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/openai-agent.md)

[OpenAI Agent](/mlflow-website/docs/latest/genai/tracing/integrations/listing/openai-agent.md)

[![DSPy Logo](/mlflow-website/docs/latest/images/logos/dspy-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/dspy.md)

[DSPy](/mlflow-website/docs/latest/genai/tracing/integrations/listing/dspy.md)

[![PydanticAI Logo](/mlflow-website/docs/latest/images/logos/pydantic-ai-logo-only.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/pydantic_ai.md)

[PydanticAI](/mlflow-website/docs/latest/genai/tracing/integrations/listing/pydantic_ai.md)

[![Google ADK Logo](/mlflow-website/docs/latest/images/logos/google-adk-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/google-adk.md)

[Google ADK](/mlflow-website/docs/latest/genai/tracing/integrations/listing/google-adk.md)

[![Microsoft Agent Framework Logo](/mlflow-website/docs/latest/images/logos/microsoft-agent-framework-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/microsoft-agent-framework.md)

[Microsoft Agent Framework](/mlflow-website/docs/latest/genai/tracing/integrations/listing/microsoft-agent-framework.md)

[![CrewAI Logo](/mlflow-website/docs/latest/images/logos/crewai-logo.svg)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/crewai.md)

[CrewAI](/mlflow-website/docs/latest/genai/tracing/integrations/listing/crewai.md)

[![LlamaIndex Logo](/mlflow-website/docs/latest/images/logos/llamaindex-logo.svg)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/llama_index.md)

[LlamaIndex](/mlflow-website/docs/latest/genai/tracing/integrations/listing/llama_index.md)

[![AutoGen Logo](/mlflow-website/docs/latest/images/logos/autogen-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/autogen.md)

[AutoGen](/mlflow-website/docs/latest/genai/tracing/integrations/listing/autogen.md)

[![Strands Agent SDK Logo](/mlflow-website/docs/latest/images/logos/strands-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/strands.md)

[Strands Agent SDK](/mlflow-website/docs/latest/genai/tracing/integrations/listing/strands.md)

[![Agno Logo](/mlflow-website/docs/latest/images/logos/agno-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/agno.md)

[Agno](/mlflow-website/docs/latest/genai/tracing/integrations/listing/agno.md)

[![Amazon Bedrock AgentCore Logo](/mlflow-website/docs/latest/images/logos/bedrock-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/bedrock-agentcore.md)

[Amazon Bedrock AgentCore](/mlflow-website/docs/latest/genai/tracing/integrations/listing/bedrock-agentcore.md)

[![Smolagents Logo](/mlflow-website/docs/latest/images/logos/smolagents-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/smolagents.md)

[Smolagents](/mlflow-website/docs/latest/genai/tracing/integrations/listing/smolagents.md)

[![Semantic Kernel Logo](/mlflow-website/docs/latest/images/logos/semantic-kernel-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/semantic_kernel.md)

[Semantic Kernel](/mlflow-website/docs/latest/genai/tracing/integrations/listing/semantic_kernel.md)

[![LangChain DeepAgent Logo](/mlflow-website/docs/latest/images/logos/deepagent-logo.svg)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/deepagent.md)

[LangChain DeepAgent](/mlflow-website/docs/latest/genai/tracing/integrations/listing/deepagent.md)

[![AG2 Logo](/mlflow-website/docs/latest/images/logos/ag2-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/ag2.md)

[AG2](/mlflow-website/docs/latest/genai/tracing/integrations/listing/ag2.md)

[![Haystack Logo](/mlflow-website/docs/latest/images/logos/haystack-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/haystack.md)

[Haystack](/mlflow-website/docs/latest/genai/tracing/integrations/listing/haystack.md)

[![Koog Logo](/mlflow-website/docs/latest/images/logos/koog.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/koog.md)

[Koog](/mlflow-website/docs/latest/genai/tracing/integrations/listing/koog.md)

[![txtai Logo](/mlflow-website/docs/latest/images/logos/txtai-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/txtai.md)

[txtai](/mlflow-website/docs/latest/genai/tracing/integrations/listing/txtai.md)

[![Pipecat Logo](/mlflow-website/docs/latest/images/logos/pipecat.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/pipecat.md)

[Pipecat](/mlflow-website/docs/latest/genai/tracing/integrations/listing/pipecat.md)

[![Watsonx Orchestrate Logo](/mlflow-website/docs/latest/images/logos/watsonx-orchestrate.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/watsonx-orchestrate.md)

[Watsonx Orchestrate](/mlflow-website/docs/latest/genai/tracing/integrations/listing/watsonx-orchestrate.md)

### Agent Frameworks (TypeScript)[​](#agent-frameworks-typescript "Direct link to Agent Frameworks (TypeScript)")

[![LangChain Logo](/mlflow-website/docs/latest/images/logos/langchain-logo-only.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/langchain.md)

[LangChain](/mlflow-website/docs/latest/genai/tracing/integrations/listing/langchain.md)

[![LangGraph Logo](/mlflow-website/docs/latest/images/logos/langgraph-logo-only.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/langgraph.md)

[LangGraph](/mlflow-website/docs/latest/genai/tracing/integrations/listing/langgraph.md)

[![Vercel AI SDK Logo](/mlflow-website/docs/latest/images/logos/vercel-logo.svg)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/vercelai.md)

[Vercel AI SDK](/mlflow-website/docs/latest/genai/tracing/integrations/listing/vercelai.md)

[![Mastra Logo](/mlflow-website/docs/latest/images/logos/mastra-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/mastra.md)

[Mastra](/mlflow-website/docs/latest/genai/tracing/integrations/listing/mastra.md)

[![VoltAgent Logo](/mlflow-website/docs/latest/images/logos/voltagent-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/voltagent.md)

[VoltAgent](/mlflow-website/docs/latest/genai/tracing/integrations/listing/voltagent.md)

### Model Providers[​](#model-providers "Direct link to Model Providers")

[![OpenAI Logo](/mlflow-website/docs/latest/images/logos/openai-logo-only.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/openai.md)

[OpenAI](/mlflow-website/docs/latest/genai/tracing/integrations/listing/openai.md)

[![Anthropic Logo](/mlflow-website/docs/latest/images/logos/anthropic-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/anthropic.md)

[Anthropic](/mlflow-website/docs/latest/genai/tracing/integrations/listing/anthropic.md)

[![Databricks Logo](/mlflow-website/docs/latest/images/logos/databricks-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/databricks.md)

[Databricks](/mlflow-website/docs/latest/genai/tracing/integrations/listing/databricks.md)

[![Gemini Logo](/mlflow-website/docs/latest/images/logos/google-gemini-logo.svg)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/gemini.md)

[Gemini](/mlflow-website/docs/latest/genai/tracing/integrations/listing/gemini.md)

[![Amazon Bedrock Logo](/mlflow-website/docs/latest/images/logos/bedrock-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/bedrock.md)

[Amazon Bedrock](/mlflow-website/docs/latest/genai/tracing/integrations/listing/bedrock.md)

[![LiteLLM Logo](/mlflow-website/docs/latest/images/logos/litellm-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/litellm.md)

[LiteLLM](/mlflow-website/docs/latest/genai/tracing/integrations/listing/litellm.md)

[![Mistral Logo](/mlflow-website/docs/latest/images/logos/mistral-ai-logo.svg)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/mistral.md)

[Mistral](/mlflow-website/docs/latest/genai/tracing/integrations/listing/mistral.md)

[![xAI / Grok Logo](/mlflow-website/docs/latest/images/logos/grok-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/xai-grok.md)

[xAI / Grok](/mlflow-website/docs/latest/genai/tracing/integrations/listing/xai-grok.md)

[![Ollama Logo](/mlflow-website/docs/latest/images/logos/ollama-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/ollama.md)

[Ollama](/mlflow-website/docs/latest/genai/tracing/integrations/listing/ollama.md)

[![Groq Logo](/mlflow-website/docs/latest/images/logos/groq-logo.svg)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/groq.md)

[Groq](/mlflow-website/docs/latest/genai/tracing/integrations/listing/groq.md)

[![DeepSeek Logo](/mlflow-website/docs/latest/images/logos/deepseek-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/deepseek.md)

[DeepSeek](/mlflow-website/docs/latest/genai/tracing/integrations/listing/deepseek.md)

[![Qwen Logo](/mlflow-website/docs/latest/images/logos/qwen-logo.jpg)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/qwen.md)

[Qwen](/mlflow-website/docs/latest/genai/tracing/integrations/listing/qwen.md)

[![Moonshot AI Logo](/mlflow-website/docs/latest/images/logos/kimi-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/moonshot.md)

[Moonshot AI](/mlflow-website/docs/latest/genai/tracing/integrations/listing/moonshot.md)

[![Cohere Logo](/mlflow-website/docs/latest/images/logos/cohere-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/cohere.md)

[Cohere](/mlflow-website/docs/latest/genai/tracing/integrations/listing/cohere.md)

[![BytePlus Logo](/mlflow-website/docs/latest/images/logos/byteplus-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/byteplus.md)

[BytePlus](/mlflow-website/docs/latest/genai/tracing/integrations/listing/byteplus.md)

[![Novita AI Logo](/mlflow-website/docs/latest/images/logos/novitaai-logo.jpg)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/novitaai.md)

[Novita AI](/mlflow-website/docs/latest/genai/tracing/integrations/listing/novitaai.md)

[![FireworksAI Logo](/mlflow-website/docs/latest/images/logos/fireworks-ai-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/fireworksai.md)

[FireworksAI](/mlflow-website/docs/latest/genai/tracing/integrations/listing/fireworksai.md)

[![Together AI Logo](/mlflow-website/docs/latest/images/logos/together-ai-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/togetherai.md)

[Together AI](/mlflow-website/docs/latest/genai/tracing/integrations/listing/togetherai.md)

### Tools[​](#tools "Direct link to Tools")

[![Instructor Logo](/mlflow-website/docs/latest/images/logos/instructor-logo.svg)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/instructor.md)

[Instructor](/mlflow-website/docs/latest/genai/tracing/integrations/listing/instructor.md)

[![Claude Code Logo](/mlflow-website/docs/latest/images/logos/claude-code-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/claude_code.md)

[Claude Code](/mlflow-website/docs/latest/genai/tracing/integrations/listing/claude_code.md)

### Gateways[​](#gateways "Direct link to Gateways")

[![MLflow AI Gateway Logo](/mlflow-website/docs/latest/images/logos/mlflow-logo.svg)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/mlflow-ai-gateway.md)

[MLflow AI Gateway](/mlflow-website/docs/latest/genai/tracing/integrations/listing/mlflow-ai-gateway.md)

[![Databricks Logo](/mlflow-website/docs/latest/images/logos/databricks-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/databricks-ai-gateway.md)

[Databricks](/mlflow-website/docs/latest/genai/tracing/integrations/listing/databricks-ai-gateway.md)

[![LiteLLM Proxy Logo](/mlflow-website/docs/latest/images/logos/litellm-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/litellm-proxy.md)

[LiteLLM Proxy](/mlflow-website/docs/latest/genai/tracing/integrations/listing/litellm-proxy.md)

[![Vercel AI Gateway Logo](/mlflow-website/docs/latest/images/logos/vercel-logo.svg)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/vercel-ai-gateway.md)

[Vercel AI Gateway](/mlflow-website/docs/latest/genai/tracing/integrations/listing/vercel-ai-gateway.md)

[![OpenRouter Logo](/mlflow-website/docs/latest/images/logos/openrouter-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/openrouter.md)

[OpenRouter](/mlflow-website/docs/latest/genai/tracing/integrations/listing/openrouter.md)

[![Portkey Logo](/mlflow-website/docs/latest/images/logos/portkey-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/portkey.md)

[Portkey](/mlflow-website/docs/latest/genai/tracing/integrations/listing/portkey.md)

[![Helicone Logo](/mlflow-website/docs/latest/images/logos/helicone-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/helicone.md)

[Helicone](/mlflow-website/docs/latest/genai/tracing/integrations/listing/helicone.md)

[![Kong AI Gateway Logo](/mlflow-website/docs/latest/images/logos/kong-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/kong.md)

[Kong AI Gateway](/mlflow-website/docs/latest/genai/tracing/integrations/listing/kong.md)

[![Pydantic AI Gateway Logo](/mlflow-website/docs/latest/images/logos/pydantic-ai-logo-only.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/pydantic-ai-gateway.md)

[Pydantic AI Gateway](/mlflow-website/docs/latest/genai/tracing/integrations/listing/pydantic-ai-gateway.md)

[![TrueFoundry Logo](/mlflow-website/docs/latest/images/logos/truefoundry-logo.png)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/truefoundry.md)

[TrueFoundry](/mlflow-website/docs/latest/genai/tracing/integrations/listing/truefoundry.md)

### No-Code[​](#no-code "Direct link to No-Code")

[![Langflow Logo](/mlflow-website/docs/latest/images/logos/langflow.svg)](/mlflow-website/docs/latest/genai/tracing/integrations/listing/langflow.md)

[Langflow](/mlflow-website/docs/latest/genai/tracing/integrations/listing/langflow.md)

<br />

Missing your favorite library?

Is your favorite library missing from the list? Consider [contributing to MLflow Tracing](/mlflow-website/docs/latest/genai/tracing/integrations/contribute.md) or [submitting a feature request](https://github.com/mlflow/mlflow/issues/new?assignees=\&labels=enhancement\&projects=\&template=feature_request_template.yaml\&title=%5BFR%5D) to our Github repository.

## Advanced Usage[​](#advanced-usage "Direct link to Advanced Usage")

### Combining Manual and Automatic Tracing[​](#combining-manual-and-automatic-tracing "Direct link to Combining Manual and Automatic Tracing")

The `@mlflow.trace` decorator can be used in conjunction with auto tracing to create powerful, integrated traces. This is particularly useful for:

1. 🔄 **Complex workflows** that involve multiple LLM calls
2. 🤖 **Multi-agent systems** where different agents use different LLM providers
3. 🔗 **Chaining multiple LLM calls** together with custom logic in between

Here's a simple example that combines OpenAI auto-tracing with manually defined spans:

python

```python
import mlflow
import openai
from mlflow.entities import SpanType

mlflow.openai.autolog()


@mlflow.trace(span_type=SpanType.CHAIN)
def run(question):
    messages = build_messages(question)
    # MLflow automatically generates a span for OpenAI invocation
    response = openai.OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=100,
        messages=messages,
    )
    return parse_response(response)


@mlflow.trace
def build_messages(question):
    return [
        {"role": "system", "content": "You are a helpful chatbot."},
        {"role": "user", "content": question},
    ]


@mlflow.trace
def parse_response(response):
    return response.choices[0].message.content


run("What is MLflow?")

```

Running this code generates a single trace that combines the manual spans with the automatic OpenAI tracing.

### Multi-Framework Example[​](#multi-framework-example "Direct link to Multi-Framework Example")

You can also combine different LLM providers in a single trace. For example:

note

This example requires installing LangChain in addition to the base requirements:

bash

```bash
pip install --upgrade langchain langchain-openai

```

python

```python
import mlflow
import openai
from mlflow.entities import SpanType
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Enable auto-tracing for both OpenAI and LangChain
mlflow.openai.autolog()
mlflow.langchain.autolog()


@mlflow.trace(span_type=SpanType.CHAIN)
def multi_provider_workflow(query: str):
    # First, use OpenAI directly for initial processing
    analysis = openai.OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Analyze the query and extract key topics."},
            {"role": "user", "content": query},
        ],
    )
    topics = analysis.choices[0].message.content

    # Then use LangChain for structured processing
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template(
        "Based on these topics: {topics}\nGenerate a detailed response to: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"topics": topics, "query": query})

    return response


# Run the function
result = multi_provider_workflow("Explain quantum computing")

```

## Disabling Tracing[​](#disabling-tracing "Direct link to Disabling Tracing")

To **disable** tracing, the [`mlflow.tracing.disable()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.tracing.html#mlflow.tracing.disable) API will cease the collection of trace data from within MLflow and will not log any data to the MLflow Tracking service regarding traces.

To **enable** tracing (if it had been temporarily disabled), the [`mlflow.tracing.enable()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.tracing.html#mlflow.tracing.enable) API will re-enable tracing functionality for instrumented models that are invoked.

## Next Steps[​](#next-steps "Direct link to Next Steps")

**[Manual Tracing](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/manual-tracing.md)**: Learn how to add custom tracing to your application logic

**[Integration Guides](/mlflow-website/docs/latest/genai/tracing/integrations.md)**: Explore detailed guides for specific libraries and frameworks

**[Viewing Traces](/mlflow-website/docs/latest/genai/tracing/observe-with-traces/ui.md)**: Learn how to explore and analyze your traces in the MLflow UI

**[Querying Traces](/mlflow-website/docs/latest/genai/tracing/search-traces.md)**: Programmatically search and retrieve trace data for analysis
