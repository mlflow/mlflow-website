# MLflow GenAI Packaging Integrations

MLflow 3 delivers built-in support for packaging and deploying applications written with the GenAI frameworks you depend on. Whether you're calling OpenAI directly, orchestrating chains with LangChain or LangGraph, indexing documents in LlamaIndex, wiring up agent patterns via ChatModel and ResponseAgent, or rolling your own with a PythonModel, MLflow provides native packaging and deployment APIs ("flavors") to streamline your path to production.

## Why MLflow Integrations?[​](#why-mlflow-integrations "Direct link to Why MLflow Integrations?")

By choosing MLflow's native flavors, you gain end-to-end visibility and control without swapping tools:

* **Unified Tracking & Models**: All calls, parameters, artifacts, and prompt templates become tracked entities within MLflow Experiments. Serialized GenAI application code becomes a LoggedModel—viewable and referenceable within the MLflow UI and APIs.
* **Zero-Boilerplate Setup**: A single `mlflow.<flavor>.log_model(...)` call (or one line of auto-instrumentation) wires into your existing code.
* **Reproducibility by Default**: MLflow freezes your prompt template, application parameters, framework versions, and dependencies so you can reproduce any result, anytime.
* **Seamless Transition to Serving**: Each integration produces a standardized MLflow Model you can deploy for batch scoring or real-time inference with `mlflow models serve`.

***

## Start Integrating in Minutes[​](#start-integrating-in-minutes "Direct link to Start Integrating in Minutes")

Before you begin, make sure you have:

* Python 3.9+ and MLflow 3.x installed (`pip install --upgrade mlflow`)
* Credentials or API keys for your chosen provider (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
* An MLflow Tracking Server (local or remote)

Ready to dive in?

Pick your integration from the list below and follow the concise guide—each one gets you up and running in under 10 minutes.

***

## Integration Guides[​](#integration-guides "Direct link to Integration Guides")

MLflow supports first-party flavors for these GenAI frameworks and patterns. Click to explore:

[![OpenAI Logo](/mlflow-website/docs/latest/assets/images/openai-logo-84ce36fa9f59f4df880cee88c0335586.png)](/mlflow-website/docs/latest/genai/flavors/openai.md)

[![LangChain Logo](/mlflow-website/docs/latest/assets/images/langchain-logo-39d51f94cc9aebac2c191cca0e8189de.png)](/mlflow-website/docs/latest/genai/flavors/langchain.md)

[![LlamaIndex Logo](/mlflow-website/docs/latest/assets/images/llamaindex-logo-dd13e5b1cfc2b77ac4bcd5a6a1d2b5af.svg)](/mlflow-website/docs/latest/genai/flavors/llama-index.md)

[![DSPy Logo](/mlflow-website/docs/latest/assets/images/dspy-logo-b3072c635a8fbacb2c6f9336db69a704.png)](/mlflow-website/docs/latest/genai/flavors/dspy.md)

[![HuggingFace Logo](/mlflow-website/docs/latest/assets/images/huggingface-logo-13b52ccce8bdc067c7f668157e4389f0.svg)](/mlflow-website/docs/latest/ml/deep-learning/transformers.md)

[![SentenceTransformers Logo](/mlflow-website/docs/latest/assets/images/sentence-transformers-logo-db10cc6f5052a7c08b1fd8a526e674a0.png)](/mlflow-website/docs/latest/ml/deep-learning/sentence-transformers.md)

[**Custom Packaging with ChatModel**](/mlflow-website/docs/latest/genai/flavors/chat-model-intro.md)

[**Custom Packaging with PythonModel**](/mlflow-website/docs/latest/genai/flavors/custom-pyfunc-for-llms.md)

***

## Continue Your Journey[​](#continue-your-journey "Direct link to Continue Your Journey")

Once your integration is in place, take advantage of MLflow's full LLMOps platform:

### 🔍 Observability & Debugging[​](#-observability--debugging "Direct link to 🔍 Observability & Debugging")

* [Tracing & Observability](/mlflow-website/docs/latest/genai/tracing.md)

### 🧪 Evaluation & QA[​](#-evaluation--qa "Direct link to 🧪 Evaluation & QA")

* [LLM Evaluation Framework](https://docs.databricks.com/aws/en/mlflow3/genai/getting-started/eval.html)

### 🚀 Deployment & Monitoring[​](#-deployment--monitoring "Direct link to 🚀 Deployment & Monitoring")

* [Prompt Engineering UI](/mlflow-website/docs/latest/genai/prompt-registry/prompt-engineering.md)
* [Application Serving](/mlflow-website/docs/latest/genai/serving.md)
