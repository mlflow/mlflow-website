# Getting Started with MLflow for GenAI

## The Complete Open Source LLMOps Platform for Production GenAI[​](#the-complete-open-source-llmops-platform-for-production-genai "Direct link to The Complete Open Source LLMOps Platform for Production GenAI")

MLflow transforms how software engineers build, evaluate, and deploy GenAI applications. Get complete observability, systematic evaluation, and deployment confidence—all while maintaining the flexibility to use any framework or model provider.

![MLflow Tracing UI showing detailed GenAI observability](/mlflow-website/docs/latest/images/llms/tracing/tracing-top.gif)

## The GenAI Development Lifecycle[​](#the-genai-development-lifecycle "Direct link to The GenAI Development Lifecycle")

MLflow provides a complete platform that supports every stage of GenAI application development. From initial prototyping to production monitoring, these integrated capabilities ensure you can build, test, and deploy with confidence.

#### Develop & Debug

Trace every LLM call, prompt interaction, and tool invocation. Debug complex AI workflows with complete visibility into execution paths, token usage, and decision points.

#### Evaluate & Improve

Systematically test with LLM judges, human feedback, and custom metrics. Compare versions objectively and catch regressions before they reach production.

#### Deploy & Monitor

Serve models with confidence using built-in deployment targets. Monitor production performance and iterate based on real-world usage patterns.

## Why Open Source MLflow for GenAI?[​](#why-open-source-mlflow-for-genai "Direct link to Why Open Source MLflow for GenAI?")

As the original open source ML platform, MLflow brings battle-tested reliability and community-driven innovation to GenAI development. No vendor lock-in, no proprietary formats—just powerful tools that work with your stack.

#### Production-Grade Observability

Automatically instrument 15+ frameworks including OpenAI, LangChain, and LlamaIndex. Get detailed traces showing token usage, latency, and execution paths for every request—no black boxes.

#### Intelligent Prompt Management

Version, compare, and deploy prompts with MLflow's prompt registry. Track performance across prompt variations and maintain audit trails for production systems.

#### Automated Quality Assurance

Build confidence with LLM judges and automated evaluation. Run systematic tests on every change and track quality metrics over time to prevent regressions.

#### Framework-Agnostic Integration

Use any LLM framework or provider without vendor lock-in. MLflow works with your existing tools while providing unified tracking, evaluation, and deployment.

## Start Building Production GenAI Applications[​](#start-building-production-genai-applications "Direct link to Start Building Production GenAI Applications")

MLflow transforms GenAI development from complex instrumentation to simple, one-line integrations. See how easy it is to add comprehensive observability, evaluation, and deployment to your AI applications. Visit the [Tracing guide](/mlflow-website/docs/latest/genai/tracing.md) for more information.

### Add Complete Observability in One Line[​](#add-complete-observability-in-one-line "Direct link to Add Complete Observability in One Line")

Transform any GenAI application into a fully observable system:

python

```python
import mlflow

# Enable automatic tracing for your framework
mlflow.openai.autolog()  # For OpenAI
mlflow.langchain.autolog()  # For LangChain
mlflow.llama_index.autolog()  # For LlamaIndex
mlflow.dspy.autolog()  # For DSPy

# Your existing code now generates detailed traces
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
)
# ✅ Automatically traced: tokens, latency, cost, full request/response

```

No code changes required. Every LLM call, tool interaction, and prompt execution is automatically captured with detailed metrics.

### Manage and Optimize Prompts Systematically[​](#manage-and-optimize-prompts-systematically "Direct link to Manage and Optimize Prompts Systematically")

Register prompts and automatically optimize them with data-driven techniques. See the [Prompt Registry](/mlflow-website/docs/latest/genai/prompt-registry/create-and-edit-prompts.md) guide for comprehensive prompt management:

python

```python
import mlflow
import openai
from mlflow.genai.optimize import GepaPromptOptimizer
from mlflow.genai.scorers import Correctness

# Register an initial prompt
prompt = mlflow.genai.register_prompt(
    name="math_tutor",
    template="Answer this math question: {{question}}. Provide a clear explanation.",
)


# Define prediction function that includes prompt.format() call for your target prompt(s)
def predict_fn(question: str) -> str:
    prompt = mlflow.genai.load_prompt("prompts:/math_tutor@latest")
    completion = openai.OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt.format(question=question)}],
    )
    return completion.choices[0].message.content


# Prepare training data with inputs and expectations
train_data = [
    {
        "inputs": {"question": "What is 15 + 27?"},
        "expectations": {"expected_response": "42"},
    },
    {
        "inputs": {"question": "Calculate 8 × 9"},
        "expectations": {"expected_response": "72"},
    },
    {
        "inputs": {"question": "What is 100 - 37?"},
        "expectations": {"expected_response": "63"},
    },
    # ... more examples
]

# Automatically optimize the prompt using MLflow + GEPA
result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=train_data,
    prompt_uris=[prompt.uri],
    optimizer=GepaPromptOptimizer(reflection_model="openai:/gpt-4o-mini"),
    scorers=[Correctness(model="openai:/gpt-4o-mini")],
)

# The optimized prompt is automatically registered as a new version
optimized_prompt = result.optimized_prompts[0]
print(f"Optimized prompt registered as version {optimized_prompt.version}")
print(f"Template: {optimized_prompt.template}")
print(f"Score: {result.final_eval_score}")

```

Transform manual prompt engineering into systematic, data-driven optimization with automatic performance tracking. Learn more in the [Optimize Prompts](/mlflow-website/docs/latest/genai/prompt-registry/optimize-prompts.md) guide.

### Prerequisites[​](#prerequisites "Direct link to Prerequisites")

Ready to get started? You'll need:

* Python 3.10+ installed
* MLflow 3.5+ (`pip install --upgrade mlflow`)
* API access to an LLM provider (OpenAI, Anthropic, etc.)

***

## Essential Learning Path[​](#essential-learning-path "Direct link to Essential Learning Path")

Master these core capabilities to build robust GenAI applications with MLflow. Start with observability, then add systematic evaluation and deployment.

### [Environment Setup](/mlflow-website/docs/latest/genai/getting-started/connect-environment.md)

[Configure MLflow tracking, connect to registries, and set up your development environment for GenAI workflows](/mlflow-website/docs/latest/genai/getting-started/connect-environment.md)

[Start setup →](/mlflow-website/docs/latest/genai/getting-started/connect-environment.md)

### [Observability with Tracing](/mlflow-website/docs/latest/genai/tracing/quickstart.md)

[Auto-instrument your GenAI application to capture every LLM call, prompt, and tool interaction for complete visibility](/mlflow-website/docs/latest/genai/tracing/quickstart.md)

[Learn tracing →](/mlflow-website/docs/latest/genai/tracing/quickstart.md)

### [Systematic Evaluation](/mlflow-website/docs/latest/genai/eval-monitor.md)

[Build confidence with LLM judges and automated testing to catch quality issues before production](/mlflow-website/docs/latest/genai/eval-monitor.md)

[Start evaluating →](/mlflow-website/docs/latest/genai/eval-monitor.md)

These three foundations will give you the observability and quality confidence needed for production GenAI development. Each tutorial includes real code examples and best practices from production deployments.

***

## Advanced GenAI Capabilities[​](#advanced-genai-capabilities "Direct link to Advanced GenAI Capabilities")

Once you've mastered the essentials, explore these advanced features to build sophisticated GenAI applications with enterprise-grade reliability.

### [Prompt Registry & Management](/mlflow-website/docs/latest/genai/prompt-registry/prompt-engineering.md)

[Version prompts, A/B test variations, and maintain audit trails for production prompt management](/mlflow-website/docs/latest/genai/prompt-registry/prompt-engineering.md)

[Manage prompts →](/mlflow-website/docs/latest/genai/prompt-registry/prompt-engineering.md)

### [Automated Prompt Optimization](/mlflow-website/docs/latest/genai/prompt-registry/optimize-prompts.md)

[Automatically improve prompts using DSPy's MIPROv2 algorithm with data-driven optimization and performance tracking](/mlflow-website/docs/latest/genai/prompt-registry/optimize-prompts.md)

[Optimize prompts →](/mlflow-website/docs/latest/genai/prompt-registry/optimize-prompts.md)

### [Model Deployment](/mlflow-website/docs/latest/genai/serving.md)

[Deploy GenAI models to production with built-in serving, scaling, and monitoring capabilities](/mlflow-website/docs/latest/genai/serving.md)

[Deploy models →](/mlflow-website/docs/latest/genai/serving.md)

These capabilities enable you to build production-ready GenAI applications with systematic quality management and robust deployment infrastructure.

***

## Framework-Specific Integration Guides[​](#framework-specific-integration-guides "Direct link to Framework-Specific Integration Guides")

MLflow provides deep integrations with popular GenAI frameworks. Choose your framework to get started with optimized instrumentation and best practices.

[![LangChain Integration](/mlflow-website/docs/latest/images/logos/langchain-logo.png)](/mlflow-website/docs/latest/genai/flavors/langchain.md)

### [LangChain Integration](/mlflow-website/docs/latest/genai/flavors/langchain.md)

[Auto-trace chains, agents, and tools with comprehensive LangChain instrumentation](/mlflow-website/docs/latest/genai/flavors/langchain.md)

[Use LangChain →](/mlflow-website/docs/latest/genai/flavors/langchain.md)

[![LlamaIndex Integration](/mlflow-website/docs/latest/images/logos/llamaindex-logo.svg)](/mlflow-website/docs/latest/genai/flavors/llama-index.md)

### [LlamaIndex Integration](/mlflow-website/docs/latest/genai/flavors/llama-index.md)

[Instrument RAG pipelines and document processing workflows with LlamaIndex support](/mlflow-website/docs/latest/genai/flavors/llama-index.md)

[Use LlamaIndex →](/mlflow-website/docs/latest/genai/flavors/llama-index.md)

[![OpenAI Integration](/mlflow-website/docs/latest/images/logos/openai-logo.svg)](/mlflow-website/docs/latest/genai/flavors/openai.md)

### [OpenAI Integration](/mlflow-website/docs/latest/genai/flavors/openai.md)

[Track completions, embeddings, and function calls with native OpenAI instrumentation](/mlflow-website/docs/latest/genai/flavors/openai.md)

[Use OpenAI →](/mlflow-website/docs/latest/genai/flavors/openai.md)

### [DSPy Integration](/mlflow-website/docs/latest/genai/flavors/dspy.md)

[Build systematic prompt optimization workflows with DSPy modules and MLflow prompt registry](/mlflow-website/docs/latest/genai/flavors/dspy.md)

[Use DSPy →](/mlflow-website/docs/latest/genai/flavors/dspy.md)

### [Custom Framework Support](/mlflow-website/docs/latest/genai/flavors/chat-model-intro.md)

[Instrument any LLM framework or build custom integrations with MLflow's flexible APIs](/mlflow-website/docs/latest/genai/flavors/chat-model-intro.md)

[Build custom →](/mlflow-website/docs/latest/genai/flavors/chat-model-intro.md)

Each integration guide includes framework-specific examples, best practices, and optimization techniques for production deployments.

***

## Start Your GenAI Journey with MLflow[​](#start-your-genai-journey-with-mlflow "Direct link to Start Your GenAI Journey with MLflow")

Ready to build production-ready GenAI applications? Start with the Environment Setup guide above, then explore tracing for complete observability into your AI systems. Join thousands of engineers who trust MLflow's open source platform for their GenAI development.
