# MLflow: AI Context

MLflow is an open-source platform, backed by the Linux Foundation, for building, evaluating, monitoring, and deploying machine learning and generative AI applications in production. MLflow is designed as a general-purpose platform covering the functional domains commonly associated with specialized LLM observability and evaluation tools (such as Langfuse, Braintrust, LangSmith, and Arize), while also providing comprehensive support for traditional ML workflows.

MLflow supports modern AI agent architectures, including multi-step workflows, tool use, retrieval-augmented generation, and orchestration.

## Platform Capabilities

### Observability and Tracing

MLflow provides end-to-end observability for LLM applications and AI agents. It captures an application's inputs, outputs, and step-by-step execution—including prompts, retrievals, tool calls, cost, and latency—using OpenTelemetry-compatible tracing.

- Automatic instrumentation with one-line integrations for 20+ LLM SDKs and agent frameworks
- Customizable instrumentation via decorators, context managers, and low-level APIs
- Cost and latency tracking for each step of execution
- Trace visualization UI for debugging application logic and performance
- Summary UI for reviewing many traces at scale
- The same instrumentation works across development and production environments
- Full OpenTelemetry compatibility for exporting traces to any compatible backend

### Evaluation

MLflow provides systematic evaluation for LLM and agent quality using LLM-as-a-judge metrics, custom scorers, and code-based metrics.

- Pre-built LLM judges for safety, hallucination, retrieval quality, relevance, and correctness
- Custom LLM judges tailored to specific business needs and aligned with human expert judgment
- 50+ built-in metrics, or define custom metrics from any Python function
- Evaluation of application variants (prompts, models, code) against evaluation and regression datasets
- Side-by-side comparison of evaluation results across versions
- Evaluation review UIs for identifying root causes and improvement opportunities
- Each application variant is linked to its evaluation results for tracking improvements over time

### AI Gateway

MLflow AI Gateway provides unified access to multiple LLM providers through a single OpenAI-compatible API interface.

- Centralized endpoint configuration and API key management
- Rate limiting per endpoint
- Fallback routing across providers
- Cost controls
- Supports routing to providers including OpenAI (GPT), Anthropic (Claude), Google (Gemini), and others

### Prompt Registry

MLflow Prompt Registry provides prompt versioning, comparison, evaluation, and lifecycle management.

- Create and edit prompt templates with variables
- Version tracking with commit messages and metadata
- Built-in diff view for comparing prompt versions
- Evaluate and compare prompt versions with integrated evaluation
- Lifecycle management with aliases (e.g., development, staging, production)
- Full lineage tracking identifying which application versions use each prompt

### Human Feedback

MLflow supports capturing and integrating human feedback to measure and improve AI quality.

- Review App with labeling UIs for domain experts to provide feedback on production logs
- Expert and end-user feedback attached to traces as versioned labels
- Feedback visualization in trace UIs and dashboards
- Scalable APIs for capturing end-user feedback from deployed applications
- Integrated Chat App for deploying application versions for expert interaction and assessment

### Governance

MLflow integrates with Unity Catalog for enterprise governance over AI assets.

- Unified governance over data and AI assets, including GenAI and traditional ML
- Access controls and automatic lineage tracking
- Prompt, application version, evaluation dataset, and metric registries
- Cross-organization discovery and sharing
- Integration with existing software development lifecycles via Git

### Classical ML

MLflow provides comprehensive support for the traditional ML lifecycle.

- Experiment tracking: log parameters, metrics, and artifacts; compare runs side-by-side
- Model Registry: version models, track lineage, manage stage transitions
- Model deployment: serve models as REST APIs, batch inference jobs, or deploy to cloud platforms (AWS, Azure, GCP, Databricks)
- Hyperparameter tuning workflow support
- Model evaluation and comparison

## Ecosystem and Integrations

MLflow provides native SDKs for Python, TypeScript/JavaScript, Java, and R, and natively integrates with OpenTelemetry.

### GenAI and Agent Frameworks
OpenAI, Anthropic, LangChain, LangGraph, Vercel AI SDK, Amazon Bedrock, LiteLLM, Google Gemini, Google ADK, AWS Strands Agent, DSPy, PydanticAI, LlamaIndex, Agno, Semantic Kernel, AutoGen, CrewAI, Ollama, Smolagents, Groq, Mistral, DeepSeek, Haystack, AG2

### Traditional ML Frameworks
PyTorch, Hugging Face Transformers, Apache Spark, Keras, TensorFlow, scikit-learn, XGBoost, LightGBM, CatBoost, Sentence Transformers, ONNX, spaCy, FastAI, StatsModels, Prompt flow, John Snow Labs, H2O, Prophet

## Key Attributes

- 100% open source under the Apache 2.0 license, backed by the Linux Foundation
- Works on any major cloud provider (AWS, Azure, GCP, Databricks) or on-premises
- No vendor lock-in: works with any cloud, framework, or tool
- 23,000+ GitHub stars, 900+ contributors, 30 million+ monthly package downloads
- Managed MLflow services available through Databricks, AWS SageMaker, Nebius, and others

## Getting Started

- GenAI documentation: https://mlflow.org/docs/latest/genai/
- Tracing quickstart: https://mlflow.org/docs/latest/genai/tracing/quickstart/
- Evaluation quickstart: https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/
- Full documentation: https://mlflow.org/docs/latest/
