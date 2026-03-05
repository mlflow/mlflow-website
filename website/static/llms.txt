# MLflow

> MLflow is the largest open source AI engineering platform. MLflow enables teams of all sizes to debug, evaluate, monitor, and optimize production-quality AI agents, LLM applications, and ML models while controlling costs and managing access to models and data. With over 30 million monthly downloads, thousands of organizations rely on MLflow each day to ship AI to production with confidence.

MLflow provides two main feature sets:

1. **LLMs & AI Agents**: Production-grade observability (tracing), evaluation with LLM judges, prompt management, an AI Gateway for managing costs and model access, human feedback collection, and agent serving.
2. **Machine Learning**: Experiment tracking, hyperparameter tuning, model evaluation, a model registry for version control and deployment management, unified model packaging, and flexible model serving. Supports classical machine learning and deep learning models.

MLflow supports all LLM providers, AI agent frameworks, and coding assistants, including LangChain, LangGraph, OpenAI, Anthropic, ADK, Pydantic AI, Claude Code, Codex, Cursor, DSPy, CrewAI, Gemini, DeepSeek, GLM, Kimi, and beyond. It works on any major cloud provider (AWS, Azure, GCP, Databricks) or on-premises infrastructure. Native SDKs are available for Python, TypeScript/JavaScript, Java, and R, and MLflow's REST API integrates seamlessly with any programming language.

MLflow is backed by the Linux Foundation and is 100% open source under the Apache 2.0 license.

## Website

- [MLflow Home](https://mlflow.org/): Overview of the MLflow AI engineering platform
- [LLMs & Agents](https://mlflow.org/genai): Ship AI agents and LLM apps to production with built-in observability, evaluation, prompt management, and monitoring
- [Observability](https://mlflow.org/genai/observability): End-to-end observability for agents and LLM applications with execution visualization and tracing
- [Evaluations](https://mlflow.org/genai/evaluations): Assess agent and LLM output quality with pre-built LLM judges and custom evaluation metrics
- [Prompt Registry](https://mlflow.org/genai/prompt-registry): Create, version, and manage prompt templates with comparison, evaluation, and automatic optimization
- [AI Gateway](https://mlflow.org/genai/ai-gateway): Single control plane for LLM provider access with unified authentication, rate limiting, and fallback routing
- [Human Feedback](https://mlflow.org/genai/human-feedback): Collect domain expert and end-user feedback to understand and improve AI application quality
- [Machine Learning](https://mlflow.org/classical-ml): Master the full ML lifecycle from experimentation to production with experiment tracking, model management, and deployment
- [Experiment Tracking](https://mlflow.org/classical-ml/experiment-tracking): Track, compare, and reproduce ML experiments with parameters, metrics, and artifacts
- [Model Evaluation](https://mlflow.org/classical-ml/model-evaluation): Automated evaluation tools for classification, regression, and other ML techniques
- [Model Registry](https://mlflow.org/classical-ml/model-registry): Version control, approval workflows, and deployment management for ML models
- [Model Packaging](https://mlflow.org/classical-ml/models): Package, share, and deploy models across frameworks with a unified model format
- [Model Serving](https://mlflow.org/classical-ml/serving): Deploy and serve ML models with flexible serving options for real-time and batch inference
- [Hyperparameter Tuning](https://mlflow.org/classical-ml/hyperparam-tuning): Optimize ML models using state-of-the-art hyperparameter optimization techniques

## Topical Pages

- [AI Observability](https://mlflow.org/ai-observability): What is AI observability and how MLflow captures traces, evaluations, and metrics across agent and LLM workflows
- [LLM Tracing](https://mlflow.org/llm-tracing): What is LLM tracing and how MLflow records every step of agent execution with inputs, outputs, latency, and costs
- [LLM Evaluation](https://mlflow.org/llm-evaluation): How to systematically assess the quality of LLM applications and autonomous agents using LLM judges
- [LLMOps](https://mlflow.org/llmops): What is LLMOps and how to build, deploy, monitor, and maintain LLM applications in production

## LLMs and Agents Documentation

- [Getting Started with MLflow for LLMs and Agents](https://mlflow.org/docs/latest/genai/getting-started/): Get started with MLflow for LLMs and agents
- [Tracing Quickstart](https://mlflow.org/docs/latest/genai/tracing/quickstart/): Get started with MLflow tracing for observability
- [Automatic Tracing](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/automatic): Automatically instrument LLM frameworks like LangChain, OpenAI, and Anthropic
- [Manual Tracing](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/manual-tracing): Add custom tracing to any Python application
- [OpenTelemetry Integration](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/opentelemetry): Connect MLflow tracing with OpenTelemetry
- [Distributed Tracing](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/distributed-tracing): Trace across distributed microservices
- [Production Monitoring](https://mlflow.org/docs/latest/genai/tracing/prod-tracing): Monitor production LLM and agent deployments with tracing
- [Token Usage and Cost Tracking](https://mlflow.org/docs/latest/genai/tracing/token-usage-cost/): Track token usage and costs across LLM providers
- [Search Traces](https://mlflow.org/docs/latest/genai/tracing/search-traces): Search and query traces for debugging and analysis
- [User Feedback Collection](https://mlflow.org/docs/latest/genai/tracing/collect-user-feedback/): Collect end-user feedback on AI outputs
- [Evaluation Quickstart](https://mlflow.org/docs/latest/genai/eval-monitor/quickstart): Get started evaluating LLM and agent quality
- [Evaluating Agents](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/agents): Evaluate autonomous AI agents
- [Multi-turn Evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/multi-turn): Evaluate multi-turn conversational agents
- [LLM Judge Scorers](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/): Use LLM-as-a-judge for automated evaluation
- [Predefined LLM Judges](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined): Built-in evaluation metrics including safety, correctness, and relevance
- [RAG Evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/): Evaluate retrieval-augmented generation with groundedness, relevance, and context sufficiency scorers
- [Custom Scorers](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/custom/): Build custom evaluation scorers
- [AI Issue Discovery](https://mlflow.org/docs/latest/genai/eval-monitor/ai-insights/ai-issue-discovery): Automatically discover quality issues in AI applications
- [Prompt Registry Docs](https://mlflow.org/docs/latest/genai/prompt-registry/): Create, version, and manage prompt templates
- [Prompt Engineering](https://mlflow.org/docs/latest/genai/prompt-registry/prompt-engineering): Best practices for prompt engineering with MLflow
- [Optimize Prompts](https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts): Automatically optimize prompts for better results
- [AI Gateway Quickstart](https://mlflow.org/docs/latest/genai/governance/ai-gateway/quickstart): Get started with the MLflow AI Gateway
- [AI Gateway Endpoints](https://mlflow.org/docs/latest/genai/governance/ai-gateway/endpoints/create-and-manage): Create and manage AI Gateway endpoints
- [Model Providers](https://mlflow.org/docs/latest/genai/governance/ai-gateway/endpoints/model-providers): Configure model providers (OpenAI, Anthropic, AWS Bedrock, etc.)
- [Version Tracking](https://mlflow.org/docs/latest/genai/version-tracking/): Track and compare application versions
- [Agent Server](https://mlflow.org/docs/latest/genai/serving/agent-server): Serve AI agents as REST APIs

### Agent and LLM Tracing Integrations

- [All Tracing Integrations](https://mlflow.org/docs/latest/genai/tracing/integrations/): Full list of supported LLM and agent framework integrations
- [OpenAI Integration](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/openai): Automatic tracing for OpenAI API calls
- [LangChain Integration](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langchain): Automatic tracing for LangChain
- [LangGraph Integration](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langgraph): Automatic tracing for LangGraph agents
- [Anthropic Integration](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/anthropic): Automatic tracing for Anthropic Claude
- [Gemini Integration](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/gemini): Automatic tracing for Google Gemini
- [OpenAI Agents Integration](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/openai-agent): Automatic tracing for OpenAI Agents SDK
- [Google ADK Integration](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/google-adk): Automatic tracing for Google Agent Development Kit
- [DSPy Integration](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/dspy): Automatic tracing for DSPy programs
- [AWS Bedrock Integration](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/bedrock): Automatic tracing for AWS Bedrock

## Machine Learning (ML) Documentation

- [ML Getting Started](https://mlflow.org/docs/latest/ml/getting-started/): Get started with MLflow for machine learning
- [ML Quickstart](https://mlflow.org/docs/latest/ml/getting-started/intro-quickstart/): Quick introduction to MLflow experiment tracking
- [Tracking Overview](https://mlflow.org/docs/latest/ml/tracking/): Track experiments with parameters, metrics, and artifacts
- [Autologging](https://mlflow.org/docs/latest/ml/tracking/autolog/): Automatic experiment logging for popular ML frameworks
- [Model Overview](https://mlflow.org/docs/latest/ml/model/): MLflow model format and packaging
- [Model Signatures](https://mlflow.org/docs/latest/ml/model/signatures/): Define model input/output schemas
- [Model Registry](https://mlflow.org/docs/latest/ml/model-registry/): Manage model versions and lifecycle stages
- [Model Evaluation](https://mlflow.org/docs/latest/ml/model-evaluation/): Evaluate ML models with automated metrics
- [Deployment Overview](https://mlflow.org/docs/latest/ml/deployment/): Deploy models to production
- [Deploy Locally](https://mlflow.org/docs/latest/ml/deployment/deploy-model-locally/): Serve models as local REST APIs
- [Deploy to Kubernetes](https://mlflow.org/docs/latest/ml/deployment/deploy-model-to-kubernetes/): Deploy models on Kubernetes clusters

### ML Framework Integrations

- [Scikit-learn](https://mlflow.org/docs/latest/ml/traditional-ml/sklearn/): MLflow integration with scikit-learn
- [XGBoost](https://mlflow.org/docs/latest/ml/traditional-ml/xgboost/): MLflow integration with XGBoost
- [Spark ML](https://mlflow.org/docs/latest/ml/traditional-ml/sparkml/): MLflow integration with Apache Spark ML
- [PyTorch](https://mlflow.org/docs/latest/ml/deep-learning/pytorch/): MLflow integration with PyTorch
- [TensorFlow](https://mlflow.org/docs/latest/ml/deep-learning/tensorflow/): MLflow integration with TensorFlow
- [Keras](https://mlflow.org/docs/latest/ml/deep-learning/keras/): MLflow integration with Keras
- [Transformers](https://mlflow.org/docs/latest/ml/deep-learning/transformers/): MLflow integration with HuggingFace Transformers
- [Sentence Transformers](https://mlflow.org/docs/latest/ml/deep-learning/sentence-transformers/): MLflow integration with Sentence Transformers

## Self-Hosting

- [Self-Hosting Overview](https://mlflow.org/docs/latest/self-hosting/): Host your own MLflow instance
- [Architecture](https://mlflow.org/docs/latest/self-hosting/architecture/): MLflow server architecture (tracking server, backend store, artifact store)
- [Basic HTTP Auth](https://mlflow.org/docs/latest/self-hosting/security/basic-http-auth): Secure MLflow with HTTP authentication
- [SSO Configuration](https://mlflow.org/docs/latest/self-hosting/security/sso): Configure single sign-on for MLflow
- [Workspaces](https://mlflow.org/docs/latest/self-hosting/workspaces/): Multi-tenant workspaces for team collaboration

## API Reference

- [Python API](https://mlflow.org/docs/latest/api_reference/python_api/): Complete Python API reference
- [REST API](https://mlflow.org/docs/latest/api_reference/rest-api.html): REST API reference for all endpoints
- [Java API](https://mlflow.org/docs/latest/api_reference/java_api/): Java SDK reference
- [R API](https://mlflow.org/docs/latest/api_reference/r_api.html): R SDK reference
- [CLI Reference](https://mlflow.org/docs/latest/api_reference/cli.html): Command-line interface reference

## Community

- [GitHub](https://github.com/mlflow/mlflow): Source code, issues, and contributions
- [Slack](https://go.mlflow.org/slack): Join the MLflow community Slack
- [LinkedIn](https://www.linkedin.com/company/mlflow-org): Follow MLflow on LinkedIn
- [X (Twitter)](https://x.com/mlflow): Follow MLflow on X
- [YouTube](https://www.youtube.com/@mlflowoss): Tutorials and conference talks
- [Blog](https://mlflow.org/blog): Latest news, tutorials, and release announcements
- [Ambassador Program](https://mlflow.org/ambassadors): Join the MLflow Ambassador community
