# MLflow 3.0

The open source MLflow community has reached a major milestone. Today, we're releasing MLflow 3.0, which brings production-ready generative AI capabilities to the platform that millions of developers trust for ML operations.

This isn't just another feature update. MLflow 3.0 fundamentally expands what's possible with open source ML tooling, addressing the observability and quality challenges that have made GenAI deployment feel like a leap of faith.

## Major Updates

### Model-centric Architecture

MLflow 3.0 introduces a refined architecture with the new LoggedModel entity as a first-class citizen, moving beyond the traditional run-centric approach. This enables better organization and comparison of GenAI agents, deep learning checkpoints, and model variants across experiments.

### Strong Lineage Support

Enhanced model tracking provides comprehensive lineage between models, runs, traces, and evaluation metrics. The new model-centric design allows you to group traces and metrics from interactive queries and automated evaluation jobs, enabling rich comparisons across model versions.

### Feedback Tracking

Built-in assessment and feedback tracking capabilities allow you to capture both automated and human evaluation feedback directly tied to model executions and traces. This provides a comprehensive view of model performance across different evaluation dimensions.

### New GenAI Evaluation Suite

Expanded evaluation capabilities with enhanced GenAI metrics including answer correctness, answer relevance, and faithfulness. The evaluation system now supports callable metrics and improved LLM-as-a-Judge functionality across multiple providers.

### Prompt Optimization

The MLflow Prompt Registry now includes prompt optimization capabilities, allowing you to automatically improve prompts using evaluation feedback and labeled datasets. This includes versioning, tracking, and systematic prompt engineering workflows.

## Breaking Changes

MLflow 3.0 includes several breaking changes as part of improving framework consistency and performance. Key changes include removal of MLflow Recipes, fastai and mleap flavors, and various deprecated API parameters.

For the complete list of breaking changes, visit the [MLflow 3.0 Breaking Changes documentation](https://mlflow.org/docs/latest/ml/mlflow-3/breaking-changes).

## Other Features

**Streaming API for ResponsesAgent**: New streaming response capabilities through the ResponsesAgent class with predict_stream method, enabling real-time streaming for GenAI applications.

**Auto-tracing support for Pydantic AI and `smolagents`**: Enhanced auto-tracing integrations for emerging GenAI frameworks, providing seamless observability out of the box.

**Add `search_prompts` API for prompt registry**: New API functionality for searching and discovering prompts in the registry, making prompt management more efficient.

**Support token tracking for OpenAI/LangChain auto-tracing**: Enhanced tracing now captures detailed token usage and cost information for better observability and cost management.

**Record environment metadata in tracing**: Traces now capture comprehensive environment metadata including dependencies, system information, and execution context for better reproducibility.

**UI support for video artifacts**: The MLflow UI now supports viewing video files directly in the artifact viewer, expanding beyond traditional ML artifacts to support multi-modal AI applications.

**and many more**: Numerous other enhancements across tracking, model registry, and UI components to improve usability, performance, and developer experience.

## Getting Started

```bash
pip install mlflow>=3.0.0
```

Explore the new MLflow 3.0 documentation and try out the enhanced GenAI capabilities with our updated quickstart guides. The model-centric architecture and improved tracing make it easier than ever to build, evaluate, and deploy production-ready AI applications.

## Notes

MLflow 3.0 represents a significant evolution toward production-ready GenAI workflows while maintaining backward compatibility where possible. The shift to a model-centric architecture with LoggedModel entities addresses the unique challenges of GenAI development, where multiple evaluation rounds and model variants are common. The enhanced evaluation suite and tracing capabilities provide the observability needed for confident GenAI deployment in production environments.
