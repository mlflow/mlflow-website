---
title: Announcing MLflow 3
tags: [mlflow, genai, tracing, evaluation, mlops]
slug: mlflow-3-launch
authors: [mlflow-maintainers]
thumbnail: /img/blog/mlflow-3-trace-ui.png
---

The open source MLflow community has reached a major milestone. Today, we're releasing **MLflow 3**, which brings production-ready generative AI capabilities to the platform that millions of developers trust for ML operations.

This isn't just another feature update. MLflow 3 fundamentally expands what's possible with open source ML tooling, addressing the observability and quality challenges that have made GenAI deployment feel like a leap of faith.

## Why GenAI Breaks Traditional MLOps

Traditional machine learning follows predictable patterns. You have datasets with ground truth labels, metrics that clearly indicate success or failure, and deployment pipelines that scale horizontally. GenAI is disruptive not only for its powerful features, but also for introducing foundational changes to how quality and stability are measured and ensured.

Consider a simple question: "How do you know if your RAG system is working correctly?" In traditional ML, you'd check accuracy against a test set. In GenAI, you're dealing with:

- **Complex execution flows** involving multiple LLM calls, retrievals, and tool interactions
- **Subjective output quality** where "correct" can mean dozens of different valid responses
- **Latency and cost concerns** that can make or break user experience
- **Debugging nightmares** when something goes wrong deep in a multi-step reasoning chain

The current solution? Most teams cobble together monitoring tools, evaluation scripts, and deployment pipelines from different vendors. The result is fragmented workflows where critical information gets lost between systems.

## A Different Approach to GenAI Infrastructure

MLflow 3 takes a different approach. Instead of building yet another specialized GenAI platform, we've extended MLflow's battle-tested foundation to handle the unique requirements of generative AI while maintaining compatibility with traditional ML workflows.

This means you can instrument a transformer training pipeline and a multi-agent RAG system with the same tools, deploy them through the same registry, and monitor them with unified observability infrastructure.

### Deep Observability with MLflow Tracing

The centerpiece of MLflow 3 is comprehensive tracing that works across the entire GenAI ecosystem. Unlike logging frameworks that capture basic inputs and outputs, MLflow Tracing provides hierarchical visibility into complex execution flows.

```python
import mlflow
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma

# Instrument your entire application with one line
mlflow.langchain.autolog()

@mlflow.trace(name="customer_support")
def answer_question(question, customer_tier="standard"):
    vectorstore = Chroma.from_documents(documents, embeddings)
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # Tracing captures the full execution tree automatically
    result = qa_chain({"query": question})

    # Add business context to traces
    mlflow.update_current_trace(
        tags={
            "customer_tier": customer_tier,
            "question_category": classify_question(question)
        }
    )

    return result["result"]
```

What makes this powerful is the automatic instrumentation. When `answer_question()` executes, MLflow tracing captures:

- The initial LLM call for query processing
- Vector database retrieval with embedding calculations
- Document ranking and selection logic
- Final answer generation with token usage
- All intermediate inputs, outputs, and timing information
- Detailed and comprehensive error messages, including stack traces, if any error occurs

This creates a complete execution timeline that you can drill into when issues arise. No more guessing why your RAG system returned irrelevant documents or why response times spiked.

### Systematic Quality Evaluation

Evaluating GenAI quality has traditionally meant manual review processes that don't scale. MLflow 3 includes a comprehensive evaluation framework that can assess quality dimensions systematically.

The evaluation harness supports both direct evaluation (where MLflow calls your application to generate fresh traces) and answer sheet evaluation (for pre-computed outputs). You can also build custom scorers for domain-specific requirements using the `@scorer` decorator for full customization of your evaluation needs.

### Application Lifecycle Management

GenAI applications are more than just models—they're complex systems involving prompts, retrieval logic, tool integrations, and orchestration code. MLflow 3 treats these applications as first-class artifacts that can be versioned, registered, and deployed atomically.

```python
import mlflow.pyfunc

class CustomerServiceBot(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load all components of your GenAI application
        self.llm = load_model_from_artifacts(context.artifacts["llm_config"])
        self.vector_store = initialize_vector_store(context.artifacts["knowledge_base"])
        self.prompt_template = load_prompt_template(context.artifacts["prompt_template"])

    def predict(self, context, model_input):
        # Your application logic
        query = model_input["query"][0]
        relevant_docs = self.vector_store.similarity_search(query, k=3)

        formatted_prompt = self.prompt_template.format(
            query=query,
            context="\n".join([doc.page_content for doc in relevant_docs])
        )

        response = self.llm.predict(formatted_prompt)
        return {"response": response, "sources": [doc.metadata for doc in relevant_docs]}

# Package and version the complete application
with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        artifact_path="customer_service_bot",
        python_model=CustomerServiceBot(),
        artifacts={
            "llm_config": "configs/llm_config.yaml",
            "knowledge_base": "data/knowledge_embeddings.pkl",
            "prompt_template": "prompts/customer_service_v2.txt"
        },
        pip_requirements=["openai", "langchain", "chromadb"],
        signature=mlflow.models.infer_signature(example_input, example_output)
    )

    # Register in model registry for deployment
    registered_model = mlflow.register_model(
        model_uri=model_info.model_uri,
        name="customer_service_bot",
        tags={"version": "v2.1", "eval_score": "0.87"}
    )
```

This approach ensures that when you deploy version 2.1 of your customer service bot, you're deploying exactly the same combination of model weights, prompts, retrieval logic, and dependencies that you tested. No more "it worked in dev" deployment surprises.

## Enhanced Traditional ML & Deep Learning

While GenAI capabilities are the headline feature, MLflow 3 includes significant improvements for traditional machine learning and deep learning workflows:

**Enhanced Model Registry**: The same versioning and deployment infrastructure that handles GenAI applications now provides better lineage tracking for all model types. Deep learning practitioners benefit from improved checkpoint management and experiment organization.

**Unified Evaluation Framework**: The evaluation system extends beyond GenAI to support custom metrics for computer vision, NLP, and tabular data models. Teams can now standardize evaluation processes across different model types.

**Improved Deployment Workflows**: Quality gates and automated testing capabilities work for any MLflow model, whether it's a scikit-learn classifier or a multi-modal foundation model.

## Getting Started Today

MLflow 3 is available now and designed to work alongside your existing ML infrastructure. Here's how to get started:

### Installation and Setup

```bash
pip install -U mlflow
```

### First Steps

```python
import mlflow

# Create your first GenAI experiment
mlflow.set_experiment("my_genai_prototype")

# Enable automatic tracing for openai (choose any tracing integration to enable auto-tracing for any of the 20+ supported tracing integrations)
mlflow.openai.autolog()

# Your existing GenAI code will now generate traces automatically
# No additional instrumentation required for supported libraries
```

## The Road Ahead

MLflow 3 represents a significant step forward in making GenAI development more systematic and reliable. But this is just the beginning. The open source community continues to drive innovation with new integrations, evaluation metrics, and deployment patterns.

**How to Get Involved:**

- **Contribute Code**: We welcome contributions of all sizes, from bug fixes to new integrations
- **Share Use Cases**: Help others learn by documenting your MLflow implementations
- **Report Issues**: Help us improve by reporting bugs and requesting features
- **Join Discussions**: Participate in technical discussions and roadmap planning

The future of AI development is unified, observable, and reliable. MLflow 3 brings that future to the open source community today.

---

**Ready to try MLflow 3?** Explore the [full documentation](https://mlflow.org/docs/latest/) to see what's possible.

_MLflow is an open source project under the Apache 2.0 license, built with contributions from the global ML community._
