# Lightweight Tracing SDK Optimized for Production Usage

MLflow offers a lightweight tracing SDK package called `mlflow-tracing` that includes only the essential functionality for tracing and monitoring of your GenAI applications. This package is designed for production environments where minimizing dependencies and deployment size is critical.

## Why Use the Lightweight SDK?[​](#why-use-the-lightweight-sdk "Direct link to Why Use the Lightweight SDK?")

🏎️ Faster Deployment

***

The package size and dependencies are significantly smaller than the full MLflow package, allowing for faster deployment times in dynamic environments such as Docker containers, serverless functions, and cloud-based applications.

🔧 Simplified Dependency Management

***

A smaller set of dependencies means less work keeping up with dependency updates, security patches, and potential breaking changes from upstream libraries. It also reduces the chances of dependency collisions and incompatibilities.

📦 Enhanced Portability

***

With fewer dependencies, MLflow Tracing can be seamlessly deployed across different environments and platforms, without worrying about compatibility issues.

🔒 Reduced Security Risk

***

Each dependency potentially introduces security vulnerabilities. By reducing the number of dependencies, MLflow Tracing minimizes the attack surface and reduces the risk of security breaches.

## Installation[​](#installation "Direct link to Installation")

Install the lightweight SDK using pip:

bash

```bash
pip install mlflow-tracing

```

warning

Do not install the full `mlflow` package together with the lightweight `mlflow-tracing` SDK, as this may cause version conflicts and namespace resolution issues.

## Quickstart[​](#quickstart "Direct link to Quickstart")

Here's a simple example using the lightweight SDK with OpenAI for logging traces to an experiment on a remote MLflow server:

python

```python
import mlflow
import openai

# Set the tracking URI to your MLflow server
mlflow.set_tracking_uri("http://your-mlflow-server:5000")
mlflow.set_experiment("genai-production-monitoring")

# Enable auto-tracing for OpenAI
mlflow.openai.autolog()

# Use OpenAI as usual - traces will be automatically logged
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini", messages=[{"role": "user", "content": "What is MLflow?"}]
)

print(response.choices[0].message.content)

```

## Choose Your Backend[​](#choose-your-backend "Direct link to Choose Your Backend")

The lightweight SDK works with various observability platforms. Choose your preferred option and follow the instructions to set up your tracing backend.

* Self-Hosted MLflow
* Databricks
* OpenTelemetry
* Amazon SageMaker
* Nebius

MLflow is a **fully open-source project**, allowing you to self-host your own MLflow server in your own infrastructure. This is a great option if you want to have full control over your data and are restricted in using cloud-based services.

In self-hosting mode, you will be responsible for running the tracking server instance and scaling it to your needs. We **strongly recommend** using a SQL-based tracking server on top of a performant database to minimize operational overhead and ensure high availability.

**Setup Steps:**

1. Install MLflow server: `pip install mlflow[extras]`
2. Configure backend store (PostgreSQL/MySQL recommended)
3. Configure artifact store (S3, Azure Blob, GCS, etc.)
4. Start server: `mlflow server --backend-store-uri postgresql://... --default-artifact-root s3://...`

Refer to the [tracking server setup guide](/mlflow-website/docs/latest/ml/tracking.md#tracking-setup) for detailed guidance.

![OSS Tracing](/mlflow-website/docs/latest/assets/images/tracing-top-dcca046565ab33be6afe0447dd328c22.gif)

[Databricks Lakehouse Monitoring for GenAI](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/monitoring) is a go-to solution for monitoring your GenAI application with MLflow Tracing. It provides access to a robust, fully functional monitoring dashboard for operational excellence and quality analysis.

Lakehouse Monitoring for GenAI can be used regardless of whether your application is hosted on Databricks or not.

[Sign up for free](https://signup.databricks.com/?destination_url=/ml/experiments-signup?source=OSS_DOCS\&dbx_source=TRY_MLFLOW\&signup_experience_step=EXPRESS\&provider=MLFLOW\&utm_source=OSS_DOCS) and [get started in a minute](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/monitoring) to run your GenAI application with complete observability.

![Monitoring](https://assets.docs.databricks.com/_static/images/generative-ai/monitoring/monitoring-hero.gif)

MLflow Tracing is **built on the OpenTelemetry tracing spec**, an industry-standard framework for observability, making it a vendor-neutral solution for monitoring your GenAI applications.

If you are using OpenTelemetry as part of your observability tech stack, you can use MLflow Tracing without any additional service onboarding. Simply configure the OTLP endpoint and MLflow will export traces to your existing observability platform.

**Setup Example:**

bash

```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="http://your-collector:4317/v1/traces"
export OTEL_SERVICE_NAME="genai-app"

```

Refer to the [OpenTelemetry Integration](/mlflow-website/docs/latest/genai/tracing/opentelemetry/export.md) documentation for detailed setup instructions.

![OpenTelemetry Backend Examples](/mlflow-website/docs/latest/assets/images/otel-backend-examples-336441b516d95409d8cc820a128ac376.png)

[MLflow on Amazon SageMaker](https://aws.amazon.com/sagemaker-ai/experiments/) is a fully managed service offered as part of the SageMaker platform by AWS, including tracing and other MLflow features such as model registry.

MLflow Tracing offers a one-line solution for [tracing Amazon Bedrock](/mlflow-website/docs/latest/genai/tracing/integrations/listing/bedrock.md), making it the best suitable solution for monitoring GenAI application powered by AWS and Amazon Bedrock.

![Managed MLflow on SageMaker](https://d1.awsstatic.com/deploy-mlflow-models.3eb857c5790a44b461845a630e3a231229838443.png)

[Nebius](https://nebius.com/services/managed-mlflow), a cutting-edge cloud platform for GenAI explorers, offers a fully managed MLflow server. Combining the powerful GPU infrastructure of Nebius for training and hosting LLMs/foundation models with the observability capabilities of MLflow Tracing, Nebius serves as a comprehensive platform for AI/ML developers.

**Key Features:**

* Fully managed MLflow service
* High-performance GPU infrastructure
* Integrated LLM training and serving
* Enterprise-grade observability

Refer to the [Nebius documentation](https://nebius.com/services/managed-mlflow) for more details about the managed MLflow service.

![Nebius Logo](data:image/png;base64,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)

## Supported Features[​](#supported-features "Direct link to Supported Features")

The lightweight SDK includes all essential tracing functionalities for monitoring your GenAI applications. Click the cards below to learn more about each supported feature.

[⚡️ Automatic Tracing for 15+ AI Libraries](/mlflow-website/docs/latest/genai/tracing/integrations.md)

***

[MLflow Tracing SDK supports one-line integration with all of the most popular LLM/GenAI libraries including OpenAI, Anthropic, LangChain, LlamaIndex, Hugging Face, DSPy, and any LLM provider that conforms to OpenAI API format. This automatic tracing capability allows you to monitor your GenAI application with minimal effort and easily switch between different libraries.](/mlflow-website/docs/latest/genai/tracing/integrations.md)

[⚙️ Manual Instrumentation](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/manual-tracing.md)

***

[MLflow Tracing SDK provides a simple and intuitive API for manually instrumenting your GenAI application. Manual instrumentation and automatic tracing can be used together, allowing you to trace advanced applications containing custom code and have fine-grained control over the tracing behavior.](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/manual-tracing.md)

[🏷️ Tagging and Filtering Traces](/mlflow-website/docs/latest/genai/tracing/track-users-sessions.md)

***

[By annotating traces with custom tags, you can add more context to your traces to group them and simplify the process of searching for them later. This is useful when you want to trace an application that runs across multiple request sessions or track specific user interactions.](/mlflow-website/docs/latest/genai/tracing/track-users-sessions.md)

[🔍 Advanced Search and Querying](/mlflow-website/docs/latest/genai/tracing/search-traces.md)

***

[Search and filter traces using powerful SQL-like syntax based on execution time, status, tags, metadata, and other attributes. Perfect for debugging issues, analyzing performance patterns, and monitoring production applications.](/mlflow-website/docs/latest/genai/tracing/search-traces.md)

[📊 Production Monitoring](/mlflow-website/docs/latest/genai/tracing/prod-tracing.md)

***

[Configure asynchronous logging, handle high-volume tracing, and integrate with enterprise observability platforms. Includes comprehensive production deployment patterns and best practices for scaling your tracing infrastructure.](/mlflow-website/docs/latest/genai/tracing/prod-tracing.md)

## Production Configuration Example[​](#production-configuration-example "Direct link to Production Configuration Example")

Here's a complete example of setting up the lightweight SDK for production use:

python

```python
import mlflow
import os
from your_app import process_user_request

# Configure MLflow for production
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "production-genai-app"))

# Enable automatic tracing for your LLM library
mlflow.openai.autolog()  # or mlflow.langchain.autolog(), etc.


@mlflow.trace
def handle_user_request(user_id: str, session_id: str, message: str):
    """Production endpoint with comprehensive tracing."""

    # Add production context to trace
    mlflow.update_current_trace(
        tags={
            "user_id": user_id,
            "session_id": session_id,
            "environment": "production",
            "service_version": os.getenv("SERVICE_VERSION", "1.0.0"),
        }
    )

    try:
        # Your application logic here
        response = process_user_request(message)

        # Log success metrics
        mlflow.update_current_trace(
            tags={"response_length": len(response), "processing_successful": True}
        )

        return response

    except Exception as e:
        # Log error information
        mlflow.update_current_trace(
            tags={
                "error": True,
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        )
        raise

```

## Features Not Included[​](#features-not-included "Direct link to Features Not Included")

The following MLflow features are not available in the lightweight package:

* **MLflow Tracking Server and UI** - Use the full MLflow package to run the server
* **Run Management APIs** - `mlflow.start_run()`, `mlflow.log_metric()`, etc.
* **Model Logging and Evaluation** - Model serialization and evaluation frameworks
* **Model Registry** - Model versioning and lifecycle management
* **MLflow Projects** - Reproducible ML project format
* **MLflow Recipes** - Predefined ML workflows
* **Other MLflow Components** - Features unrelated to tracing

For these features, use the full MLflow package: `pip install mlflow`

## Migration from Full MLflow[​](#migration-from-full-mlflow "Direct link to Migration from Full MLflow")

If you're currently using the full MLflow package and want to switch to the lightweight SDK for production:

### 1. Update Dependencies[​](#1-update-dependencies "Direct link to 1. Update Dependencies")

bash

```bash
# Remove full MLflow
pip uninstall mlflow

# Install lightweight SDK
pip install mlflow-tracing

```

### 2. Update Import Statements[​](#2-update-import-statements "Direct link to 2. Update Import Statements")

Most tracing functionality remains the same:

python

```python
# These imports work the same way
import mlflow
import mlflow.openai
from mlflow.tracing import trace

# These features are NOT available in mlflow-tracing:
# import mlflow.sklearn  # ❌ Model logging
# mlflow.start_run()     # ❌ Run management
# mlflow.log_metric()    # ❌ Metric logging

```

### 3. Update Configuration[​](#3-update-configuration "Direct link to 3. Update Configuration")

Focus on tracing-specific configuration:

python

```python
# Configure tracking URI (same as before)
mlflow.set_tracking_uri("http://your-server:5000")
mlflow.set_experiment("your-experiment")


# Tracing works the same way
@mlflow.trace
def your_function():
    # Your code here
    pass

```

## Package Size Comparison[​](#package-size-comparison "Direct link to Package Size Comparison")

| Package          | Size     | Dependencies | Use Case                                        |
| ---------------- | -------- | ------------ | ----------------------------------------------- |
| `mlflow`         | \~1000MB | 20+ packages | Development, experimentation, full ML lifecycle |
| `mlflow-tracing` | \~5MB    | 5-8 packages | Production tracing, monitoring, observability   |

The lightweight SDK is **95% smaller** than the full MLflow package, making it ideal for:

* Container deployments
* Serverless functions
* Edge computing
* Production microservices
* CI/CD pipelines

## Summary[​](#summary "Direct link to Summary")

The MLflow Tracing SDK provides a production-optimized solution for monitoring GenAI applications with:

* **Minimal footprint** for fast deployments
* **Full tracing capabilities** for comprehensive monitoring
* **Flexible backend options** from self-hosted to enterprise platforms
* **Easy migration path** from full MLflow package
* **Production-ready features** including async logging and error handling

Whether you're running a small prototype or a large-scale production system, the lightweight SDK provides the observability you need without the overhead you don't.
