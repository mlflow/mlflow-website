# MLflow Tracing for LLM Observability

MLflow Tracing is a fully **OpenTelemetry-compatible** LLM observability solution for your applications. It captures the inputs, outputs, and metadata associated with each intermediate step of a request, enabling you to easily pinpoint the source of bugs and unexpected behaviors.

[](/mlflow-website/docs/latest/images/llms/tracing/tracing-top.mp4)

## Use Cases Throughout the ML Lifecycle[​](#use-cases-throughout-the-ml-lifecycle "Direct link to Use Cases Throughout the ML Lifecycle")

MLflow Tracing empowers you throughout the end-to-end lifecycle of a machine learning project. Here's how it helps you at each step of the workflow, click on the tabs below to learn more:

* Build & Debug
* Human Feedback
* Evaluation
* Production Monitoring
* Dataset Collection

#### Debug Issues in Your IDE or Notebook[​](#debug-issues-in-your-ide-or-notebook "Direct link to Debug Issues in Your IDE or Notebook")

Traces provide deep insights into what happens beneath the abstractions of GenAI libraries, helping you precisely identify where issues occur.

You can navigate traces seamlessly within your preferred IDE, notebook, or the MLflow UI, eliminating the hassle of switching between multiple tabs or searching through an overwhelming list of traces.

[Learn more →](/mlflow-website/docs/latest/genai/tracing/observe-with-traces/ui.md)

![Trace Debugging](/mlflow-website/docs/latest/assets/images/genai-trace-debug-405f9c8b61d5f89fb1d3891242fcd265.png)

#### Track Annotation and Human Feedback[​](#track-annotation-and-human-feedback "Direct link to Track Annotation and Human Feedback")

Human feedback is essential for building high-quality GenAI applications that meet user expectations. MLflow supports collecting, managing, and utilizing feedback from end-users and domain experts.

Feedback are attached to traces and recorded with metadata, including user, timestamp, revisions, etc.

[Learn more →](/mlflow-website/docs/latest/genai/assessments/feedback.md)

![Trace Feedback](/mlflow-website/docs/latest/assets/images/genai-human-feedback-9a8ea2ba10a5f7c7bb192aea22345b19.png)

#### Evaluate and Enhance Quality[​](#evaluate-and-enhance-quality "Direct link to Evaluate and Enhance Quality")

Systematically assessing and improving the quality of GenAI applications is a challenge. Combined with [MLflow GenAI Evaluation](/mlflow-website/docs/latest/genai/eval-monitor.md), MLflow offers a seamless experience for evaluating your applications.

Tracing helps by allowing you to track quality assessment and inspect the evaluation results with visibility into the internals of the system.

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor.md)

![Trace Evaluation](/mlflow-website/docs/latest/assets/images/genai-trace-evaluation-5b5e6ba86f0f0f06ee27db356e4e59e4.png)

#### Monitor Applications in Production[​](#monitor-applications-in-production "Direct link to Monitor Applications in Production")

Understanding and optimizing GenAI application performance is crucial for efficient operations. MLflow Tracing captures key metrics like latency and token usage at each step, as well as various quality metrics, helping you identify bottlenecks, monitor efficiency, and find optimization opportunities.

[Learn more →](/mlflow-website/docs/latest/genai/tracing/prod-tracing.md)

![Monitoring](/mlflow-website/docs/latest/assets/images/genai-monitoring-8ebda32e5cc07cb9cc97cb0297e583c3.png)

#### Create a High-Quality Dataset from Real World Traffic[​](#create-a-high-quality-dataset-from-real-world-traffic "Direct link to Create a High-Quality Dataset from Real World Traffic")

Evaluating the performance of your GenAI application is crucial, but creating a reliable evaluation dataset is challenging.

Traces from production systems capture perfect data for building high-quality datasets with precise details for internal components like retrievers and tools.

[Learn more →](/mlflow-website/docs/latest/genai/datasets.md)

![Trace Dataset](/mlflow-website/docs/latest/assets/images/genai-trace-dataset-0db517dfd5b8e13ae6732b0a1b0b098f.png)

## What Makes MLflow Tracing Unique?[​](#what-makes-mlflow-tracing-unique "Direct link to What Makes MLflow Tracing Unique?")

#### Open Source

MLflow is open source and 100% FREE. You don't need to pay additional SaaS costs to add observability to your GenAI stack. Your trace data is hosted on your own infrastructure.

#### OpenTelemetry

MLflow Tracing is fully compatible with OpenTelemetry, making it free from vendor lock-in and easy to integrate with your existing observability stack.

#### Framework Agnostic

MLflow Tracing integrates with 20+ GenAI libraries, including OpenAI, LangChain, LlamaIndex, DSPy, Pydantic AI, allowing you to switch between frameworks with ease.

#### End-to-End Platform

MLflow Tracing empowers you throughout the end-to-end machine learning lifecycle, combined with its version tracking and evaluation capabilities.

#### Strong Community

MLflow boasts a vibrant Open Source community as a part of the Linux Foundation, with 20K+ GitHub Stars and 20MM+ monthly downloads.

## Getting Started[​](#getting-started "Direct link to Getting Started")

[![Quickstart (Python)](/mlflow-website/docs/latest/images/logos/python-logo.png)](/mlflow-website/docs/latest/genai/tracing/quickstart.md)

### [Quickstart (Python)](/mlflow-website/docs/latest/genai/tracing/quickstart.md)

[Get started with MLflow Tracing in Python](/mlflow-website/docs/latest/genai/tracing/quickstart.md)

[Start building →](/mlflow-website/docs/latest/genai/tracing/quickstart.md)

[![Quickstart (JS/TS)](/mlflow-website/docs/latest/images/logos/javascript-typescript-logo.png)](/mlflow-website/docs/latest/genai/tracing/quickstart.md)

### [Quickstart (JS/TS)](/mlflow-website/docs/latest/genai/tracing/quickstart.md)

[Get started with MLflow Tracing in JavaScript or TypeScript](/mlflow-website/docs/latest/genai/tracing/quickstart.md)

[Start building →](/mlflow-website/docs/latest/genai/tracing/quickstart.md)

[![Quickstart (Otel)](/mlflow-website/docs/latest/images/logos/opentelemetry-logo.svg)](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/opentelemetry.md)

### [Quickstart (Otel)](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/opentelemetry.md)

[Export traces from an app instrumented with OpenTelemetry to MLflow.](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/opentelemetry.md)

[Start building →](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/opentelemetry.md)

## One-line Auto Tracing Integrations[​](#one-line-auto-tracing-integrations "Direct link to One-line Auto Tracing Integrations")

MLflow Tracing is integrated with various GenAI libraries, such as OpenAI, LangChain, DSPy, Vercel AI, and provides one-line automatic tracing experience for each library (and combinations of them!):

python

```python
import mlflow

mlflow.openai.autolog()  # or replace 'openai' with other library names, e.g., "anthropic"

```

View the full list of supported libraries and detailed setup instructions on the [Integrations](/mlflow-website/docs/latest/genai/tracing/integrations.md) page.

## Flexible and Customizable[​](#flexible-and-customizable "Direct link to Flexible and Customizable")

In addition to the one-line auto tracing experience, MLflow offers Python SDK for manually instrumenting your code and manipulating traces:

* [Trace a function with `@mlflow.trace` decorator](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/manual-tracing.md#decorator)
* [Trace any block of code](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/manual-tracing.md#code-block)
* [Combine multiple auto-tracing integrations](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/automatic.md#multi-framework-example)
* [Instrument multi-threaded applications](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/manual-tracing.md#multi-threading)
* [Native async support](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/manual-tracing.md#async-support)
* [Group and filter traces using sessions](/mlflow-website/docs/latest/genai/tracing/track-users-sessions.md)
* [Redact PII data from traces](/mlflow-website/docs/latest/genai/tracing/observe-with-traces/masking.md)
* [Disable tracing globally](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/automatic.md#disabling-tracing)
* [Configure sampling ratio to control trace throughput](/mlflow-website/docs/latest/genai/tracing/prod-tracing.md#sampling-traces)
* [Propagate trace context across services](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/distributed-tracing.md)

## Production Readiness[​](#production-readiness "Direct link to Production Readiness")

MLflow Tracing is production ready and provides comprehensive monitoring capabilities for your GenAI applications in production environments. By enabling [async logging](/mlflow-website/docs/latest/genai/tracing/prod-tracing.md#asynchronous-trace-logging), trace logging is done in the background and does not impact the performance of your application.

For production deployments, it is recommended to use the [Production Tracing SDK](/mlflow-website/docs/latest/genai/tracing/lightweight-sdk.md) (`mlflow-tracing`) that is optimized for reducing the total installation size and minimizing dependencies while maintaining full tracing capabilities. Compared to the full `mlflow` package, the `mlflow-tracing` package requires 95% smaller footprint.

Read [Production Monitoring](/mlflow-website/docs/latest/genai/tracing/prod-tracing.md) for complete guidance on using MLflow Tracing for monitoring models in production and various backend configuration options.
