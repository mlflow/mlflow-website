# Instrument Your App with MLflow Tracing

tip

New to MLflow Tracing? Checkout the [Quick Start Guide](/mlflow-website/docs/latest/genai/tracing/quickstart/python-openai.md) to get started.

## Three Steps to Trace Your App/Agents[​](#three-steps-to-trace-your-appagents "Direct link to Three Steps to Trace Your App/Agents")

### 1. Installation[​](#1-installation "Direct link to 1. Installation")

* Python
* JS/TS

Add [`mlflow`](https://pypistats.org/packages/mlflow) to your Python environment.

bash

```
pip install mlflow
```

Install the [mlflow-tracing](https://www.npmjs.com/package/mlflow-tracing) package and other auto-tracing integrations (e.g., [mlflow-openai](https://www.npmjs.com/package/mlflow-tracing-openai)).

bash

```
npm install mlflow-tracing
```

### 2. Instrumenting Your Application Logic[​](#2-instrumenting-your-application-logic "Direct link to 2. Instrumenting Your Application Logic")

MLflow offers different ways to instrument your application logic. Follow the links below to learn more about each approach to instrument your application:

[![Automatic Tracing](/mlflow-website/docs/latest/images/llms/tracing/app-instrumentation/autolog-logos.png)](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/automatic.md)

### [Automatic Tracing](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/automatic.md)

[Instrument your application with a single line of code. We recommend starting from here.](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/automatic.md)

[Learn more →](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/automatic.md)

[![Manual Tracing](/mlflow-website/docs/latest/images/llms/tracing/app-instrumentation/manual-tracing2.png)](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/manual-tracing.md)

### [Manual Tracing](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/manual-tracing.md)

[Instrument any Python code with a few lines of code, with full control and flexibility.](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/manual-tracing.md)

[Learn more →](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/manual-tracing.md)

[![Typescript SDK](/mlflow-website/docs/latest/images/logos/javascript-typescript-logo.png)](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/typescript-sdk.md)

### [Typescript SDK](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/typescript-sdk.md)

[Instrument your Node.js applications with MLflow Tracing Typescript SDK.](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/typescript-sdk.md)

[Learn more →](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/typescript-sdk.md)

[![OpenTelemetry](/mlflow-website/docs/latest/images/logos/opentelemetry-logo.svg)](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/opentelemetry.md)

### [OpenTelemetry](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/opentelemetry.md)

[When you are using OpenTelemetry as your observability platform, you can export traces to MLflow directly.](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/opentelemetry.md)

[Learn more →](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/opentelemetry.md)

### 3. Choose Destination for Your Traces[​](#3-choose-destination-for-your-traces "Direct link to 3. Choose Destination for Your Traces")

MLflow Tracing supports exporting traces to various destinations.

[![MLflow Experiment](/mlflow-website/docs/latest/images/logos/mlflow-logo.svg)](/mlflow-website/docs/latest/self-hosting.md)

### [MLflow Experiment](/mlflow-website/docs/latest/self-hosting.md)

[Export traces to MLflow Tracking Server hosted on your machine.](/mlflow-website/docs/latest/self-hosting.md)

[Self-hosting guide →](/mlflow-website/docs/latest/self-hosting.md)

[![Databricks](/mlflow-website/docs/latest/images/logos/databricks-logo.png)](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/)

### [Databricks](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/)

[Export traces to Databricks for centralized monitoring and governance.](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/)

[View documentation →](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/)

[![OpenTelemetry](/mlflow-website/docs/latest/images/logos/opentelemetry-logo.svg)](/mlflow-website/docs/latest/genai/tracing/opentelemetry/export.md)

### [OpenTelemetry](/mlflow-website/docs/latest/genai/tracing/opentelemetry/export.md)

[Export traces to any observability platform that supports OpenTelemetry protocol.](/mlflow-website/docs/latest/genai/tracing/opentelemetry/export.md)

[Learn more →](/mlflow-website/docs/latest/genai/tracing/opentelemetry/export.md)

## Common System Patterns[​](#common-system-patterns "Direct link to Common System Patterns")

### Production Considerations[​](#production-considerations "Direct link to Production Considerations")

MLflow Tracing is production ready, but in order to ensure the scalability and reliability of the tracing system, we recommend the following best practices:

1. Enable [Async Logging](/mlflow-website/docs/latest/genai/tracing/prod-tracing.md#asynchronous-trace-logging) and set up appropriate queue size and timeout.
2. Use the lightweight [`mlflow-tracing`](https://pypistats.org/packages/mlflow-tracing) package for minimizing the package footprint and dependencies.
3. Use [managed MLflow services](/mlflow-website/docs/latest/genai/tracing/prod-tracing.md#managed-monitoring-with-databricks) for reducing the operational overhead and ensure the scalability of the tracing system.
4. When using self-hosted MLflow, make sure to use the **SQL Backend** with a scalable database like PostgreSQL. The default file-based backend has scalability limitations and is not recommended for production use.

### Async Applications[​](#async-applications "Direct link to Async Applications")

Async programming is an effective tool for improving the throughput of your application, particularly for LLM-based applications that are typically I/O bound. MLflow Tracing natively [supports instrumenting async applications](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/manual-tracing.md#async-support).

### Multi-Threaded Applications[​](#multi-threaded-applications "Direct link to Multi-Threaded Applications")

Multi-threading is a common strategy for parallelizing IO-bound operations in applications. MLflow Tracing supports [multi-threaded applications](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/manual-tracing.md#multi-threading) using context propagation.

### Managing User sessions[​](#managing-user-sessions "Direct link to Managing User sessions")

Many LLM-based applications are deployed as chat-based applications, where each user session is a separate thread. Grouping traces by user session is a common practice. MLflow Tracing supports [managing user sessions](/mlflow-website/docs/latest/genai/tracing/track-users-sessions.md).

### Redacting PII Data[​](#redacting-pii-data "Direct link to Redacting PII Data")

Traces can contain sensitive data such as raw user inputs, internal document contents, etc. MLflow Tracing supports [redacting PII data](/mlflow-website/docs/latest/genai/tracing/observe-with-traces/masking.md) using flexible masking rules, custom functions, and integration with external PII masking libraries.

### Collecting User Feedbacks[​](#collecting-user-feedbacks "Direct link to Collecting User Feedbacks")

User feedback is a valuable source of information for improving the user experience of your application. MLflow Tracing supports [collecting user feedback on traces](/mlflow-website/docs/latest/genai/tracing/collect-user-feedback.md) to track and analyze the feedbacks effectively.
