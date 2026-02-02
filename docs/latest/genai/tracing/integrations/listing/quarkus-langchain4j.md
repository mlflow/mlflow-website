# Tracing Quarkus LangChain4j

![Quarkus LangChain4j Logo](/mlflow-website/docs/latest/images/logos/langchain4j.svg)

#### Integration via OpenTelemetry

Quarkus LangChain4j<!-- --> can be integrated with MLflow via OpenTelemetry. Configure <!-- -->Quarkus LangChain4j<!-- -->'s OpenTelemetry exporter to send traces to MLflow's OTLP endpoint.

info

OpenTelemetry trace ingestion is supported in **MLflow 3.6.0 and above**.

## OpenTelemetry endpoint (OTLP)[​](#opentelemetry-endpoint-otlp "Direct link to OpenTelemetry endpoint (OTLP)")

MLflow Server exposes an OTLP endpoint at `/v1/traces` ([OTLP](https://opentelemetry.io/docs/specs/otlp/)). This endpoint accepts traces from any native OpenTelemetry instrumentation, allowing you to trace applications written in other languages such as Java, Go, Rust, etc.

To use this endpoint, start MLflow Server with a SQL-based backend store. The following command starts MLflow Server with an SQLite backend store:

bash

```bash
mlflow server

```

To use other types of SQL databases such as PostgreSQL, MySQL, and MSSQL, change the store URI as described in the [backend store documentation](/mlflow-website/docs/latest/self-hosting/architecture/backend-store.md).

In your application, configure the server endpoint and set the MLflow experiment ID in the OTLP header `x-mlflow-experiment-id`.

bash

```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:5000/v1/traces
export OTEL_EXPORTER_OTLP_TRACES_HEADERS=x-mlflow-experiment-id=123

```

note

Currently, MLflow Server supports only the OTLP/HTTP endpoint, and the OTLP/gRPC endpoint is not yet supported.

## Enable OpenTelemetry in Quarkus LangChain4j[​](#enable-opentelemetry-in-quarkus-langchain4j "Direct link to Enable OpenTelemetry in Quarkus LangChain4j")

Refer to the [Quarkus LangChain4j Observability documentation](https://docs.quarkiverse.io/quarkus-langchain4j/dev/observability.html) for setting up tracing in Quarkus LangChain4j and specify OTLP HTTP exporter with above environment variables.

## Reference[​](#reference "Direct link to Reference")

For complete step-by-step instructions on sending traces to MLflow from OpenTelemetry compatible frameworks, see the [Collect OpenTelemetry Traces into MLflow](/mlflow-website/docs/latest/genai/tracing/opentelemetry/ingest.md).
