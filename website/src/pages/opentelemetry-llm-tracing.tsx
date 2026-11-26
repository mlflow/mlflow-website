import { useState } from "react";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../components/Header/Header";
import { SocialLinksFooter } from "../components/SocialLinksFooter/SocialLinksFooter";
import { ArticleSidebar } from "../components/ArticleSidebar/ArticleSidebar";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import TraceDetail from "@site/static/img/opentelemetry-llm-tracing/mlflow-trace-detail-demo.png";
import { CopyButton } from "../components/CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";

const SEO_TITLE =
  "OpenTelemetry LLM Tracing with MLflow | OTLP & GenAI Semconv";
const SEO_DESCRIPTION =
  "Trace LLM apps and agents with OpenTelemetry. MLflow ingests OTLP traces, exports to any OTel backend, and supports GenAI semantic conventions. Open source.";
const PAGE_URL = "https://mlflow.org/opentelemetry-llm-tracing";
const OTEL_DOCS = MLFLOW_GENAI_DOCS_URL + "tracing/opentelemetry/";

const faqs: { question: string; answer: string }[] = [
  {
    question: "What is OpenTelemetry LLM tracing?",
    answer:
      "OpenTelemetry LLM tracing means recording each step of an LLM application or agent (model calls, retrievals, tool calls) as OpenTelemetry spans, so the trace can travel over the vendor-neutral OTLP protocol to any compatible backend. The OpenTelemetry GenAI semantic conventions define the standard attribute names for model, provider, token usage, and messages, so different tools can read the same data.",
  },
  {
    question: "Does MLflow support OpenTelemetry?",
    answer:
      "Yes. The MLflow Tracing SDK is built on the OpenTelemetry SDK, the MLflow Server exposes an OTLP endpoint at /v1/traces for ingesting traces from any OpenTelemetry client, MLflow can export traces and metrics to any OTLP-compatible backend, and MLflow reads and writes the OpenTelemetry GenAI semantic conventions.",
  },
  {
    question: "How do I send OpenTelemetry traces to MLflow?",
    answer:
      "Start the MLflow Server with a SQL backend store, set OTEL_EXPORTER_OTLP_TRACES_ENDPOINT to http://<server>/v1/traces, and set the x-mlflow-experiment-id header in OTEL_EXPORTER_OTLP_TRACES_HEADERS to the ID of the experiment that should hold the traces. Any OpenTelemetry SDK or the OpenTelemetry Collector can then export to MLflow. Ingestion is supported in MLflow 3.6.0 and above.",
  },
  {
    question: "Does the MLflow OTLP endpoint support gRPC?",
    answer:
      "No. The MLflow Server currently accepts OTLP over HTTP only. If your application exports over gRPC, route it through an OpenTelemetry Collector and use the otlphttp exporter to forward the traces to MLflow.",
  },
  {
    question: "Can I trace applications written in Java, Go, or Rust?",
    answer:
      "Yes. Because the MLflow Server accepts standard OTLP, any language with an OpenTelemetry SDK can send traces to it. The MLflow Tracing SDK itself is available for Python and TypeScript/JavaScript, and the OTLP endpoint covers other languages such as Java, Go, and Rust.",
  },
  {
    question: "Can I export MLflow traces to Datadog, Grafana, or Jaeger?",
    answer:
      "Yes. Set OTEL_EXPORTER_OTLP_TRACES_ENDPOINT to your OpenTelemetry Collector or backend before starting a trace, and MLflow exports spans over OTLP. MLflow's exporter uses gRPC by default; set OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf to export to an HTTP endpoint. Set MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT=true to keep sending traces to the MLflow Tracking Server at the same time. The MLflow docs link to setup guides for Datadog, New Relic, SigNoz, Splunk, Grafana, Jaeger, Dynatrace, and others.",
  },
  {
    question: "What are the OpenTelemetry GenAI semantic conventions?",
    answer:
      "They are a CNCF-backed standard that names the attributes of AI telemetry, such as gen_ai.operation.name, gen_ai.request.model, gen_ai.provider.name, gen_ai.usage.input_tokens, and gen_ai.input.messages. MLflow ingests traces that use these attributes, and it can export its own traces in this format by setting MLFLOW_ENABLE_OTEL_GENAI_SEMCONV=true (MLflow 3.11 and above).",
  },
  {
    question:
      "Can I use MLflow auto-tracing together with other OpenTelemetry instrumentation?",
    answer:
      "Yes. Set MLFLOW_USE_DEFAULT_TRACER_PROVIDER to false so MLflow uses the global OpenTelemetry tracer provider, then call mlflow.tracing.set_destination so MLflow's span processors attach to it. Spans from MLflow auto-tracing and from libraries such as the FastAPI instrumentation then land in the same trace.",
  },
  {
    question:
      "Do OpenTelemetry traces in MLflow work with evaluation and monitoring?",
    answer:
      "Traces stored in an MLflow experiment appear in the MLflow UI with span trees, inputs, outputs, and token usage, and you can search them and attach feedback. Because they are regular MLflow traces, you can also use them as data for MLflow evaluation with LLM judges and other scorers.",
  },
  {
    question: "Is MLflow tracing free and open source?",
    answer:
      "Yes. MLflow is open source under the Apache 2.0 license and backed by the Linux Foundation. Tracing, the OTLP endpoint, and OpenTelemetry export are all part of the open-source project, and you control where trace data is stored.",
  },
];

const faqJsonLd = {
  "@context": "https://schema.org",
  "@type": "FAQPage",
  mainEntity: faqs.map((faq) => ({
    "@type": "Question",
    name: faq.question,
    acceptedAnswer: { "@type": "Answer", text: faq.answer },
  })),
};

const articleJsonLd = {
  "@context": "https://schema.org",
  "@type": "TechArticle",
  headline: "OpenTelemetry LLM Tracing with MLflow",
  description: SEO_DESCRIPTION,
  mainEntityOfPage: PAGE_URL,
  about: ["OpenTelemetry", "LLM tracing", "OTLP", "GenAI semantic conventions"],
  publisher: {
    "@type": "Organization",
    name: "MLflow",
    url: "https://mlflow.org",
  },
};

const softwareJsonLd = {
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  name: "MLflow",
  applicationCategory: "DeveloperApplication",
  operatingSystem: "Cross-platform",
  offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
  description:
    "Open-source AI engineering platform with OpenTelemetry-compatible tracing for LLM applications and agents.",
  url: "https://mlflow.org",
  license: "https://www.apache.org/licenses/LICENSE-2.0",
};

function CodeBlock({ code, language }: { code: string; language: string }) {
  return (
    <div
      className="rounded-lg border border-white/10 overflow-hidden"
      style={{ backgroundColor: CODE_BG, margin: "8px 0 24px" }}
    >
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
        <span className="text-xs text-white/50 font-mono">{language}</span>
        <CopyButton code={code} />
      </div>
      <div className="p-3 overflow-x-auto">
        <Highlight theme={customNightOwl} code={code} language={language}>
          {({ style, tokens, getLineProps, getTokenProps }) => (
            <pre
              className="text-xs font-mono !m-0 !p-0 text-left"
              style={{ ...style, backgroundColor: "transparent" }}
            >
              {tokens.map((line, i) => (
                <div key={i} {...getLineProps({ line })}>
                  {line.map((token, key) => (
                    <span key={key} {...getTokenProps({ token })} />
                  ))}
                </div>
              ))}
            </pre>
          )}
        </Highlight>
      </div>
    </div>
  );
}

function Table({
  headers,
  rows,
}: {
  headers: string[];
  rows: React.ReactNode[][];
}) {
  const cell = { padding: "14px 16px", textAlign: "left" as const };
  return (
    <div style={{ overflowX: "auto", margin: "32px 0" }}>
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: "14px",
          lineHeight: "1.6",
        }}
      >
        <thead>
          <tr
            style={{
              backgroundColor: "#f8fafc",
              borderBottom: "2px solid #e2e8f0",
            }}
          >
            {headers.map((h) => (
              <th
                key={h}
                style={{ ...cell, fontWeight: 600, color: "#1a1a1a" }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} style={{ borderBottom: "1px solid #e2e8f0" }}>
              {row.map((c, j) => (
                <td
                  key={j}
                  style={{
                    ...cell,
                    color: "#3d3d3d",
                    fontWeight: j === 0 ? 600 : 400,
                    verticalAlign: "top",
                  }}
                >
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function OpenTelemetryLLMTracing() {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content={PAGE_URL} />
        <link rel="canonical" href={PAGE_URL} />
        <script type="application/ld+json">{JSON.stringify(faqJsonLd)}</script>
        <script type="application/ld+json">
          {JSON.stringify(articleJsonLd)}
        </script>
        <script type="application/ld+json">
          {JSON.stringify(softwareJsonLd)}
        </script>
        <style>{`
          /* Import MLflow docs fonts */
          @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300;1,400;1,500&family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&display=swap');

          /* Black header */
          header, nav {
            background: #000000 !important;
          }

          body {
            background: #ffffff;
            margin: 0;
            padding: 0;
            font-family: 'DM Sans', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          }
          .article-page {
            background: #ffffff;
            min-height: 100vh;
          }
          .article-container {
            max-width: 900px;
            margin: 0 auto;
            padding: 60px 24px 100px;
          }
          .article-container h1 {
            font-family: 'DM Sans', sans-serif;
            font-size: 3rem !important;
            font-weight: 700 !important;
            color: #1a1a1a !important;
            margin: 48px 0 32px 0 !important;
            line-height: 1.0 !important;
            letter-spacing: -0.03em !important;
          }
          .article-container h2 {
            font-family: 'DM Sans', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            color: #1a1a1a;
            margin: 64px 0 32px 0;
            line-height: 1.2;
            letter-spacing: -0.01em;
          }
          .article-container h3 {
            font-family: 'DM Sans', sans-serif;
            font-size: 1.5rem;
            font-weight: 600;
            color: #1a1a1a;
            margin: 32px 0 16px 0;
            line-height: 1.2;
          }
          .article-container p {
            font-family: 'DM Sans', sans-serif;
            font-size: 16px;
            color: #3d3d3d;
            line-height: 1.7;
            margin: 0 0 24px 0;
          }
          .article-container strong {
            font-weight: 600;
            color: #1a1a1a;
          }
          .article-container a {
            color: #0194e2 !important;
            text-decoration: none;
            transition: all 0.2s ease;
          }
          .article-container a:hover {
            color: #0072b0 !important;
            text-decoration: underline;
          }
          .info-box a {
            color: #0194e2 !important;
            font-weight: 500;
          }
          .info-box a:hover {
            color: #0072b0 !important;
          }
          .article-container ul {
            list-style-type: disc;
            margin: 20px 0;
            padding-left: 24px;
            list-style-position: outside;
          }
          .article-container li {
            font-family: 'DM Sans', sans-serif;
            font-size: 16px;
            color: #3d3d3d;
            line-height: 1.7;
            margin-bottom: 20px;
            padding-left: 8px;
            text-wrap: pretty;
          }
          .article-container li a {
            font-weight: 600;
            color: #0194e2 !important;
          }
          .grid-2 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 24px;
            margin: 40px 0 56px 0;
          }
          .card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            padding: 32px;
            box-shadow: 0 0 0 1px rgba(50, 50, 93, 0.05), 0 0 14px 5px rgba(50, 50, 93, 0.08), 0 0 10px 3px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
          }
          .card:hover {
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
          }
          .card h3 {
            font-family: 'DM Sans', sans-serif;
            margin-top: 0 !important;
            margin-bottom: 32px !important;
            font-size: 1.3rem;
            font-weight: 600;
            color: #1a1a1a;
            line-height: 1.2;
          }
          .card p {
            font-family: 'DM Sans', sans-serif;
            margin-bottom: 14px;
            font-size: 14px;
            line-height: 1.5;
            color: #505050;
          }
          .card p:last-child {
            margin-bottom: 0;
          }
          .faq-list {
            margin: 32px 0;
          }
          .faq-item {
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            margin-bottom: 16px;
            background: #ffffff;
            transition: all 0.2s ease;
          }
          .faq-item:hover {
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
          }
          .faq-question {
            font-family: 'DM Sans', sans-serif;
            padding: 20px 24px;
            font-size: 18px;
            font-weight: 500;
            color: #1a1a1a;
            cursor: pointer;
            background: transparent;
            border: none;
            width: 100%;
            text-align: left;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.2s ease;
          }
          .faq-question:hover {
            background: #f9fafb;
          }
          .faq-answer {
            font-family: 'DM Sans', sans-serif;
            padding: 0 24px 20px;
            font-size: 16px;
            color: #3d3d3d;
            line-height: 1.6;
          }
          .faq-chevron {
            transition: transform 0.2s ease;
            flex-shrink: 0;
            margin-left: 16px;
            color: #6b7280;
            font-size: 12px;
          }
          .faq-chevron.open {
            transform: rotate(180deg);
          }
          .info-box {
            background: rgb(238, 249, 253);
            border: 1px solid rgb(41, 120, 115);
            padding: 20px 24px;
            margin: 40px 0;
            border-radius: 5.6px;
          }
          .info-box p {
            font-family: 'DM Sans', sans-serif;
            margin: 0;
            font-size: 16px;
            line-height: 1.6;
            color: #1e293b;
          }
          .article-sidebar {
            position: fixed;
            top: 100px;
            left: calc(50% + 900px / 2 + 48px);
            width: 280px;
            max-height: calc(100vh - 120px);
            overflow-y: auto;
          }
          .article-sidebar .toc-title {
            font-family: 'DM Sans', sans-serif;
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9ca3af;
            margin: 0 0 12px 0;
          }
          .article-sidebar ul {
            margin: 0;
            padding: 0;
            list-style: none;
            border-left: 1px solid #e5e7eb;
          }
          .article-sidebar li {
            margin: 0;
            padding: 0;
          }
          .article-sidebar a {
            font-family: 'DM Sans', sans-serif;
            display: block;
            padding: 8px 0 8px 16px;
            font-size: 16px;
            color: #0194e2 !important;
            text-decoration: none !important;
            transition: all 0.15s ease;
            line-height: 1.4;
          }
          .article-sidebar a:hover {
            color: #0072b0 !important;
          }
          .article-sidebar .toc-divider {
            border: none;
            border-top: 1px solid #e5e7eb;
            margin: 12px 0 12px 0;
          }
          @media (max-width: 1600px) {
            .article-sidebar {
              display: none;
            }
          }
          @media (max-width: 768px) {
            .article-container {
              padding: 40px 20px 80px;
            }
            .article-container h1 {
              font-size: 36px !important;
              margin-bottom: 24px !important;
            }
            .article-container h2 {
              font-size: 28px;
              margin: 40px 0 20px 0;
            }
            .grid-2 {
              grid-template-columns: 1fr;
              gap: 20px;
            }
            .card {
              padding: 24px;
            }
            .faq-question {
              padding: 18px 20px;
              font-size: 17px;
            }
            .faq-answer {
              padding: 0 20px 20px;
              font-size: 16px;
            }
          }
        `}</style>
      </Head>

      <div className="article-page">
        <Header />

        <div className="article-container">
          <h1>OpenTelemetry LLM Tracing with MLflow</h1>

          <p>
            <strong>OpenTelemetry LLM tracing</strong> records every step of an
            LLM application or agent (model calls, retrieval, tool calls) as
            OpenTelemetry spans, so the trace can move between tools over the
            vendor-neutral OTLP protocol. MLflow supports it in both directions:
            the MLflow Server ingests OTLP traces from any OpenTelemetry client,
            and MLflow Tracing exports traces and metrics to any OTLP-compatible
            backend.
          </p>

          <p>
            MLflow Tracing is built on the OpenTelemetry SDK and reads and
            writes the{" "}
            <a href="#genai-semconv">
              OpenTelemetry GenAI semantic conventions
            </a>
            , so your trace data is not tied to one vendor's format. You can
            instrument with the MLflow SDK, with plain OpenTelemetry libraries
            in any language, or with both in the same application.
          </p>

          <div style={{ margin: "40px 0", textAlign: "center" }}>
            <img
              src={TraceDetail}
              alt="MLflow UI showing a trace of an agent run with nested spans for model calls and tool calls, plus latency, token count, and cost per span"
              style={{
                width: "100%",
                borderRadius: "8px",
                boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
              }}
            />
            <p
              style={{
                marginTop: "12px",
                fontSize: "14px",
                color: "#6b7280",
                fontStyle: "italic",
              }}
            >
              A trace in the MLflow UI: nested spans for model and tool calls,
              with latency, token usage, and cost on each span (MLflow demo
              data)
            </p>
          </div>

          <div
            style={{
              background: "#f9fafb",
              border: "1px solid #e5e7eb",
              borderRadius: "8px",
              padding: "20px 24px",
              margin: "40px 0",
            }}
          >
            <p style={{ marginBottom: "12px" }}>
              <strong>Quick Navigation:</strong>
            </p>
            <ul style={{ margin: 0, paddingLeft: "24px" }}>
              <li style={{ marginBottom: "8px" }}>
                <a href="#what-is">What is OpenTelemetry LLM tracing?</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#three-ways">
                  Three ways to use OpenTelemetry with MLflow
                </a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#ingest">How to send OpenTelemetry traces to MLflow</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#export">
                  Export MLflow traces to other OpenTelemetry backends
                </a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#genai-semconv">GenAI semantic conventions</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#combine">
                  Combine OpenTelemetry libraries with MLflow auto-tracing
                </a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#why-mlflow">What MLflow adds to OpenTelemetry</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#get-started">Get started</a>
              </li>
              <li style={{ marginBottom: "0" }}>
                <a href="#faq">FAQ</a>
              </li>
            </ul>
          </div>

          <h2 id="what-is">What Is OpenTelemetry LLM Tracing?</h2>

          <p>
            <a href="https://opentelemetry.io/">OpenTelemetry</a> is a
            CNCF-backed project that defines vendor-neutral APIs, SDKs, and a
            wire protocol (OTLP) for telemetry. A <strong>trace</strong> is a
            tree of <strong>spans</strong>, where each span is one timed
            operation with attributes. For an LLM application, a span might be a
            call to a model, a vector search, or a tool invocation, nested under
            the span for the request or agent run that triggered it.
          </p>

          <p>
            What makes LLM tracing different from ordinary service tracing is
            the content of the spans: prompts and responses, the model and
            provider, token counts, and the structure of an agent's reasoning
            steps. The{" "}
            <a href="https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/">
              OpenTelemetry GenAI semantic conventions
            </a>{" "}
            standardize the attribute names for this data so that a trace
            produced by one tool can be understood by another.
          </p>

          <p>
            Using OpenTelemetry for LLM tracing means you can instrument once
            and choose where the data goes: an AI-focused platform for debugging
            and evaluation, an existing observability stack for operations, or
            both. For the broader picture of tracing agents and LLM
            applications, see{" "}
            <Link href="/llm-tracing">LLM tracing with MLflow</Link>.
          </p>

          <h2 id="three-ways">How Can You Use OpenTelemetry with MLflow?</h2>

          <Table
            headers={["Direction", "What it does", "Use it when"]}
            rows={[
              [
                "Instrument with the MLflow SDK",
                "MLflow Tracing, built on the OpenTelemetry SDK, adds one-line auto-tracing for popular LLM and agent frameworks and enriches spans with prompts, token usage, and model names.",
                "Your application is in Python or TypeScript/JavaScript and you want the fastest path to useful traces.",
              ],
              [
                "Ingest OTLP into MLflow",
                "The MLflow Server accepts OTLP traces over HTTP at /v1/traces.",
                "You use OpenTelemetry instrumentation directly, or write in Java, Go, Rust, or another language.",
              ],
              [
                "Export from MLflow over OTLP",
                "MLflow sends traces and span-duration metrics to any OTLP endpoint, optionally at the same time as MLflow.",
                "You already run Datadog, Grafana, Jaeger, or another OpenTelemetry-compatible backend.",
              ],
            ]}
          />

          <h2 id="ingest">How Do You Send OpenTelemetry Traces to MLflow?</h2>

          <p>
            The MLflow Server exposes an OTLP endpoint at{" "}
            <code>/v1/traces</code>. Start the server with a SQL backend store,
            then point any OpenTelemetry exporter at it and tell MLflow which
            experiment should hold the traces with the{" "}
            <code>x-mlflow-experiment-id</code> header. Trace ingestion is
            available in MLflow 3.6.0 and above.
          </p>

          <CodeBlock
            language="bash"
            code={`# Start MLflow (SQLite backend by default)
mlflow server

# In your application's environment
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:5000/v1/traces
export OTEL_EXPORTER_OTLP_TRACES_HEADERS=x-mlflow-experiment-id=123`}
          />

          <p>
            Any OpenTelemetry SDK reads these standard variables, so the same
            setup works for a FastAPI app instrumented with OpenTelemetry, a
            Java service using Spring AI, or a Go worker. If your applications
            send traces to an OpenTelemetry Collector, add an{" "}
            <code>otlphttp</code> exporter that targets MLflow:
          </p>

          <CodeBlock
            language="yaml"
            code={`receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

exporters:
  otlphttp:
    endpoint: http://localhost:5000
    headers:
      x-mlflow-experiment-id: "123"

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlphttp]`}
          />

          <p>A few details worth knowing:</p>
          <ul>
            <li>
              The MLflow Server accepts OTLP over <strong>HTTP only</strong>;
              OTLP/gRPC is not supported yet. A Collector can receive gRPC from
              your apps and forward over HTTP.
            </li>
            <li>
              Since MLflow 3.7.0 the endpoint accepts <code>gzip</code> and{" "}
              <code>deflate</code> compressed payloads. Clients enable it with{" "}
              <code>OTEL_EXPORTER_OTLP_TRACES_COMPRESSION</code> on their
              exporter.
            </li>
            <li>
              When workspaces are enabled on the server, clients can send the{" "}
              <code>X-MLFLOW-WORKSPACE</code> header to override the default
              workspace.
            </li>
          </ul>

          <p>
            See{" "}
            <Link href={OTEL_DOCS + "ingest"}>
              Collect OpenTelemetry Traces into MLflow
            </Link>{" "}
            for the full guide.
          </p>

          <h2 id="export">
            How Do You Export MLflow Traces to Other OpenTelemetry Backends?
          </h2>

          <p>
            Traces created by MLflow follow the OpenTelemetry trace spec, so you
            can send them to any platform that accepts OTLP. Set the endpoint
            before starting a trace and install the exporter package. MLflow's
            OTLP exporter uses gRPC unless you set the protocol, so the example
            below selects <code>http/protobuf</code> to match an HTTP endpoint:
          </p>

          <CodeBlock
            language="python"
            code={`# pip install opentelemetry-exporter-otlp
import os
import mlflow

os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://localhost:4318/v1/traces"
os.environ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = "http/protobuf"
os.environ["OTEL_SERVICE_NAME"] = "my-llm-service"  # optional, groups traces

with mlflow.start_span(name="foo") as span:
    span.set_inputs({"a": 1})
    span.set_outputs({"b": 2})`}
          />

          <p>
            By default, once an OTLP endpoint is configured MLflow sends traces
            to that endpoint only. Set{" "}
            <code>MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT=true</code> to{" "}
            <strong>dual export</strong>: traces go to both the MLflow Tracking
            Server and your OpenTelemetry backend.
          </p>

          <CodeBlock
            language="python"
            code={`import os
import mlflow

os.environ["MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT"] = "true"
os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://localhost:4318/v1/traces"
os.environ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = "http/protobuf"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-experiment")

with mlflow.start_span(name="foo") as span:
    span.set_inputs({"a": 1})
    span.set_outputs({"b": 2})`}
          />

          <p>
            All standard OpenTelemetry exporter settings apply, such as{" "}
            <code>OTEL_EXPORTER_OTLP_TRACES_PROTOCOL</code> and{" "}
            <code>OTEL_EXPORTER_OTLP_TRACES_HEADERS</code> for API keys. The
            MLflow docs include setup links for Datadog, New Relic, SigNoz,
            Splunk, Grafana, Jaeger, Dynatrace, and other
            OpenTelemetry-compatible platforms.
          </p>

          <p>
            <strong>Metrics.</strong> When an OTLP metrics endpoint is
            configured (<code>OTEL_EXPORTER_OTLP_METRICS_ENDPOINT</code>),
            MLflow also exports a <code>mlflow.trace.span.duration</code>{" "}
            histogram in milliseconds. It carries attributes such as the span
            type, span status, whether the span is the root, and the experiment
            ID, so you can chart latency and error rates in Prometheus, Grafana,
            or whichever metrics system you run. See{" "}
            <Link href={OTEL_DOCS + "export"}>
              Export MLflow Traces/Metrics via OTLP
            </Link>
            .
          </p>

          <h2 id="genai-semconv">
            What Are the OpenTelemetry GenAI Semantic Conventions?
          </h2>

          <p>
            The OpenTelemetry GenAI semantic conventions define a shared schema
            for AI telemetry. MLflow supports them on both sides:
          </p>

          <ul>
            <li>
              <strong>Ingest:</strong> the MLflow OTLP endpoint recognizes
              traces that use <code>gen_ai.*</code> attributes and shows them
              with proper span types, token counts, model information, and
              structured inputs and outputs. The MLflow docs list integrations
              that produce them, including{" "}
              <Link
                href={
                  MLFLOW_GENAI_DOCS_URL +
                  "tracing/integrations/listing/google-adk"
                }
              >
                Google ADK
              </Link>
              ,{" "}
              <Link
                href={
                  MLFLOW_GENAI_DOCS_URL + "tracing/integrations/listing/livekit"
                }
              >
                LiveKit Agents
              </Link>
              , and{" "}
              <Link
                href={
                  MLFLOW_GENAI_DOCS_URL +
                  "tracing/integrations/listing/spring-ai"
                }
              >
                Spring AI
              </Link>
              .
            </li>
            <li>
              <strong>Export:</strong> by default MLflow exports its own{" "}
              <code>mlflow.*</code> attributes. Set{" "}
              <code>MLFLOW_ENABLE_OTEL_GENAI_SEMCONV=true</code> (MLflow 3.11
              and above) to translate them to <code>gen_ai.*</code> at export
              time. Message conversion is supported for OpenAI (Chat Completions
              and Responses), Anthropic, Google Gemini, and Amazon Bedrock.
            </li>
          </ul>

          <Table
            headers={[
              "MLflow attribute",
              "GenAI semantic convention attribute",
            ]}
            rows={[
              [
                <code key="a">mlflow.spanType</code>,
                <code key="b">gen_ai.operation.name</code>,
              ],
              [
                <code key="a">mlflow.llm.model</code>,
                <code key="b">gen_ai.request.model</code>,
              ],
              [
                <code key="a">mlflow.llm.provider</code>,
                <code key="b">gen_ai.provider.name</code>,
              ],
              [
                <code key="a">mlflow.chat.tokenUsage</code>,
                <code key="b">
                  gen_ai.usage.input_tokens / gen_ai.usage.output_tokens
                </code>,
              ],
              [
                <code key="a">mlflow.spanInputs</code>,
                <code key="b">gen_ai.input.messages</code>,
              ],
              [
                <code key="a">mlflow.spanOutputs</code>,
                <code key="b">gen_ai.output.messages</code>,
              ],
            ]}
          />

          <p>
            The full mapping is in the{" "}
            <Link href={OTEL_DOCS + "attribute-mapping"}>
              attribute mapping reference
            </Link>
            .
          </p>

          <h2 id="combine">
            Can You Combine OpenTelemetry Libraries with MLflow Auto-Tracing?
          </h2>

          <p>
            Because MLflow Tracing is built on the OpenTelemetry SDK, you can
            keep the OpenTelemetry instrumentation you already have for HTTP
            frameworks, databases, and network calls, and add MLflow's
            AI-specific tracing on top. Set{" "}
            <code>MLFLOW_USE_DEFAULT_TRACER_PROVIDER</code> to{" "}
            <code>false</code> so MLflow uses the global tracer provider, then
            call <code>mlflow.tracing.set_destination</code> after your provider
            is configured. MLflow spans nest under the OpenTelemetry spans in a
            single trace.
          </p>

          <CodeBlock
            language="python"
            code={`import os
import mlflow
from mlflow.entities.trace_location import MlflowExperimentLocation
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider

# Use the OpenTelemetry tracer provider instead of MLflow's default one
os.environ["MLFLOW_USE_DEFAULT_TRACER_PROVIDER"] = "false"

mlflow.set_tracking_uri("http://localhost:5000")
experiment_id = mlflow.set_experiment("my_experiment").experiment_id

otel_trace.set_tracer_provider(TracerProvider())

# Must be called after the tracer provider is configured
mlflow.tracing.set_destination(MlflowExperimentLocation(experiment_id))

otel_tracer = otel_trace.get_tracer("my_app")

with otel_tracer.start_as_current_span("http_request") as external_span:
    external_span.set_attribute("http.method", "GET")

    # This MLflow span is nested under the OpenTelemetry span
    with mlflow.start_span("model_prediction") as mlflow_span:
        mlflow_span.set_inputs({"query": "test"})
        mlflow_span.set_outputs({"result": "success"})`}
          />

          <p>
            The same approach lets MLflow's one-line auto-tracing, for example{" "}
            <code>mlflow.openai.autolog()</code>, share a trace with spans from
            OpenTelemetry's FastAPI instrumentation. See{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "tracing/app-instrumentation/opentelemetry"
              }
            >
              Tracing with OpenTelemetry
            </Link>{" "}
            for a full FastAPI example.
          </p>

          <h2 id="why-mlflow">What Does MLflow Add to OpenTelemetry?</h2>

          <p>
            Any OTLP backend can store spans. Debugging an LLM application
            usually needs more than storage, and this is where MLflow focuses:
          </p>

          <ul>
            <li>
              <strong>An AI-aware trace UI.</strong> Span trees show prompts,
              responses, tool calls, token usage, and cost together instead of
              as raw attributes.
            </li>
            <li>
              <strong>Auto-tracing for popular frameworks.</strong> One line of
              code instruments OpenAI, Anthropic, LangChain, LangGraph, and many
              other providers and frameworks (see the{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/integrations"}>
                integrations list
              </Link>
              ).
            </li>
            <li>
              <strong>Evaluation and feedback on the same traces.</strong> You
              can{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/search-traces"}>
                search traces
              </Link>
              , attach human or automated feedback, and run{" "}
              <Link href="/llm-evaluation">LLM evaluation</Link> with judges
              against the traces you collected.
            </li>
            <li>
              <strong>Ownership of the data.</strong> MLflow is open source
              under Apache 2.0, and you decide where traces are stored. With
              dual export you can keep an existing observability stack as well.
            </li>
          </ul>

          <p>
            If you only need generic service monitoring, an OpenTelemetry
            backend alone may be enough. You can also run both: OTLP for the
            operations view and MLflow for debugging and improving the AI
            behavior.
          </p>

          <h2 id="get-started">
            How Do You Get Started with OpenTelemetry Tracing in MLflow?
          </h2>

          <ol>
            <li>
              Install and start MLflow: <code>pip install mlflow</code>, then{" "}
              <code>mlflow server</code>.
            </li>
            <li>Create an experiment in the MLflow UI and note its ID.</li>
            <li>
              Pick a path: instrument with{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/quickstart"}>
                the MLflow Tracing SDK
              </Link>
              , or set <code>OTEL_EXPORTER_OTLP_TRACES_ENDPOINT</code> and the{" "}
              <code>x-mlflow-experiment-id</code> header in any OpenTelemetry
              application.
            </li>
            <li>
              Run your application and open the experiment's Traces tab to
              inspect spans.
            </li>
            <li>
              Optional: set{" "}
              <code>MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT=true</code> to also
              send traces to your existing OpenTelemetry backend.
            </li>
          </ol>

          <div className="info-box">
            <p>
              <Link href="/genai" style={{ color: "#007bff" }}>
                <strong>MLflow</strong>
              </Link>{" "}
              is an open-source{" "}
              <strong>
                AI engineering platform for agents, LLMs, and ML models
              </strong>
              , backed by the Linux Foundation and licensed under Apache 2.0,
              MLflow provides OpenTelemetry-compatible tracing with no vendor
              lock-in.{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
                Read the tracing docs →
              </Link>
            </p>
          </div>

          <h2 id="faq">Frequently Asked Questions</h2>

          <div className="faq-list">
            {faqs.map((faq, index) => (
              <div key={index} className="faq-item">
                <button
                  className="faq-question"
                  onClick={() =>
                    setOpenFaqIndex(openFaqIndex === index ? null : index)
                  }
                >
                  <span>{faq.question}</span>
                  <span
                    className={`faq-chevron ${openFaqIndex === index ? "open" : ""}`}
                  >
                    ▼
                  </span>
                </button>
                {openFaqIndex === index && (
                  <div className="faq-answer">{faq.answer}</div>
                )}
              </div>
            ))}
          </div>

          <h2>Related Resources</h2>

          <ul>
            <li>
              <Link href={OTEL_DOCS}>
                OpenTelemetry Integration Documentation
              </Link>
            </li>
            <li>
              <Link href={OTEL_DOCS + "ingest"}>
                Collect OpenTelemetry Traces into MLflow
              </Link>
            </li>
            <li>
              <Link href={OTEL_DOCS + "export"}>
                Export MLflow Traces/Metrics via OTLP
              </Link>
            </li>
            <li>
              <Link href={OTEL_DOCS + "genai-semconv"}>
                GenAI Semantic Conventions in MLflow
              </Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/lightweight-sdk"}>
                mlflow-tracing: the lightweight production tracing SDK
              </Link>
            </li>
            <li>
              <Link href="/llm-tracing">LLM Tracing FAQ</Link>
            </li>
            <li>
              <Link href="/ai-observability">AI Observability FAQ</Link>
            </li>
            <li>
              <Link href="/llm-evaluation">
                LLM Evaluation and Agent Evaluation
              </Link>
            </li>
            <li>
              <Link href="/genai">MLflow for Agents and LLMs Overview</Link>
            </li>
          </ul>
        </div>

        <ArticleSidebar />
        <SocialLinksFooter />
      </div>
    </>
  );
}
