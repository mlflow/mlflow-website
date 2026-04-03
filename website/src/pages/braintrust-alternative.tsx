import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../components/Header/Header";
import { ArticleSidebar } from "../components/ArticleSidebar/ArticleSidebar";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import { CopyButton } from "../components/CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";
import MlflowTracingUI from "@site/static/img/braintrust-alternative/mlflow-tracing-ui.png";
import MlflowLogo from "@site/static/img/braintrust-alternative/mlflow-logo.png";
import MlflowEvalUI from "@site/static/img/braintrust-alternative/mlflow-eval-ui.png";
import MlflowGatewayUI from "@site/static/img/braintrust-alternative/mlflow-gateway-ui.png";
import BraintrustEvalUI from "@site/static/img/braintrust-alternative/braintrust-eval-UI.png";
import BraintrustExperiment from "@site/static/img/braintrust-alternative/braintrust-experiment.png";

function ComparisonTable({ rows }: { rows: [string, string, string][] }) {
  const [header, ...body] = rows;
  return (
    <div className="comparison-table-wrap">
      <table className="comparison-table">
        <thead>
          <tr>
            {header.map((cell, i) => (
              <th key={i}>{cell}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j} className={j === 0 ? "feature-cell" : ""}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MlflowCodeLabel() {
  return (
    <span className="code-logo-inline">
      <img src={MlflowLogo} alt="MLflow" className="code-logo-icon" />
      MLflow
    </span>
  );
}

function BraintrustCodeLabel() {
  return (
    <span className="code-logo-inline">
      Braintrust
    </span>
  );
}

function CodeBlock({ code, label }: { code: string; label: string }) {
  return (
    <div className="code-side">
      <div className="code-side-header">
        <span className="code-side-label">
          {label === "MLflow" ? <MlflowCodeLabel /> : <BraintrustCodeLabel />}
        </span>
        <CopyButton text={code} />
      </div>
      <Highlight theme={customNightOwl} code={code} language="python">
        {({ tokens, getLineProps, getTokenProps }) => (
          <pre className="code-tabs-pre" style={{ background: CODE_BG }}>
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
  );
}

const selfHostingTable: [string, string, string][] = [
  ["Feature", "MLflow", "Braintrust"],
  [
    "Architecture",
    "Server + DB + storage",
    "PostgreSQL + Redis + S3 + Brainstore + Web Server",
  ],
  [
    "Database Choices",
    "PostgreSQL, MySQL, MSSQL, SQLite, and more",
    "Locked by Braintrust",
  ],
  [
    "Storage Choices",
    "S3, R2, GCS, Azure Blob, HDFS, local",
    "AWS, GCP, and Azure supported object storage",
  ],
  [
    "Control Plane",
    "Fully self-hosted",
    "Hosted by Braintrust (hybrid)",
  ],
];

const tracingTable: [string, string, string][] = [
  ["Feature", "MLflow", "Braintrust"],
  [
    "Auto-instrumentation",
    "60+ frameworks via autolog()",
    "SDK wrappers + gateway",
  ],
  [
    "Manual tracing",
    "Python, R, JS/TS, Java SDKs",
    "Python, TS, Go, Ruby, C#, Java SDKs",
  ],
  ["OpenTelemetry", "Native (+export/import)", "Ingest-only"],
  ["Trace comparison", "\u2705", "\u2705"],
  ["Session view (multi-turn)", "\u2705", "\u2705"],
  [
    "Production SDK",
    "mlflow-tracing (lightweight)",
    "Lightweight SDK available",
  ],
  [
    "Data access",
    "SQL over Delta Tables / user DB",
    "Proprietary query language over Brainstore",
  ],
  [
    "Cost tracking",
    "\u2705 Token usage + cost calculation",
    "\u2705 Token + estimated cost",
  ],
];

const evalTable: [string, string, string][] = [
  ["Feature", "MLflow", "Braintrust"],
  ["Built-in metrics", "60+ (5 third-party libraries)", "AutoEvals only"],
  [
    "Third-party integration",
    "RAGAS, DeepEval, Phoenix, TruLens, Guardrails AI",
    "\u274C",
  ],
  ["Multi-turn eval", "Native + auto-simulation", "\u274C"],
  ["Metric versioning", "\u2705", "\u274C"],
  ["Judge alignment", "SIMBA, MemAlign, Custom", "\u274C"],
  ["CI/CD", "SDK-based", "GitHub Action with PR gating"],
];

const gatewayTable: [string, string, string][] = [
  ["Feature", "MLflow", "Braintrust"],
  ["Multi-provider routing", "\u2705", "\u2705"],
  ["Caching", "\u274C", "\u2705"],
  ["Rate limiting", "\u2705", "\u274C"],
  ["Fallbacks", "\u2705", "\u274C"],
  ["Budget alerts", "\u2705", "\u274C"],
  ["Guardrails", "\u2705", "\u274C"],
  ["A/B testing", "\u2705", "\u274C"],
  ["Credential management", "\u2705", "\u2705"],
];

export default function BraintrustAlternative() {
  return (
    <>
      <Head>
        <title>
          Open Source Braintrust Alternative? Braintrust vs MLflow | MLflow
        </title>
        <meta
          name="description"
          content="Compare MLflow and Braintrust for LLM observability, tracing, evaluation, and AI engineering. Learn why teams choose MLflow for true open source, simpler self-hosting, and production-grade evaluation."
        />
        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300;1,400;1,500&family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&display=swap');

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
            margin: 48px 0 16px 0 !important;
            line-height: 1.0 !important;
            letter-spacing: -0.03em !important;
          }
          .article-container .subtitle {
            font-family: 'DM Sans', sans-serif;
            font-size: 18px;
            color: #505050;
            line-height: 1.6;
            margin: 0 0 48px 0;
          }
          .article-container h2 {
            font-family: 'DM Sans', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            color: #1a1a1a;
            margin: 64px 0 24px 0;
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
          .article-container ul {
            list-style-type: disc;
            margin: 12px 0 24px 0;
            padding-left: 24px;
            list-style-position: outside;
          }
          .article-container li {
            font-family: 'DM Sans', sans-serif;
            font-size: 16px;
            color: #3d3d3d;
            line-height: 1.7;
            margin-bottom: 8px;
            padding-left: 4px;
          }

          /* Screenshot images */
          .screenshot-wrap {
            margin: 24px 0 32px 0;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #e5e7eb;
            box-shadow: 0 0 0 1px rgba(50, 50, 93, 0.05), 0 4px 20px rgba(50, 50, 93, 0.1);
          }
          .screenshot-wrap img {
            display: block;
            width: 100%;
            height: auto;
          }

          /* TL;DR cards */
          .tldr-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 24px;
            margin: 24px 0 56px 0;
          }
          .tldr-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            padding: 32px;
            box-shadow: 0 0 0 1px rgba(50, 50, 93, 0.05), 0 0 14px 5px rgba(50, 50, 93, 0.08), 0 0 10px 3px rgba(0, 0, 0, 0.05);
          }
          .tldr-card.highlight {
            border-color: #0194e2;
          }
          .tldr-card h3 {
            margin-top: 0 !important;
            margin-bottom: 16px !important;
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            gap: 10px;
          }
          .tldr-logo {
            width: 28px;
            height: 28px;
            object-fit: contain;
            flex-shrink: 0;
          }
          .article-container .tldr-card ul {
            padding-left: 28px !important;
          }
          .tldr-card li {
            font-size: 14px;
            margin-bottom: 6px;
          }

          /* Comparison tables */
          .comparison-table-wrap {
            width: 100%;
            overflow-x: auto;
            margin: 16px 0 40px 0;
          }
          .comparison-table {
            width: 100%;
            min-width: 100%;
            table-layout: fixed;
            border-collapse: collapse;
            font-family: 'DM Sans', sans-serif;
            font-size: 14px;
          }
          .comparison-table thead th {
            text-align: left;
            padding: 12px 16px;
            border-bottom: 2px solid #e5e7eb;
            color: #1a1a1a;
            font-weight: 600;
            background: #f9fafb;
          }
          .comparison-table tbody td {
            padding: 12px 16px;
            border-bottom: 1px solid #f0f0f0;
            color: #505050;
            vertical-align: top;
          }
          .comparison-table tbody td.feature-cell {
            color: #1a1a1a;
            font-weight: 500;
          }
          .comparison-table tbody tr:hover {
            background: #f9fafb;
          }

          /* Code tabs */
          .code-tabs {
            margin: 24px 0 40px 0;
            border-radius: 8px;
            overflow: hidden;
            border: none;
            background: #111318;
          }
          .code-tabs-header {
            display: flex;
            align-items: center;
            gap: 0;
            background: #1e1e2e;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding: 0 12px;
          }
          .code-tab {
            font-family: 'DM Sans', sans-serif;
            font-size: 13px;
            font-weight: 500;
            padding: 10px 16px;
            background: transparent;
            border: none;
            color: rgba(255,255,255,0.5);
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s ease;
          }
          .code-tab:hover {
            color: rgba(255,255,255,0.8);
          }
          .code-tab.active {
            color: #ffffff;
            border-bottom-color: #0194e2;
          }
          .code-side-by-side {
            display: grid;
            grid-template-columns: 1fr 1fr;
          }
          .code-side {
            display: flex;
            flex-direction: column;
            min-width: 0;
          }
          .code-side:first-child {
            border-right: 1px solid rgba(255,255,255,0.1);
          }
          .code-side-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 16px;
            background: #111318;
            border-bottom: 1px solid rgba(255,255,255,0.08);
          }
          .code-side-label {
            display: flex;
            align-items: center;
          }
          .code-side-label img {
            display: block;
          }
          .code-logo-inline {
            display: flex;
            align-items: center;
            gap: 6px;
            font-family: 'DM Sans', sans-serif;
            font-size: 13px;
            font-weight: 600;
            color: #ffffff;
          }
          .code-logo-icon {
            width: 16px;
            height: 16px;
            object-fit: contain;
          }
          .code-tabs-pre {
            margin: 0 !important;
            padding: 16px 16px !important;
            font-family: 'DM Mono', monospace !important;
            font-size: 12px !important;
            line-height: 1.6 !important;
            overflow-x: auto;
            flex: 1;
          }
          @media (max-width: 768px) {
            .code-side-by-side {
              grid-template-columns: 1fr;
            }
            .code-side:first-child {
              border-right: none;
              border-bottom: 1px solid rgba(255,255,255,0.1);
            }
          }

          /* Eval checklist */
          .eval-checklist {
            width: 100%;
            min-width: 100%;
            table-layout: fixed;
            border-collapse: collapse;
            font-family: 'DM Sans', sans-serif;
            font-size: 15px;
            margin: 24px 0 40px 0;
          }
          .eval-checklist thead th {
            text-align: left;
            padding: 12px 16px;
            border-bottom: 2px solid #e5e7eb;
            color: #1a1a1a;
            font-weight: 600;
            background: #f9fafb;
          }
          .eval-checklist thead th:first-child {
            width: 40%;
          }
          .eval-checklist thead th:not(:first-child) {
            text-align: center;
            width: 30%;
          }
          .eval-checklist tbody td {
            padding: 12px 16px;
            border-bottom: 1px solid #f0f0f0;
            color: #3d3d3d;
          }
          .eval-checklist tbody td:not(:first-child) {
            text-align: center;
            font-size: 18px;
          }
          .eval-checklist tbody tr:hover {
            background: #f9fafb;
          }

          /* Related resources */
          .related-resources ul {
            list-style-type: none;
            padding-left: 0;
          }
          .related-resources li {
            padding-left: 0;
            margin-bottom: 12px;
          }
          .related-resources li a {
            font-weight: 500;
          }

          /* Sidebar TOC */
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
          @media (max-width: 1400px) {
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
              margin-bottom: 12px !important;
            }
            .article-container h2 {
              font-size: 28px;
              margin: 40px 0 20px 0;
            }
            .tldr-grid {
              grid-template-columns: 1fr;
            }
          }
        `}</style>
      </Head>

      <div className="article-page">
        <Header />

        <div className="article-container">
          <h1>Open Source Braintrust Alternative? Braintrust vs MLflow</h1>
          <p className="subtitle">
            Braintrust and MLflow are platforms that help teams ship
            production-grade AI agents. Teams need tracing, evaluation, prompt
            management and optimization, and governance. In this article, we
            compare Braintrust's SaaS-first approach with MLflow's open source
            AI engineering platform and help you decide which is the right fit.
          </p>

          {/* What is Braintrust? */}
          <h2 id="what-is-braintrust">What is Braintrust?</h2>
          <div className="screenshot-wrap">
            <img
              src={BraintrustEvalUI}
              alt="Braintrust evaluation UI showing traces and scoring"
            />
          </div>
          <p>
            <strong>
              <Link to="https://www.braintrust.dev/">Braintrust</Link>
            </strong>{" "}
            is a proprietary AI observability and evaluation platform for
            monitoring LLM applications in production. Its core capabilities
            include tracing, LLM-as-a-judge evaluation, a prompt playground,
            and an AI assistant called Loop that generates datasets, scorers,
            and optimized prompts from natural language. Braintrust stores trace
            data in Brainstore, a purpose-built database for AI observability
            workloads. The platform offers SDKs for Python, TypeScript, Go,
            Ruby, C#, and Java.
          </p>

          {/* What is MLflow? */}
          <h2 id="what-is-mlflow">What is MLflow?</h2>
          <div className="screenshot-wrap">
            <img
              src={MlflowTracingUI}
              alt="MLflow tracing UI showing a trace with tool calls, messages, and assessments"
            />
          </div>
          <p>
            <strong>
              <Link to="/">MLflow</Link>
            </strong>{" "}
            is an open source AI engineering platform for agents, LLMs, and
            models that enables teams to debug, evaluate, monitor, and optimize
            production-quality AI applications while controlling costs and
            managing access to models and data. MLflow is 100% open source under
            the Apache 2.0 license and governed by the Linux Foundation. With
            50+ million monthly downloads and 20K+ GitHub stars, thousands of
            organizations rely on MLflow to ship AI to production. MLflow's
            feature set includes production-grade tracing, evaluation, prompt
            management and optimization, and an AI Gateway.
          </p>

          {/* Quick Comparison */}
          <h2 id="quick-comparison">Quick Comparison</h2>
          <div className="tldr-grid">
            <div className="tldr-card highlight">
              <h3>
                <img src={MlflowLogo} alt="MLflow" className="tldr-logo" />
                Choose MLflow if you...
              </h3>
              <ul>
                <li>
                  Care about avoiding <strong>vendor lock-in</strong> with a
                  fully open source platform.
                </li>
                <li>
                  Want <strong>simple, flexible self-hosting</strong> with
                  minimal operational overhead.
                </li>
                <li>
                  Need <strong>production-grade evaluation</strong> with 70+
                  metrics and multi-turn agent support.
                </li>
                <li>
                  Need research-backed{" "}
                  <strong>prompt optimization</strong> (GEPA, MemAlign).
                </li>
                <li>
                  Want a unified solution for{" "}
                  <strong>managing LLM access</strong> via an{" "}
                  <strong>AI Gateway</strong>.
                </li>
              </ul>
            </div>
            <div className="tldr-card">
              <h3>Choose Braintrust if you...</h3>
              <ul>
                <li>
                  Comfortable with storing trace data in a{" "}
                  <strong>proprietary vendor</strong>.
                </li>
                <li>
                  Want a simple <strong>prompt playground</strong> for
                  prototyping.
                </li>
                <li>
                  Need turnkey <strong>CI/CD integration</strong> via a
                  dedicated GitHub Action.
                </li>
                <li>
                  Want native SDK for{" "}
                  <strong>Ruby, C#, and Go</strong>.
                </li>
              </ul>
            </div>
          </div>

          {/* Open Source & Governance */}
          <h2 id="open-source-governance">Open Source &amp; Governance</h2>
          <p>
            <strong>Braintrust</strong> is a proprietary, closed-source
            platform. The core platform is commercial software with certain
            features gated behind paid tiers. Self-hosting uses a hybrid model
            where the data plane runs in your infrastructure but the control
            plane (UI, authentication, metadata) remains hosted by Braintrust.
          </p>
          <p>
            <strong>MLflow</strong> is an open source project under Apache 2.0,
            governed by the{" "}
            <Link to="https://www.linuxfoundation.org/">
              Linux Foundation
            </Link>
            . MLflow's core capabilities — tracing, evaluation, prompt
            management, model registry, and the AI Gateway — are fully available
            in the open source release with no gated tiers or feature flags.
          </p>

          {/* Self-Hosting & Architecture */}
          <h2 id="self-hosting-architecture">
            Self-Hosting &amp; Architecture
          </h2>
          <p>
            <strong>Braintrust</strong>'s self-hosting is available only for
            enterprise plans and uses a hybrid architecture. You deploy the data
            plane (API, PostgreSQL, Redis, S3, and Brainstore) in your own cloud
            via Terraform, while Braintrust hosts the control plane. This means
            a dependency on Braintrust's cloud persists even in self-hosted
            deployments.
          </p>
          <p>
            <strong>MLflow</strong> uses a minimal server + database + object
            storage architecture. Teams can plug in PostgreSQL, MySQL, SQLite,
            or any supported DB, paired with S3, GCS, Azure Blob, or local
            storage. Most deployments take minutes with familiar infrastructure.
          </p>
          <ComparisonTable rows={selfHostingTable} />

          {/* Tracing & Observability */}
          <h2 id="tracing-observability">Tracing &amp; Observability</h2>
          <p>
            Both platforms provide core tracing for LLM applications with
            dashboards and cost tracking.
          </p>
          <p>
            <strong>Braintrust</strong> instruments via native SDK wrappers and
            its gateway. Tracing can be enabled by setting a header on gateway
            requests or by wrapping LLM clients with the Braintrust SDK. Native
            SDKs are available for Python, TypeScript, Go, Ruby, C#, and Java.
          </p>
          <p>
            <strong>MLflow</strong> auto-instruments 60+ frameworks with a{" "}
            <strong>one-line unified</strong> <code>autolog()</code> API,
            including OpenAI, LangGraph, DSPy, Anthropic, LangChain, Pydantic
            AI, CrewAI, and many more. MLflow uses the native OpenTelemetry data
            model (Trace + Span + Events) and supports bidirectional OTel
            (export and ingest) while Braintrust only ingests OTel spans into
            its proprietary store.
          </p>

          <div className="code-tabs">
            <div className="code-tabs-header">
              <button className="code-tab active">Tracing</button>
            </div>
            <div className="code-side-by-side">
              <CodeBlock
                code={`import mlflow

mlflow.langchain.autolog()
# All chains, agents, retrievers, and tool calls
# traced automatically`}
                label="MLflow"
              />
              <CodeBlock
                code={`import braintrust
from openai import OpenAI

logger = braintrust.init_logger(project="My Project")
client = braintrust.wrap_openai(OpenAI())

@braintrust.traced
def answer_question(question):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content

answer_question("What is MLflow?")`}
                label="Braintrust"
              />
            </div>
          </div>

          <ComparisonTable rows={tracingTable} />

          <p>
            Braintrust has broader native SDK language coverage across six
            languages. MLflow supports Python, JS/TS, and Java natively, plus R
            for experiment tracking, and covers additional languages (Go, Rust,
            etc.) via OpenTelemetry ingestion.
          </p>

          {/* Evaluation */}
          <h2 id="evaluation">Evaluation</h2>
          <p>
            Evaluation is where the gap between MLflow and Braintrust is most
            pronounced.
          </p>
          <div className="screenshot-wrap">
            <img
              src={BraintrustExperiment}
              alt="Braintrust experiment UI showing evaluation results and scoring"
            />
          </div>
          <div className="screenshot-wrap">
            <img
              src={MlflowEvalUI}
              alt="MLflow evaluation UI showing scorers, results, and detailed assessment views"
            />
          </div>
          <p>
            <strong>Metric ecosystem.</strong> MLflow integrates natively with
            five third-party evaluation libraries — RAGAS, DeepEval, Phoenix,
            TruLens, and Guardrails AI — providing access to 60+ built-in and
            community metrics. Braintrust supports only its own AutoEvals
            library.
          </p>
          <p>
            <strong>Multi-turn agent evaluation.</strong> MLflow evaluates
            multi-turn conversations natively and supports automated
            conversation simulation. Braintrust requires assembling chat history
            into datasets with no automated conversation simulation.
          </p>
          <p>
            <strong>Judge alignment.</strong> MLflow provides multiple judge
            alignment optimizers. SIMBA (the default) uses DSPy's Simplified
            Multi-Bootstrap Aggregation to iteratively refine judge instructions
            from human feedback, achieving 30–50% reduction in evaluation
            errors. MemAlign uses a lightweight dual-memory system that adapts
            in seconds with fewer than 50 examples — up to 100× faster than
            SIMBA. Custom optimizers are also supported via a pluggable
            interface. Braintrust has no equivalent.
          </p>
          <p>
            <strong>GitHub Action for CI/CD with PR comments.</strong>{" "}
            Braintrust has dedicated GitHub Action for CI/CD quality gates.
          </p>
          <ComparisonTable rows={evalTable} />

          {/* Prompt Management & Optimization */}
          <h2 id="prompt-management">
            Prompt Management &amp; Optimization
          </h2>
          <p>
            Both platforms support prompt versioning. Braintrust's playground is
            more mature for interactive prompt iteration. PMs and domain experts
            can edit prompts, swap models, compare outputs, and run evals — all
            in the browser, no code required.
          </p>
          <p>
            For systematic optimization, <strong>MLflow</strong> ships
            research-backed algorithms:
          </p>
          <ul>
            <li>
              <strong>GEPA</strong> — Iteratively refines prompts using
              LLM-driven reflection. Supports multi-prompt agent optimization.
            </li>
            <li>
              <strong>MetaPrompting</strong> — Restructures prompts in zero-shot
              or few-shot mode.
            </li>
          </ul>
          <div className="code-tabs">
            <div className="code-tabs-header">
              <button className="code-tab active">Prompt Optimization</button>
            </div>
            <div style={{ padding: "0" }}>
              <div className="code-side" style={{ borderRight: "none" }}>
                <div className="code-side-header">
                  <span className="code-side-label">
                    <MlflowCodeLabel />
                  </span>
                </div>
                <Highlight
                  theme={customNightOwl}
                  code={`from mlflow.genai.optimize import GepaPromptOptimizer
from mlflow.genai.scorers import Correctness

result = mlflow.genai.optimize_prompts(
    predict_fn=run_agent,
    train_data=dataset,
    prompt_uris=["prompts:/my-prompt@latest"],
    optimizer=GepaPromptOptimizer(
        reflection_model="openai:/gpt-5",
        max_metric_calls=300
    ),
    scorers=[Correctness()],
)`}
                  language="python"
                >
                  {({ tokens, getLineProps, getTokenProps }) => (
                    <pre
                      className="code-tabs-pre"
                      style={{ background: CODE_BG }}
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
          </div>
          <p>
            Braintrust's Loop takes a natural-language approach that's more
            accessible enabling non-technical teams to iterate on prompts but
            has no published benchmarks against optimization baselines.
          </p>

          {/* AI Gateway */}
          <h2 id="ai-gateway">AI Gateway</h2>
          <p>
            <strong>Braintrust</strong> offers a gateway (currently in beta) for
            routing requests to any supported provider with automatic caching,
            cross-SDK compatibility, and observability. The gateway does not
            currently include rate limiting, budget controls, fallbacks, or
            guardrails.
          </p>
          <p>
            <strong>MLflow</strong> provides a full{" "}
            <Link to={`${MLFLOW_GENAI_DOCS_URL}gateway/`}>AI Gateway</Link>{" "}
            with governance built in: rate limiting, fallbacks, budget alerts,
            credential management, guardrails, and A/B testing. Teams can route
            requests across providers — OpenAI, Anthropic, Bedrock, Azure
            OpenAI, Gemini, and more — while enforcing cost controls and usage
            policies without changing application code.
          </p>
          <div className="screenshot-wrap">
            <img
              src={MlflowGatewayUI}
              alt="MLflow AI Gateway UI showing token usage, cost tracking, and endpoint management"
            />
          </div>
          <ComparisonTable rows={gatewayTable} />

          {/* Fine-Tuning & Reinforcement Learning */}
          <h2 id="fine-tuning-rl">
            Fine-Tuning &amp; Reinforcement Learning
          </h2>
          <p>
            For teams that need to go beyond prompt optimization to model
            training, the platforms diverge completely.
          </p>
          <p>
            <strong>Braintrust</strong> is focused on LLM observability and
            evaluation and does not provide capabilities for fine-tuning or
            reinforcement learning. Braintrust datasets can be exported for use
            with external fine-tuning tools, but teams must bring a separate
            platform for model training workflows.
          </p>
          <p>
            <strong>MLflow</strong> covers the full AI development lifecycle,
            including fine-tuning and RL. MLflow integrates with leading
            training libraries — <strong>Transformers</strong>,{" "}
            <strong>PEFT</strong>, <strong>Unsloth</strong>, and{" "}
            <strong>TRL</strong> — to track training runs, log model artifacts,
            and evaluate fine-tuned models. Teams can manage their entire
            workflow from LLM tracing and evaluation through model fine-tuning
            and deployment in a single platform.
          </p>

          {/* Summary */}
          <h2 id="summary">Summary</h2>
          <p>
            <strong>Braintrust</strong> is a capable evaluation and
            observability SaaS with a polished playground, broad SDK support,
            and convenient CI/CD integration. It fits teams that want a managed
            experience and primarily need tracing, evals, and prompt iteration.
          </p>
          <p>
            <strong>MLflow</strong> is a complete, open source AI engineering
            platform — self-hostable with zero dependencies, deeper evaluation
            capabilities, research-backed prompt optimization, a full AI
            Gateway, and end-to-end lifecycle coverage from tracing through
            fine-tuning. For teams that need vendor independence, cost
            predictability, and room to grow, MLflow is the stronger technical
            foundation.
          </p>

          {/* Sources & Further Reading */}
          <h2 id="sources">Sources &amp; Further Reading</h2>
          <div className="related-resources">
            <h3>MLflow</h3>
            <ul>
              <li>
                <Link to="/">MLflow Official Site</Link>
              </li>
              <li>
                <Link to={MLFLOW_GENAI_DOCS_URL}>
                  MLflow GenAI Documentation
                </Link>
              </li>
              <li>
                <Link to="https://github.com/mlflow/mlflow">
                  MLflow GitHub
                </Link>
              </li>
              <li>
                <Link
                  to={`${MLFLOW_GENAI_DOCS_URL}prompt-registry/optimize-prompts/`}
                >
                  MLflow Prompt Optimization
                </Link>
              </li>
            </ul>
            <h3>Braintrust</h3>
            <ul>
              <li>
                <Link to="https://www.braintrust.dev/">
                  Braintrust Official Site
                </Link>
              </li>
              <li>
                <Link to="https://www.braintrust.dev/docs/guides/self-hosting">
                  Braintrust Self-Hosting Guide
                </Link>
              </li>
            </ul>
          </div>
        </div>

        <ArticleSidebar />
      </div>
    </>
  );
}
