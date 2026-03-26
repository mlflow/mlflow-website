import { useState } from "react";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../components/Header/Header";
import { ArticleSidebar } from "../components/ArticleSidebar/ArticleSidebar";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import { CopyButton } from "../components/CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";
import MlflowTracingUI from "@site/static/img/langsmith-alternative/mlflow-tracing-ui.png";
import LangSmithTracingUI from "@site/static/img/langsmith-alternative/langsmith-tracing-ui.png";
import MlflowLogo from "@site/static/img/langsmith-alternative/mlflow-logo.png";
import LangSmithLogoImg from "@site/static/img/langsmith-alternative/langsmith-logo.png";
import MlflowEvalUI from "@site/static/img/langsmith-alternative/mlflow-eval-ui.png";
import LangSmithEvalUI from "@site/static/img/langsmith-alternative/langsmith-eval-ui.png";
import MlflowGatewayUI from "@site/static/img/langsmith-alternative/mlflow-gateway-ui.png";

const tracingExamples: { label: string; mlflow: string; langsmith: string }[] =
  [
    {
      label: "LangGraph",
      mlflow: `import mlflow

mlflow.langgraph.autolog()

# That's it — every node, edge, and tool call
# is traced automatically.`,
      langsmith: `import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "<your-api-key>"

# Zero-code for LangChain/LangGraph only.
# Non-LangChain code requires @traceable
# decorator on every function.`,
    },
    {
      label: "OpenAI",
      mlflow: `import mlflow

mlflow.openai.autolog()

# That's it — all OpenAI calls are
# traced automatically.`,
      langsmith: `from langsmith.wrappers import wrap_openai
import openai

# Must use wrapped client instead of
# the official openai package
client = wrap_openai(openai.OpenAI())
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hi"}],
)`,
    },
    {
      label: "DSPy",
      mlflow: `import mlflow

mlflow.dspy.autolog()

# That's it — every DSPy module call
# is traced automatically.`,
      langsmith: `# DSPy is not natively supported.
# Requires manual instrumentation with
# @traceable decorator on each function,
# or third-party OpenTelemetry instrumentor.
from langsmith import traceable

@traceable
def my_dspy_pipeline(query):
    # Must wrap every function manually
    ...
`,
    },
  ];

const selfHostingTable: [string, string, string][] = [
  ["Feature", "MLflow", "LangSmith"],
  ["Availability", "All users (open source)", "Enterprise plan only"],
  [
    "Architecture",
    "Server + DB + storage",
    "Kubernetes-based, multi-service deployment",
  ],
  [
    "Database Choices",
    "PostgreSQL, MySQL, MSSQL, SQLite, and more",
    "Vendor-specified",
  ],
  [
    "Storage Choices",
    "S3, R2, GCS, Azure Blob, HDFS, local",
    "Vendor-specified",
  ],
  [
    "Operational Complexity",
    "Minimal with familiar tools",
    "Requires Enterprise contract and vendor support",
  ],
];

const evalFeatures: {
  feature: string;
  mlflow: boolean | string;
  langsmith: boolean | string;
}[] = [
  { feature: "Built-in LLM Judges", mlflow: true, langsmith: true },
  { feature: "Custom Metrics", mlflow: true, langsmith: true },
  {
    feature: "Judge Alignment & Optimization",
    mlflow: true,
    langsmith: false,
  },
  { feature: "Versioning LLM Judges", mlflow: true, langsmith: false },
  {
    feature: "Multi-Turn Conversation Evaluation",
    mlflow: true,
    langsmith: false,
  },
  { feature: "Conversation Simulation", mlflow: true, langsmith: false },
  { feature: "Visualization & Comparison", mlflow: true, langsmith: true },
  { feature: "Prompt Optimization", mlflow: true, langsmith: false },
  {
    feature: "Integrated Libraries",
    mlflow: "RAGAS, TruLens, Phoenix",
    langsmith: "RAGAS, DeepEval",
  },
];

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

function LangSmithCodeLabel() {
  return (
    <span className="code-logo-inline">
      <img src={LangSmithLogoImg} alt="LangSmith" className="code-logo-icon" />
      LangSmith
    </span>
  );
}

function CodeBlock({ code, label }: { code: string; label: string }) {
  return (
    <div className="code-side">
      <div className="code-side-header">
        <span className="code-side-label">
          {label === "MLflow" ? <MlflowCodeLabel /> : <LangSmithCodeLabel />}
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

function CodeTabs({
  tabs,
}: {
  tabs: { label: string; mlflow: string; langsmith: string }[];
}) {
  const [active, setActive] = useState(0);
  return (
    <div className="code-tabs">
      <div className="code-tabs-header">
        {tabs.map((tab, i) => (
          <button
            key={i}
            className={`code-tab ${i === active ? "active" : ""}`}
            onClick={() => setActive(i)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="code-side-by-side">
        <CodeBlock code={tabs[active].mlflow} label="MLflow" />
        <CodeBlock code={tabs[active].langsmith} label="LangSmith" />
      </div>
    </div>
  );
}

export default function LangSmithAlternative() {
  return (
    <>
      <Head>
        <title>
          Open Source LangSmith Alternative? LangSmith vs MLflow | MLflow
        </title>
        <meta
          name="description"
          content="Compare MLflow and LangSmith for LLM observability, tracing, evaluation, and agent lifecycle management. Learn why teams choose MLflow for enterprise governance, framework-neutral tracing, and automated prompt optimization."
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
          <h1>Open Source LangSmith Alternative? LangSmith vs MLflow</h1>
          <p className="subtitle">
            LangSmith and MLflow both help teams build and monitor production AI
            agents. LangSmith is LangChain&apos;s commercial observability
            platform with native LangGraph integration. MLflow is an open source
            AI engineering platform that provides enterprise governance,
            end-to-end agent lifecycle management, and automated prompt
            optimization. In this article, we compare both platforms and help
            you decide which is the right fit.
          </p>

          {/* What is LangSmith? */}
          <h2 id="what-is-langsmith">What is LangSmith?</h2>
          <div className="screenshot-wrap">
            <img
              src={LangSmithTracingUI}
              alt="LangSmith tracing UI showing traces, spans, and trace detail view"
            />
          </div>
          <p>
            <strong>
              <Link to="https://www.langchain.com/langsmith">LangSmith</Link>
            </strong>{" "}
            is a commercial platform by LangChain Inc. for building, monitoring,
            and evaluating LLM applications. It is built by the same team as
            LangChain and LangGraph, offering deep native integration with those
            frameworks. Key capabilities include tracing and observability,
            evaluation with managed LLM judges, prompt engineering via LangChain
            Hub, agent deployment, built-in production alerting (webhooks and
            PagerDuty), and visual development tools including LangSmith Studio
            for no-code agent building. LangSmith also offers conversation clustering that
            automatically surfaces usage patterns from production traffic. It
            provides SDKs for Python, TypeScript, Go, and Java, and is available
            as a cloud-hosted SaaS with self-hosting available only on the
            Enterprise plan.
          </p>

          {/* What is MLflow? */}
          <h2 id="what-is-mlflow">What is MLflow?</h2>
          <div className="screenshot-wrap">
            <img
              src={MlflowTracingUI}
              alt="MLflow tracing UI showing a LangGraph agent trace with tool calls, messages, and assessments"
            />
          </div>
          <p>
            <strong>
              <Link to="/">MLflow</Link>
            </strong>{" "}
            is an open source AI engineering platform for agents, LLMs, and
            models that enables teams of all sizes to debug, evaluate, monitor,
            and optimize production-quality AI applications while controlling
            costs and managing access to models and data. MLflow provides
            one-line integration with 60+ frameworks, enterprise-grade
            governance through Unity Catalog, and research-backed automated
            prompt optimization. With over 30 million monthly downloads and
            adoption by 60%+ of the Fortune 500, thousands of organizations rely
            on MLflow each day to ship AI to production with confidence.
          </p>

          {/* TL;DR */}
          <h2 id="quick-comparison">Quick Comparison</h2>
          <div className="tldr-grid">
            <div className="tldr-card highlight">
              <h3>
                <img src={MlflowLogo} alt="MLflow" className="tldr-logo" />
                Choose MLflow if you...
              </h3>
              <ul>
                <li>
                  Need <strong>enterprise-grade governance</strong> with traces
                  and AI assets co-located in Unity Catalog
                </li>
                <li>
                  Want an <strong>end-to-end agent lifecycle</strong> &mdash;
                  tracing, evaluation, prompt optimization, and AI Gateway in
                  one governed platform
                </li>
                <li>
                  Need <strong>framework-neutral</strong> observability with
                  60+ integrations, not locked to one ecosystem
                </li>
                <li>
                  Care about <strong>open source</strong> (Apache 2.0, Linux
                  Foundation) with near-zero trace costs
                </li>
              </ul>
            </div>
            <div className="tldr-card">
              <h3>
                <img
                  src={LangSmithLogoImg}
                  alt="LangSmith"
                  className="tldr-logo"
                />
                Choose LangSmith if you...
              </h3>
              <ul>
                <li>
                  Are building primarily on <strong>LangChain/LangGraph</strong>{" "}
                  and want native, deeply-tested integration
                </li>
                <li>
                  Want a <strong>visual no-code builder</strong> (Studio) for
                  rapid experimentation and POCs
                </li>
                <li>
                  Need <strong>built-in alerting</strong> with webhooks and
                  PagerDuty native integration
                </li>
                <li>
                  Want <strong>conversation clustering</strong> to automatically
                  surface production usage patterns
                </li>
              </ul>
            </div>
          </div>

          {/* Open Source & Governance */}
          <h2 id="open-source-governance">
            Open Source, Governance &amp; Cost
          </h2>
          <p>
            <strong>LangSmith</strong> is a{" "}
            <strong>closed-source proprietary product</strong> by LangChain Inc.
            While LangChain (the framework) is open source under MIT, the
            LangSmith platform &mdash; its UI, backend, and hosted
            infrastructure &mdash; is closed-source and requires a paid
            subscription for production use. Critical enterprise features
            including SSO, RBAC, audit logs, and self-hosting are{" "}
            <strong>gated behind the Enterprise tier</strong>. Traces are stored
            in LangSmith&apos;s own infrastructure, separate from your broader
            data stack, making large-scale analytics or joining with other
            business data more cumbersome. LangSmith&apos;s per-trace pricing
            can scale from $2K to over $200K/year with seat-based licensing on
            top.
          </p>
          <p>
            <strong>MLflow</strong> is a fully open source project{" "}
            <strong>
              backed by the{" "}
              <Link href="https://www.linuxfoundation.org/">
                Linux Foundation
              </Link>
            </strong>
            , licensed under Apache 2.0 with full feature parity between its
            open source release and managed offerings. Prompts, traces, and
            monitoring live in the <strong>same governed data space</strong> as
            your other assets, with Unity Catalog applying consistent controls
            out of the box. MLflow has <strong>near-zero trace costs</strong>{" "}
            &mdash; no per-trace fees, no per-seat fees, and no feature gating.
            With adoption by 60%+ of the Fortune 500, MLflow is one of the most
            widely deployed AI platforms in the enterprise.
          </p>

          {/* Self-Hosting & Architecture */}
          <h2 id="self-hosting-architecture">
            Self-Hosting &amp; Architecture
          </h2>
          <p>
            <strong>LangSmith</strong> is a cloud-first SaaS by default.
            Self-hosting and BYOC (bring-your-own-cloud) options{" "}
            <strong>require an Enterprise contract</strong> plus Kubernetes
            infrastructure. There is no self-hosting option for Developer or
            Plus tier users &mdash; teams on these tiers must send all trace
            data to LangChain&apos;s cloud.
          </p>
          <p>
            <strong>MLflow</strong> is designed for{" "}
            <strong>simplicity and flexibility</strong>. It adopts a simple
            server + DB + storage architecture, and enables teams to use their
            own choice of database and storage solution, such as PostgreSQL,
            MySQL, AWS RDS, GCP Cloud SQL, Neon, Supabase, or even SQLite. The
            storage can be any object storage solution, such as S3, GCS, Azure
            Blob, HDFS, or even local file system.{" "}
            <strong>Most teams can deploy MLflow in minutes</strong> with
            familiar infrastructure. MLflow is also available as a managed
            service on Databricks, AWS SageMaker, Nebius, and Azure ML.
          </p>
          <ComparisonTable rows={selfHostingTable} />

          {/* Tracing & Observability */}
          <h2 id="tracing-observability">Tracing &amp; Observability</h2>
          <p>
            Both platforms provide core tracing for LLM applications with
            OpenTelemetry compatibility, operational dashboards, and cost
            tracking.
          </p>
          <p>
            <strong>LangSmith</strong>&apos;s tracing works{" "}
            <strong>seamlessly within the LangChain ecosystem</strong> &mdash;
            set an environment variable and all LangChain/LangGraph calls are
            traced automatically. For non-LangChain code, it requires the{" "}
            <code>@traceable</code> decorator or wrapper functions like{" "}
            <code>wrap_openai</code>. LangSmith supports Python, TypeScript, Go,
            and Java SDKs, but trace data is only accessible via the LangSmith
            UI or SDK APIs, with no easy way to query traces directly with SQL.
          </p>
          <p>
            <strong>MLflow</strong> provides{" "}
            <strong>one-line integration with 60+ frameworks</strong> (OpenAI,
            Anthropic, LangChain, LlamaIndex, DSPy, Pydantic AI, Vercel AI SDK,
            and more) via a unified <code>autolog()</code> API across Python,
            TypeScript, Java, and R. Traces are stored in Unity Catalog &mdash;{" "}
            <strong>
              queryable via dashboards, Genie, or custom SQL analytics
            </strong>
            , making MLflow powerful for agent analytics at scale.
          </p>
          <CodeTabs tabs={tracingExamples} />

          {/* Evaluation */}
          <h2 id="evaluation">Evaluation</h2>
          <p>
            Both platforms offer evaluation capabilities, but they differ
            significantly in depth and automation.
          </p>
          <p>
            <strong>LangSmith</strong> provides managed LLM judges, custom code
            evaluators, and dataset management with support for RAGAS and
            DeepEval. However, it lacks{" "}
            <strong>
              judge alignment with human feedback, multi-turn conversation
              evaluation, conversation simulation
            </strong>
            , and <strong>automated prompt optimization</strong> &mdash;
            capabilities that are essential for teams shipping AI agents to
            production. Evaluation is also tightly coupled to the LangChain
            ecosystem.
          </p>
          <div className="screenshot-wrap">
            <img
              src={LangSmithEvalUI}
              alt="LangSmith evaluation UI showing scoring and annotation"
            />
          </div>
          <p>
            <strong>MLflow</strong> provides production-grade evaluation backed
            by a dedicated research team. It supports built-in LLM judges with{" "}
            <strong>judge alignment and optimization</strong>, versioning of LLM
            judges, integration with leading evaluation libraries (RAGAS,
            TruLens, Phoenix), and advanced capabilities like{" "}
            <strong>multi-turn conversation evaluation</strong> with built-in
            conversation simulation. Judge costs are transparently displayed
            alongside traces. If your team needs to move beyond vibe checks to
            rigorous quality assurance, MLflow is purpose-built for it.
          </p>
          <div className="screenshot-wrap">
            <img
              src={MlflowEvalUI}
              alt="MLflow evaluation UI showing scorers, results, and detailed assessment views"
            />
          </div>
          <table className="eval-checklist">
            <thead>
              <tr>
                <th>Capability</th>
                <th>MLflow</th>
                <th>LangSmith</th>
              </tr>
            </thead>
            <tbody>
              {evalFeatures.map((row, i) => (
                <tr key={i}>
                  <td>{row.feature}</td>
                  <td>
                    {typeof row.mlflow === "string"
                      ? row.mlflow
                      : row.mlflow
                        ? "\u2705"
                        : "\u274C"}
                  </td>
                  <td>
                    {typeof row.langsmith === "string"
                      ? row.langsmith
                      : row.langsmith
                        ? "\u2705"
                        : "\u274C"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Prompt Management & Optimization */}
          <h2 id="prompt-management">Prompt Management &amp; Optimization</h2>
          <p>
            Both platforms offer prompt management with versioning, but they
            differ fundamentally in their approach to improving prompt quality.
          </p>
          <p>
            <strong>LangSmith</strong> offers{" "}
            <strong>
              <a href="https://docs.smith.langchain.com/prompt_engineering">
                LangChain Hub
              </a>
            </strong>{" "}
            for prompt management with versioning and sharing. Its Prompt
            Playground allows interactive testing against live models, and
            LangSmith Studio provides a{" "}
            <strong>visual, no-code interface</strong> for building and testing
            agents &mdash; a genuine strength for teams focused on rapid
            experimentation and POCs.
          </p>
          <p>
            <strong>MLflow</strong> supports versioning with aliases, Jinja2
            templates, structured outputs, and text/chat message formats. Beyond
            management, MLflow offers{" "}
            <strong>
              <a href="https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/">
                automated prompt optimization
              </a>
            </strong>{" "}
            &mdash; native, research-backed algorithms (GEPA, memAlign) that
            automatically improve prompts using evaluation feedback, for both
            individual prompts and end-to-end agents. No manual iteration
            required.{" "}
            <strong>This capability does not exist in LangSmith.</strong>
          </p>
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
                  code={`import mlflow
from mlflow.genai.optimize import GepaPromptOptimizer
from mlflow.genai.scorers import Correctness


# Optimize the prompt automatically
result = mlflow.genai.optimize_prompts(
    predict_fn=run_agent,
    train_data=dataset,
    prompt_uris=["prompts:/my-prompt@latest"],
    optimizer=GepaPromptOptimizer(
        reflection_model="openai:/gpt-5", max_metric_calls=300
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

          {/* AI Gateway */}
          <h2 id="ai-gateway">AI Gateway</h2>
          <p>
            As LLM applications move to production, teams face growing
            challenges around managing API keys, controlling costs, switching
            between providers, and enforcing governance policies. This is where
            an <Link to="/ai-gateway">AI Gateway</Link>, a centralized layer
            between your applications and LLM providers, has become an essential
            piece of production AI infrastructure.
          </p>
          <p>
            <strong>LangSmith</strong> is solely an agent observability and
            reliability platform. It does not offer a gateway capability or a
            model registry. Teams using LangSmith must bolt on a separate tool
            such as LiteLLM, PortKey, or build a custom gateway solution.
          </p>
          <p>
            <strong>MLflow</strong> offers a built-in{" "}
            <Link to={`${MLFLOW_GENAI_DOCS_URL}gateway/`}>AI Gateway</Link> for
            governing LLM access across your organization. It provides a
            standard endpoint that routes requests to any supported provider
            (OpenAI, Anthropic, AWS Bedrock, Azure OpenAI, Google Gemini, and
            more), with built-in{" "}
            <strong>
              rate limiting, fallbacks, usage tracking, and credential
              management
            </strong>
            . Tracing, evaluation, and gateway are integrated with Model
            Serving, Vector Search, Databricks Apps, and more &mdash; forming a
            complete end-to-end platform rather than requiring teams to stitch
            together disparate tools.
          </p>
          <div className="screenshot-wrap">
            <img
              src={MlflowGatewayUI}
              alt="MLflow AI Gateway UI showing token usage, cost tracking, and endpoint management"
            />
          </div>

          {/* Summary */}
          <h2 id="summary">Summary</h2>
          <p>
            <strong>
              LangSmith is a capable observability and evaluation platform
            </strong>{" "}
            with genuine strengths in native LangChain/LangGraph integration,
            visual agent building with Studio, built-in production alerting,
            and conversation clustering for
            production insights. However, it is a closed-source proprietary
            product, tightly coupled to the LangChain ecosystem, stores traces
            in a silo separate from your broader data stack, and lacks automated
            prompt optimization and an AI Gateway.{" "}
            <strong>Choose LangSmith</strong> if you are building primarily on
            LangGraph and want native integration with a managed SaaS for rapid
            experimentation.
          </p>
          <p>
            <strong>
              MLflow is a complete AI engineering platform for the end-to-end
              agent lifecycle.
            </strong>{" "}
            It provides framework-neutral tracing for 60+ integrations,
            production-grade evaluation with judge alignment and multi-turn
            support, automated prompt optimization, an AI Gateway for LLM
            governance, and enterprise-grade data governance through Unity
            Catalog &mdash; all open source under the Linux Foundation.{" "}
            <strong>Choose MLflow</strong> if you need a governed,
            vendor-neutral platform that covers the full agent lifecycle from
            development through production monitoring and optimization.
          </p>

          {/* Related Resources */}
          <h2>Related Resources</h2>
          <div className="related-resources">
            <ul>
              <li>
                <Link to="https://docs.smith.langchain.com/">
                  LangSmith Documentation
                </Link>
              </li>
              <li>
                <Link to="https://docs.smith.langchain.com/getting-started">
                  LangSmith Quickstart
                </Link>
              </li>
              <li>
                <Link to={MLFLOW_GENAI_DOCS_URL}>MLflow Documentation</Link>
              </li>
              <li>
                <Link to={`${MLFLOW_GENAI_DOCS_URL}tracing/quickstart/`}>
                  MLflow Tracing Quickstart
                </Link>
              </li>
              <li>
                <Link to="/langfuse-alternative">MLflow vs Langfuse</Link>
              </li>
              <li>
                <Link to="/llmops">LLMOps Guide</Link>
              </li>
            </ul>
          </div>
        </div>

        <ArticleSidebar />
      </div>
    </>
  );
}
