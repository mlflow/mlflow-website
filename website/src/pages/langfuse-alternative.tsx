import { useState } from "react";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../components/Header/Header";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import { CopyButton } from "../components/CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";
import MlflowTracingUI from "@site/static/img/langfuse-alternative/mlflow-tracing-ui.png";
import LangfuseTracingUI from "@site/static/img/langfuse-alternative/langfuse-tracing-ui.png";
import MlflowLogo from "@site/static/img/langfuse-alternative/mlflow-logo.png";
import LangfuseLogoImg from "@site/static/img/langfuse-alternative/langfuse-logo.png";

const tracingExamples: { label: string; mlflow: string; langfuse: string }[] = [
  {
    label: "LangGraph",
    mlflow: `import mlflow

mlflow.langgraph.autolog()

# That's it — every node, edge, and tool call
# is traced automatically.`,
    langfuse: `from langfuse.callback import CallbackHandler

handler = CallbackHandler()

# Must pass handler to each invocation
result = app.invoke(
    {"messages": [("user", "Plan a trip")]},
    config={"callbacks": [handler]},
)`,
  },
  {
    label: "OpenAI",
    mlflow: `import mlflow

mlflow.openai.autolog()

# That's it — all OpenAI calls are
# traced automatically.`,
    langfuse: `from langfuse.openai import openai

# Must use wrapped client instead of
# the official openai package
client = openai.OpenAI()
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
    langfuse: `# Requires installing a separate
# third-party package
# pip install openinference-instrumentation-dspy
from openinference.instrumentation.dspy import (
    DSPyInstrumentor,
)

DSPyInstrumentor().instrument()
`
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

function LangfuseCodeLabel() {
  return (
    <span className="code-logo-inline">
      <img src={LangfuseLogoImg} alt="Langfuse" className="code-logo-icon" />
      Langfuse
    </span>
  );
}

function CodeBlock({ code, label }: { code: string; label: string }) {
  return (
    <div className="code-side">
      <div className="code-side-header">
        <span className="code-side-label">
          {label === "MLflow" ? <MlflowCodeLabel /> : <LangfuseCodeLabel />}
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
  tabs: { label: string; mlflow: string; langfuse: string }[];
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
        <CodeBlock code={tabs[active].langfuse} label="Langfuse" />
      </div>
    </div>
  );
}

const selfHostingTable: [string, string, string][] = [
  ["Feature", "MLflow", "Langfuse"],
  [
    "Architecture",
    "Server + DB + storage",
    "ClickHouse + PostgreSQL + Redis + S3 + Web Server",
  ],
  ["Database Choices", "PostgreSQL, MySQL, MSSQL, SQLite, and more", "ClickHouse required"],
  ["Storage", "S3, R2, GCS, Azure Blob, HDFS, local", "S3 or GCS"],
  [
    "Operational Complexity",
    "Minimal with familiar tools",
    "ClickHouse expertise needed",
  ],
];

const evalFeatures: { feature: string; mlflow: boolean | string; langfuse: boolean | string }[] =
  [
    { feature: "Built-in LLM Judges", mlflow: true, langfuse: true },
    { feature: "Custom Metrics", mlflow: true, langfuse: true },
    { feature: "Versioning Metrics", mlflow: true, langfuse: false },
    {
      feature: "Aligning Judges with Human Feedback",
      mlflow: true,
      langfuse: false,
    },
    { feature: "Multi-Turn Evaluation", mlflow: true, langfuse: false },
    { feature: "Online Evaluation", mlflow: true, langfuse: false },
    {
      feature: "Integrated Libraries",
      mlflow: "RAGAS, DeepEval, Phoenix, TruLens, Guardrails AI",
      langfuse: "RAGAS",
    },
  ];

export default function LangfuseAlternative() {
  return (
    <>
      <Head>
        <title>
          Open Source Langfuse Alternative? MLflow vs Langfuse | MLflow
        </title>
        <meta
          name="description"
          content="Compare MLflow and Langfuse for LLM observability, tracing, and evaluation. Learn why teams choose MLflow for true open source, simpler self-hosting, and production-grade evaluation."
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
          <h1>Open Source Langfuse Alternative? Langfuse vs MLflow</h1>
          <p className="subtitle">
            Langfuse and MLflow are leading open-source tools that help
            developers monitor, debug, and optimize their LLM-powered
            applications. In this article, we'll explore the differences between the two platforms, compare their features, and help you decide which platform
            is the right fit for your team.
          </p>


          {/* What is Langfuse? */}
          <h2>What is Langfuse?</h2>
          <div className="screenshot-wrap">
            <img
              src={LangfuseTracingUI}
              alt="Langfuse tracing UI showing traces, spans, and trace detail view"
            />
          </div>
          <p>
            <strong><Link to="https://langfuse.com/">Langfuse</Link></strong> is an open-source
            LLM engineering platform that helps teams collaboratively develop,
            monitor, and debug AI applications. It provides tracing,
            prompt management, evaluation, and analytics to give full visibility
            into LLM app behavior - capturing every operation, timing, inputs,
            outputs, and metadata. Langfuse integrates with popular frameworks
            like OpenAI SDK, LangChain, and LlamaIndex, and offers both a
            cloud-hosted SaaS and a self-hosted deployment option.
          </p>

          {/* What is MLflow? */}
          <h2>What is MLflow?</h2>
          <div className="screenshot-wrap">
            <img
              src={MlflowTracingUI}
              alt="MLflow tracing UI showing a LangGraph agent trace with tool calls, messages, and assessments"
            />
          </div>
          <p>
            <strong><Link to="/">MLflow</Link></strong> is an open-source AI engineering
            platform that enables teams of all sizes to debug, evaluate,
            monitor, and optimize production-quality AI agents, LLM
            applications, and ML models while controlling costs and managing
            access to models and data. With over 30 million monthly downloads,
            thousands of organizations rely on MLflow each day to ship AI to
            production with confidence.
          </p>

          {/* TL;DR */}
          <h2>TL;DR</h2>
          <div className="tldr-grid">
            <div className="tldr-card highlight">
              <h3><img src={MlflowLogo} alt="MLflow" className="tldr-logo" />Choose MLflow if you...</h3>
              <ul>
                <li>
                  Care about avoiding <strong>vendor lock-in</strong>
                </li>
                <li>
                  Want <strong>simple, flexible self-hosting</strong> with
                  minimal operational overhead
                </li>
                <li>
                  Need <strong>production-grade evaluation</strong> and{" "}
                  <strong>prompt optimization</strong> for AI agents
                </li>
                <li>
                  Want a unified solution for <strong>managing and governing access</strong> to LLMs for your organization
                </li>
              </ul>
            </div>
            <div className="tldr-card">
              <h3><img src={LangfuseLogoImg} alt="Langfuse" className="tldr-logo" />Choose Langfuse if you...</h3>
              <ul>
                <li>
                  Already run <strong>ClickHouse-based</strong> infrastructure
                </li>
                <li>
                  Prefer a <strong>SaaS-first</strong> onboarding experience
                </li>
                <li>
                  Want a convenient tool for{" "}
                  <strong>manual prompt engineering</strong>
                </li>
                <li>
                  Use low-code tools like Flowise or LobeChat for quick <strong>prototyping</strong>
                </li>
              </ul>
            </div>
          </div>

          {/* Open Source & Governance */}
          <h2>Open Source &amp; Governance</h2>
          <p>
            <strong>Langfuse</strong> is an open source project under the MIT license and
            was <strong>acquired by <Link href="https://clickhouse.com/">ClickHouse Inc.</Link></strong> in 2025. While the project remains
            open source, its roadmap and development priorities are now shaped by
            ClickHouse Inc.'s strategy. Langfuse also gates certain
            features behind its paid cloud plans, creating a gap between the
            open source and commercial versions. The vendor lock-in concern is sometimes a barrier
            for Enterprises to adopt Langfuse.
          </p>
          <p>
            <strong>MLflow</strong> is also an open source project but <strong>backed by the{" "}
            <Link href="https://www.linuxfoundation.org/">
              Linux Foundation
            </Link></strong>
            , a non-profit vendor-neutral organization, ensuring long-term community stewardship with no single company controlling its direction.
            MLflow is licensed under{" "}
            Apache 2.0 and maintains full feature parity between
            its open-source release and managed offerings. With adoption by{" "}
            60%+ of the Fortune 500, MLflow is one of the most widely deployed AI platform in the enterprise.
          </p>

          {/* Self-Hosting & Architecture */}
          <h2>Self-Hosting &amp; Architecture</h2>
          <p>
            Both platform offer self-hosting options for teams who want to control their own data and infrastructure.
          </p>
          <p>
            <strong>Langfuse</strong> architecture is built around{" "}
            <strong>ClickHouse</strong>, giving it strong analytical query
            performance for teams already invested in the ClickHouse ecosystem.
            A full Langfuse deployment requires{" "}
            <strong>5+ services</strong>, including ClickHouse, PostgreSQL, Redis, S3,
            and the application server, which often requires a dedicated operation and
             introduces challenges for teams without ClickHouse expertise.
          </p>
          <p>
            <strong>MLflow</strong> is designed for{" "}
              <strong>simplicity and flexibility</strong>. It adopts a simple server + DB + storage architecture,
              and allowing teams to use their own choice of database and storage solution, such as PostgreSQL,
              MySQL, AWS RDS, GCP Cloud SQL, Neon, Supabase, or even SQLite. The storage can be any object storage
              solution, such as S3, GCS, Azure Blob, HDFS, or even local file system. Most teams can deploy MLflow
              in minutes with familiar infrastructure.
          </p>
          <ComparisonTable rows={selfHostingTable} />

          {/* Tracing & Observability */}
          <h2>Tracing &amp; Observability</h2>
          <p>
            Both platforms provide core tracing for LLM applications with full
            OpenTelemetry compatibility and support for Python and JS/TS SDKs.
            Both offer operational dashboard and cost tracking.
          </p>
          <p>
            <strong>Langfuse</strong>'s instrumentation <strong>varies by SDK and framework</strong>, some use a wrapper, some uses a callback handler, and others require a separate third-party package. The SDK is compatible with OpenTelemetry but exposes a different data model (Trace + Observation).
          </p>
          <p>
            <strong>MLflow</strong> auto-instruments 30+ frameworks with a <strong>one-line unified</strong> <code>autolog()</code> API, including OpenAI,
            LangGraph, DSPy, Anthropic, LangChain, Pydantic AI, CrewAI, and many
            more. MLflow uses the native OpenTelemetry data model (Trace + Span + Events).
          </p>
          <CodeTabs tabs={tracingExamples} />

          {/* Evaluation */}
          <h2>Evaluation</h2>
          <p>
            Evaluation is where the gap between MLflow and Langfuse is most
            pronounced. <strong>MLflow</strong> provides{" "} production-grade evaluation backed by a dedicated
            research team, with capabilities designed for teams shipping AI
            agents to production. <strong>Langfuse</strong> offers basic evaluation support that
            may suit simpler use cases and quick vibe checks.
          </p>
          <table className="eval-checklist">
            <thead>
              <tr>
                <th>Capability</th>
                <th>MLflow</th>
                <th>Langfuse</th>
              </tr>
            </thead>
            <tbody>
              {evalFeatures.map((row, i) => (
                <tr key={i}>
                  <td>{row.feature}</td>
                  <td>{typeof row.mlflow === "string" ? row.mlflow : row.mlflow ? "✅" : "❌"}</td>
                  <td>{typeof row.langfuse === "string" ? row.langfuse : row.langfuse ? "✅" : "❌"}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Prompt Management */}
          <h2>Prompt Management</h2>
          <p>
            Both platforms offer prompt management capabilities. While there are many common features such as versioning, tagging, lineage, caching, they
            differ in their approach to developing prompt quality.
          </p>
          <p>
            <strong>Langfuse</strong> offers an{" "}
            <strong><a href="https://langfuse.com/docs/prompt-management/features/playground">easy-to-use prompt playground</a></strong>, making it an excellent
            choice for teams focused on manual prompt engineering - casually
            iterating, testing variations, and refining prompts by hand.
          </p>
          <p>
            <strong>MLflow</strong> targets <strong>systematic prompt improvement</strong> and offers {" "}
            state-of-the-art <strong><a href="https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/">prompt optimization</a></strong> algorithms such as GEPA and MIPRO to automatically improve prompts based on evaluation results,
            for both individual prompts and end-to-end agents. MLflow is suitable for
            teams who want to go from manual prompt tweaking to a more systematic approach for developing production-grade prompts.
          </p>

          {/* AI Gateway */}
          <h2>AI Gateway</h2>
          <p>
            As LLM applications move to production, teams face growing
            challenges around managing API keys, controlling costs, switching
            between providers, and enforcing governance policies. This is where an{" "}
            <Link to="/ai-gateway">AI Gateway</Link>, a centralized layer
            between your applications and LLM providers, has become an
            essential piece of production AI infrastructure.
          </p>
          <p>
            <strong>Langfuse</strong> does not offer a gateway capability. To manage costs and model access, teams often use a different tool such as LiteLLM, PortKey, or build a custom gateway solution.
          </p>
          <p>
            <strong>MLflow</strong> offers a built-in{" "}
            <Link to={`${MLFLOW_GENAI_DOCS_URL}gateway/`}>
              AI Gateway
            </Link> for governing LLM access
            across your organization. It provides a standard endpoint
            that routes requests to any supported provider (OpenAI, Anthropic,
            AWS Bedrock, Azure OpenAI, Google Gemini, and more), with built-in{" "}
            <strong>rate limiting, fallbacks, usage tracking, and credential management</strong>.
            Teams can switch providers, add guardrails, or enforce usage policies
            without changing application code.
          </p>

          {/* Reinforcement Learning */}
          <h2>Reinforcement Learning</h2>
          <p>
            For advanced research teams, reinforcement learning from human
            feedback (RLHF) and other RL-based techniques are becoming
            increasingly important for aligning and improving LLM behavior.
            Managing these workflows requires robust experiment tracking, model
            versioning, and evaluation infrastructure.
          </p>
          <p>
            <strong>Langfuse</strong> is focused on LLM observability and does
            not provide capabilities for fine-tuning or reinforcement learning.
            Teams using Langfuse would need a separate tool to manage model
            training workflows.
          </p>
          <p>
            <strong>MLflow</strong> goes beyond LLM tracing and evaluation to
            cover the <strong>full AI development lifecycle</strong>. MLflow
            integrates with leading fine-tuning and reinforcement learning
            libraries, including <strong>Transformers</strong>,{" "}
            <strong>PEFT</strong>, <strong>Unsloth</strong>, and{" "}
            <strong>TRL</strong>, to track training runs, log model artifacts,
            and evaluate fine-tuned models. This means teams can manage their
            entire workflow from LLM applications through model fine-tuning in a
            single platform.
          </p>
          {/* Summary */}
          <h2>Summary</h2>
          <p>
            The core difference between MLflow and Langfuse lies in multiple dimensions, including{" "}
            <strong>openness, quality philosophy, and operational complexity</strong>. MLflow is a full AI engineering
            platform that covers tracing, evaluation, prompt optimization, an AI
            Gateway, fine-tuning, and RL, governed by the Linux Foundation with
            complete open-source feature parity. Langfuse focuses on LLM
            observability with a strong playground experience, but its evaluation
            capabilities are limited and hosting complexity is a challenge.
          </p>
          <p>
            <strong>Choose MLflow</strong> if you need a
            vendor-neutral platform with rich evaluation capabilities, prompt
            optimization, and coverage across the full AI lifecycle.{" "}
            <strong>Choose Langfuse</strong> if you want a lightweight
            observability tool with a playground for manual prompt
            engineering, or if you're already invested in ClickHouse
            infrastructure.
          </p>

          {/* Related Resources */}
          <h2>Related Resources</h2>
          <div className="related-resources">
            <ul>
              <li>
                <Link to="https://langfuse.com/docs">
                  Langfuse Documentation
                </Link>
              </li>
              <li>
                <Link to="https://langfuse.com/docs/get-started">
                  Langfuse Quickstart
                </Link>
              </li>
              <li>
                <Link to={MLFLOW_GENAI_DOCS_URL}>
                  MLflow Documentation
                </Link>
              </li>
              <li>
                <Link to={`${MLFLOW_GENAI_DOCS_URL}tracing/quickstart/`}>
                  MLflow Tracing Quickstart
                </Link>
              </li>
              <li>
                <Link to="/llmops">LLMOps Guide</Link>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </>
  );
}
