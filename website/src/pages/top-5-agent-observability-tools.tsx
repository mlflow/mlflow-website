import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../components/Header/Header";
import { ArticleSidebar } from "../components/ArticleSidebar/ArticleSidebar";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import { CopyButton } from "../components/CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";

/* ───────── images ───────── */
import MlflowTracingUI from "@site/static/img/langfuse-alternative/mlflow-tracing-ui.png";
import MlflowEvalUI from "@site/static/img/langfuse-alternative/mlflow-eval-ui.png";
import MlflowGatewayUI from "@site/static/img/langfuse-alternative/mlflow-gateway-ui.png";
import LangfuseTracingUI from "@site/static/img/langfuse-alternative/langfuse-tracing-ui.png";
import LangsmithTracingUI from "@site/static/img/langsmith-alternative/langsmith-tracing-ui.png";
import ArizePhoenixUI from "@site/static/img/arize-phoenix-alternative/arize-phoenix-ui.png";
import BraintrustTraceUI from "@site/static/img/top-5-observability/braintrust-trace-view.png";

/* ───────── data ───────── */

const quickComparisonRows: [string, string, string, string, string, string][] =
  [
    [
      "Capability",
      "MLflow",
      "Langfuse",
      "LangSmith",
      "Arize Phoenix",
      "Braintrust",
    ],
    ["Open Source", "✔️", "✔️", "No", "Partial (ELv2)", "No"],
    [
      "License",
      "Apache 2.0 (Linux Foundation)",
      "MIT (ClickHouse Inc.)",
      "Proprietary",
      "Elastic License 2.0 (ELv2)",
      "Proprietary",
    ],
    ["PyPI Downloads", "30M+/mo", "15M+/mo", "65M+/mo ¹", "1M+/mo", "3M+/mo"],
    [
      "Integration",
      "60+ frameworks via OpenTelemetry",
      "60+ frameworks via OpenTelemetry",
      "LangChain-native + OpenTelemetry",
      "40+ via OpenInference + OpenTelemetry",
      "50+ frameworks",
    ],
    [
      "OpenTelemetry",
      "✔️",
      "Partial (ingest)",
      "Partial (ingest)",
      "✔️",
      "Partial (ingest)",
    ],
    ["Governance\n(AI Gateway)", "✔️", "No", "No", "No", "✔️"],
    [
      "Self-Hosting",
      "Simple",
      "Complex (5+ services + ClickHouse)",
      "Enterprise-only",
      "Simple",
      "Not available",
    ],
    [
      "Production Scale",
      "✔️ (self-hosted, scales with your infra)",
      "✔️ (ClickHouse-based)",
      "✔️ (managed SaaS)",
      "Single-node OSS; managed SaaS for scale",
      "✔️ (managed SaaS)",
    ],
    [
      "Data Retention",
      "Unlimited",
      "30 days (free) to 3 years (pro)",
      "14 days (free); 400 days (paid add-on)",
      "7 days (free); 15 days (pro)",
      "14 days (starter); 30 days (pro)",
    ],
  ];

/* ───────── components ───────── */

function QuickComparisonTable({
  rows,
}: {
  rows: [string, string, string, string, string, string][];
}) {
  const [header, ...body] = rows;
  return (
    <div className="comparison-table-wrap">
      <table className="comparison-table">
        <thead>
          <tr>
            {header.map((cell, i) => (
              <th
                key={i}
                style={i === 1 ? { background: "#e0f2fe" } : undefined}
              >
                {cell}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td
                  key={j}
                  className={j === 0 ? "feature-cell" : ""}
                  style={j === 1 ? { background: "#f0f9ff" } : undefined}
                >
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

function ProsConsTable({ pros, cons }: { pros: string[]; cons: string[] }) {
  const maxLen = Math.max(pros.length, cons.length);
  return (
    <table className="pros-cons-table">
      <thead>
        <tr>
          <th>Pros</th>
          <th>Cons</th>
        </tr>
      </thead>
      <tbody>
        {Array.from({ length: maxLen }).map((_, i) => (
          <tr key={i}>
            <td>{pros[i] || ""}</td>
            <td>{cons[i] || ""}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function SingleCodeBlock({ code }: { code: string }) {
  return (
    <div className="code-tabs">
      <div className="code-tabs-header">
        <button className="code-tab active">MLflow</button>
      </div>
      <div style={{ padding: 0 }}>
        <div className="code-side" style={{ borderRight: "none" }}>
          <div className="code-side-header">
            <span className="code-side-label">
              <span className="code-logo-inline">MLflow</span>
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
      </div>
    </div>
  );
}

/* ───────── page ───────── */

export default function Top5AgentObservabilityTools() {
  return (
    <>
      <Head>
        <title>Top 5 LLM and Agent Observability Tools in 2026 | MLflow</title>
        <meta
          name="description"
          content="Compare the best agent observability tools for tracing, evaluation, and monitoring AI agents in production. See how MLflow, Langfuse, LangSmith, Arize Phoenix, and Braintrust stack up."
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
            padding-bottom: 12px;
            border-bottom: 1px solid #e0e0e0;
            line-height: 1.2;
            letter-spacing: -0.01em;
          }
          .article-container h3 {
            font-family: 'DM Sans', sans-serif;
            font-size: 1.5rem;
            font-weight: 600;
            color: #1a1a1a;
            margin: 32px 0 16px 0 !important;
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
          .article-container a strong {
            color: inherit;
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

          /* TL;DR recommendation card */
          .recommendation-card {
            background: linear-gradient(135deg, #f0f9ff 0%, #e8f4fd 100%);
            border: 2px solid #0194e2;
            border-radius: 8px;
            padding: 32px;
            margin: 32px 0 56px 0;
            box-shadow: 0 0 0 1px rgba(1, 148, 226, 0.1), 0 4px 20px rgba(1, 148, 226, 0.08);
          }
          .recommendation-card h3 {
            margin-top: 0 !important;
            margin-bottom: 12px !important;
            font-size: 1.3rem;
            color: #0072b0;
          }
          .recommendation-card p {
            margin-bottom: 0;
          }

          /* Tool cards */
          .tool-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 32px;
            margin: 24px 0 40px 0;
            box-shadow: 0 0 0 1px rgba(50, 50, 93, 0.05), 0 0 14px 5px rgba(50, 50, 93, 0.08), 0 0 10px 3px rgba(0, 0, 0, 0.05);
          }
          .tool-card.highlight {
            border-color: #0194e2;
            border-width: 2px;
          }
          .tool-card h3 {
            margin-top: 0 !important;
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
            white-space: nowrap;
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
            white-space: pre-line;
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
          .code-side {
            display: flex;
            flex-direction: column;
            min-width: 0;
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
          .code-logo-inline {
            display: flex;
            align-items: center;
            gap: 6px;
            font-family: 'DM Sans', sans-serif;
            font-size: 13px;
            font-weight: 600;
            color: #ffffff;
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

          /* Pros/Cons table */
          .pros-cons-table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'DM Sans', sans-serif;
            font-size: 15px;
            margin: 24px 0 32px 0;
          }
          .pros-cons-table thead th {
            text-align: left;
            padding: 12px 16px;
            border-bottom: 2px solid #e5e7eb;
            font-weight: 600;
            background: #f9fafb;
            width: 50%;
          }
          .pros-cons-table thead th:first-child {
            color: #15803d;
          }
          .pros-cons-table thead th:last-child {
            color: #b91c1c;
          }
          .pros-cons-table tbody td {
            padding: 10px 16px;
            border-bottom: 1px solid #f0f0f0;
            color: #3d3d3d;
            vertical-align: top;
            line-height: 1.6;
          }

          /* Best-for badges */
          .best-for {
            background: #f0f9ff;
            border-left: 3px solid #0194e2;
            padding: 12px 16px;
            margin: 16px 0 0 0;
            border-radius: 0 4px 4px 0;
            font-size: 15px;
            color: #3d3d3d;
          }
          .best-for strong {
            color: #0072b0;
          }

          /* TL;DR box */
          .tldr-box {
            background: #fafafa;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 24px 28px;
            margin: 32px 0 48px 0;
            position: relative;
            overflow: hidden;
          }
          .tldr-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 3px;
            height: 100%;
            background: linear-gradient(180deg, #1a1a1a 0%, #6b7280 100%);
          }
          .tldr-label {
            display: inline-block;
            font-family: 'DM Mono', monospace;
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #0072b0;
            background: #e8f4fd;
            padding: 3px 10px;
            border-radius: 4px;
            margin-bottom: 14px;
          }
          .article-container .tldr-box p {
            font-size: 15px;
            margin-bottom: 10px;
            line-height: 1.7;
          }
          .article-container .tldr-box p:last-child {
            margin-bottom: 0;
          }

          /* Per-product FAQ */
          .product-faq {
            margin: 24px 0 0 0;
          }
          .product-faq summary {
            font-family: 'DM Sans', sans-serif;
            font-size: 15px;
            font-weight: 600;
            color: #1a1a1a;
            cursor: pointer;
            padding: 10px 0;
            border-bottom: 1px solid #f0f0f0;
            list-style: none;
          }
          .product-faq summary::-webkit-details-marker {
            display: none;
          }
          .product-faq summary::before {
            content: '+';
            display: inline-block;
            width: 20px;
            font-weight: 400;
            color: #6b7280;
          }
          .product-faq details[open] > summary::before {
            content: '−';
          }
          .article-container .product-faq p {
            font-size: 14px;
            color: #4b5563;
            margin: 8px 0 16px 20px;
            line-height: 1.7;
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
          }
        `}</style>
      </Head>

      <div className="article-page">
        <Header />

        <div className="article-container">
          <h1>Top 5 Agent Observability Tools in 2026</h1>
          <p className="subtitle">
            AI agents are quickly becoming the default architecture for
            production LLM applications. Multi-step reasoning, tool use,
            planning, and autonomous decision-making introduce complexity that
            makes traditional logging woefully inadequate. In this guide, we
            compare the top five agent observability tools and help you choose
            the right one for your team.
          </p>

          <div className="tldr-box">
            <span className="tldr-label">TL;DR</span>
            <p>
              <strong>
                <Link to="/">MLflow</Link>
              </strong>
              , the most widely adopted open source AI engineering platform with
              30M+ monthly downloads, is the top pick for teams who care about
              trace data ownership and want a complete platform for building
              production-grade agents. It covers observability, evaluation,
              prompt optimization, and governance in one place, with no
              enterprise paywalls.
            </p>
            <p>
              <strong>Alternatives:</strong>{" "}
              <Link to="https://langfuse.com">Langfuse</Link> for
              ClickHouse-native self-hosting,{" "}
              <Link to="https://www.langchain.com/langsmith/observability">
                LangSmith
              </Link>{" "}
              for teams fully committed to LangChain,{" "}
              <Link to="https://www.braintrust.dev">Braintrust</Link> for fast
              prototyping with non-technical stakeholders.
            </p>
          </div>

          <div
            style={{
              margin: "40px 0",
              borderRadius: "8px",
              overflow: "hidden",
              border: "1px solid #e5e7eb",
            }}
          >
            <video width="100%" controls autoPlay loop muted playsInline>
              <source
                src="https://mlflow.org/docs/latest/images/llms/tracing/tracing-top.mp4"
                type="video/mp4"
              />
              Your browser does not support the video tag.
            </video>
          </div>

          {/* What to Look For */}
          <h2 id="what-to-look-for" data-toc="What to Look For">
            What to Look For in an Agent Observability Tool
          </h2>
          <p>
            <Link to="/ai-observability">Agent observability</Link> is
            end-to-end visibility into every step an AI agent takes in
            production: LLM calls, tool invocations, retrieval steps, and
            planning decisions. Every tool on this list can capture traces. The real question is
            what happens after the trace lands. Before comparing platforms, here
            are the three capabilities that separate production-grade
            observability from expensive logging.
          </p>

          <h3>1. Framework and ecosystem flexibility</h3>
          <p>
            The agent framework landscape moves fast: LangGraph, OpenAI Agents
            SDK, DSPy, Pydantic AI, CrewAI, and new entrants every quarter. Your
            observability platform should integrate with all of them through a
            unified API, not lock you into a single framework's ecosystem. The
            same goes for LLM providers, coding agents, and deployment targets.
            If switching frameworks means rebuilding your observability setup,
            the tool is a liability, not an asset.
          </p>

          <h3>2. Tight integration with the agent development loop</h3>
          <p>
            Traces that sit in a dashboard forever do not improve your agents. A
            well-integrated AI platform converts your trace data into fuel for
            the agent improvement loop. Once traces flow into the platform, you
            can{" "}
            <Link to="https://mlflow.org/docs/latest/genai/eval-monitor/">
              <strong>evaluate </strong>
            </Link>
            the agent's performance, <strong>optimize</strong> prompts, and{" "}
            <strong>monitor </strong>
            the agent's behavior in production.
          </p>

          <h3>3. Vendor lock-in risk on your trace data</h3>
          <p>
            Traces are among the most valuable data an AI team generates. They
            capture what your agents actually do in production and can contain
            sensitive information that must be protected. If that data is locked
            inside a proprietary SaaS with no export path, you are handing a
            strategic asset to a vendor. Look for{" "}
            <strong>full open source availability</strong> so you can self-host
            on your own infrastructure and use the database and storage systems
            that best fit your environment, without being locked into a single
            vendor's architecture.
          </p>

          {/* Quick Comparison Table */}
          <h2 id="quick-comparison" data-toc="Quick Comparison">
            Quick Comparison
          </h2>
          <QuickComparisonTable rows={quickComparisonRows} />
          <p style={{ fontSize: "13px", marginTop: "-28px" }}>
            ¹ LangSmith's PyPI count is inflated because it is an automatic
            dependency of the <code>langchain</code> package.
          </p>

          {/* ───── 1. MLflow ───── */}
          <h2 id="mlflow" data-toc="1. MLflow">
            1. MLflow - The Complete Open Source AI Platform
          </h2>
          <p>
            <strong>
              <Link to="/">MLflow</Link>
            </strong>{" "}
            is the most widely deployed open source AI engineering platform.
            Built on top of its{" "}
            <Link to={`${MLFLOW_GENAI_DOCS_URL}tracing/`}>
              OpenTelemetry-native observability layer
            </Link>
            , MLflow provides a{" "}
            <strong>
              complete, production-focused AI engineering platform
            </strong>{" "}
            that covers the full lifecycle from prototyping to production,
            including tracing,{" "}
            <Link to={`${MLFLOW_GENAI_DOCS_URL}evaluation/`}>evaluation</Link>,{" "}
            <Link to={`${MLFLOW_GENAI_DOCS_URL}prompt-registry/`}>
              prompt management and optimization
            </Link>
            , and governance. While other tools on this list focus on one slice
            of the problem, MLflow is built for teams that need to get agents
            into production and keep them there.
          </p>
          <div className="screenshot-wrap">
            <img
              src={MlflowTracingUI}
              alt="MLflow tracing UI showing a LangGraph agent trace with tool calls, messages, and assessments"
            />
          </div>

          <h4 style={{ color: "black" }}>
            Fully Open Source, Backed by the Linux Foundation
          </h4>
          <p>
            Unlike tools tied to a single vendor's commercial interests, MLflow
            is governed by the{" "}
            <Link to="https://www.linuxfoundation.org/">Linux Foundation</Link>{" "}
            - the trusted foundation for open source projects like Linux,
            Kubernetes, and PyTorch. Every feature in MLflow is available in the
            open source release and will remain so. There is no paywall that
            gates critical capabilities. The strong commitment to openness is
            also reflected in the technical choices MLflow makes, for example,
            OpenTelemetry, the vendor-neutral observability standard, is used as
            a foundation layer for MLflow's observability capabilities.
          </p>

          <h4 style={{ color: "black" }}>
            A Complete AI Platform, Not Just a Tracing Layer
          </h4>
          <p>
            Most observability tools stop at tracing. MLflow goes far beyond.
            Its{" "}
            <Link to={`${MLFLOW_GENAI_DOCS_URL}evaluation/`}>
              <strong>production-grade evaluation</strong>
            </Link>{" "}
            system includes built-in LLM judges, multi-turn evaluation,
            integration with leading eval libraries (RAGAS, DeepEval, Phoenix,
            TruLens, Guardrails AI), metric versioning, and the ability to align
            judges with human feedback.
          </p>
          <p>
            On top of that, MLflow offers <strong>prompt optimization</strong>{" "}
            with state-of-the-art algorithms like{" "}
            <Link
              to={`${MLFLOW_GENAI_DOCS_URL}prompt-registry/optimize-prompts/`}
            >
              GEPA and MIPRO
            </Link>{" "}
            that automatically improve prompts based on evaluation results, so
            you can stop tweaking prompts by hand and let the optimizer find
            what works.
          </p>
          <p>
            Finally, the{" "}
            <strong>
              <Link to="/ai-gateway">AI Gateway</Link>
            </strong>{" "}
            provides a centralized layer for governing LLM access across your
            organization, with routing, rate limiting, fallbacks, usage
            tracking, and credential management across providers (OpenAI,
            Anthropic, Bedrock, Azure, Gemini, and more). MLflow also includes a
            built-in{" "}
            <Link to={`${MLFLOW_GENAI_DOCS_URL}tracing/observe-with-traces/ui`}>
              AI Assistant
            </Link>{" "}
            that helps you debug traces and diagnose issues directly within the
            UI.
          </p>

          <h4 style={{ color: "black" }}>Simple, Flexible Self-Hosting</h4>
          <p>
            MLflow's architecture is intentionally simple: a server, a database,
            and object storage. You choose the database (PostgreSQL, MySQL,
            SQLite, AWS RDS, GCP Cloud SQL, Neon, Supabase) and the storage
            backend (S3, GCS, Azure Blob, HDFS, or local filesystem). Most teams
            deploy MLflow in minutes using infrastructure they already know. See
            the{" "}
            <Link to={`${MLFLOW_GENAI_DOCS_URL}tracing/quickstart`}>
              Tracing Quickstart
            </Link>{" "}
            to get started.
          </p>

          <ProsConsTable
            pros={[
              "Fully open source (Apache 2.0) with Linux Foundation governance. No feature gating or vendor lock-in.",
              "Complete platform: tracing, evaluation, prompt optimization, and governance in one tool",
              "Simple self-hosting with flexible backends",
            ]}
            cons={[
              "Might not be the best fit for teams that only need quick prototyping",
              "Broader feature set than single-purpose tracing tools, which may not be needed for simple use cases",
            ]}
          />

          <div className="best-for">
            <strong>Best for:</strong> Teams who care about trace data ownership
            and want to get the most value from it for building production-grade
            agents. The only fully open source platform that covers
            observability, evaluation, prompt optimization, governance, and AI
            gateway in one place, with no enterprise paywalls.
          </div>

          <div className="product-faq">
            <details>
              <summary>What languages does MLflow support?</summary>
              <p>
                MLflow provides native SDKs for both Python and TypeScript. On
                top of that, the{" "}
                <Link to={`${MLFLOW_GENAI_DOCS_URL}tracing/`}>tracing API</Link>{" "}
                is built on OpenTelemetry, so any language with an OTel SDK can
                export traces to MLflow's tracking server, giving you broad
                compatibility beyond the first-party SDKs.
              </p>
            </details>
            <details>
              <summary>
                How does MLflow handle high-volume production traces?
              </summary>
              <p>
                MLflow's self-hosted architecture scales with your choice of
                database and storage backend. Teams run PostgreSQL or MySQL for
                metadata and S3/GCS/Azure Blob for artifacts. There are no
                vendor-imposed retention limits or per-trace pricing.
              </p>
            </details>
            <details>
              <summary>Is MLflow really free? What is the catch?</summary>
              <p>
                Every feature in MLflow is available under the Apache 2.0
                license with no enterprise paywall. The project is governed by
                the Linux Foundation, which ensures long-term neutrality.
                Databricks offers a managed version for teams that prefer not to
                self-host, but the open source release is fully featured.
              </p>
            </details>
            <details>
              <summary>What agent frameworks does MLflow support?</summary>
              <p>
                MLflow provides auto-instrumentation for 60+ frameworks via
                OpenTelemetry, including OpenAI Agents SDK, LangGraph,
                LlamaIndex, DSPy, Pydantic AI, CrewAI, Anthropic, AWS Bedrock,
                Google ADK, and more. See the full list in the{" "}
                <Link to={`${MLFLOW_GENAI_DOCS_URL}tracing/integrations/`}>
                  integrations docs
                </Link>
                .
              </p>
            </details>
            <details>
              <summary>Does MLflow have a managed solution?</summary>
              <p>
                Yes. MLflow is available as a managed service on multiple cloud
                platforms, including Databricks, Amazon SageMaker, Azure ML, Red
                Hat OpenShift AI, and Nebius AI. All of these offer the same
                MLflow features without the overhead of self-hosting.
              </p>
            </details>
          </div>

          {/* ───── 2. Langfuse ───── */}
          <h2 id="langfuse" data-toc="2. Langfuse">
            2. Langfuse - Tracing for ClickHouse Experts
          </h2>
          <p>
            <strong>
              <Link to="https://langfuse.com">Langfuse</Link>
            </strong>{" "}
            is an open source observability platform focused primarily on
            tracing and monitoring LLM applications. Built around{" "}
            <strong>ClickHouse</strong> for its analytical query engine,
            Langfuse provides a clean UI for exploring traces, a prompt
            playground for manual iteration, basic LLM-as-judge scoring, and
            cost analytics. Teams already invested in the ClickHouse ecosystem
            will feel at home, though others may find the infrastructure
            requirements steep.
          </p>
          <div className="screenshot-wrap">
            <img
              src={LangfuseTracingUI}
              alt="Langfuse tracing UI showing traces, spans, and trace detail view"
            />
          </div>
          <h4 style={{ color: "black" }}>
            Open Source with Self-Hosting Support
          </h4>
          <p>
            Langfuse is open source under the MIT license. Teams can self-host
            Langfuse on their own infrastructure, though doing so requires
            running 5+ services including ClickHouse, PostgreSQL, Redis, and the
            Langfuse application server. The managed (cloud) version handles
            this complexity for you, with free and paid tiers.
          </p>

          <h4 style={{ color: "black" }}>Playground Experience</h4>
          <p>
            Langfuse offers a prompt playground that lets you iterate on prompts
            directly within the UI. You can compare outputs across model
            configurations and test prompt variations side-by-side, which is
            useful for manual prompt engineering workflows. Basic LLM-as-judge
            scoring is also available for lightweight evaluation.
          </p>

          <h4 style={{ color: "black" }}>Strong Analytics Backend</h4>
          <p>
            Built on ClickHouse, Langfuse can handle high-throughput trace
            ingestion and provides fast analytical queries over large datasets.
            Teams already running ClickHouse infrastructure will appreciate the
            familiar operational model. However, this also means self-hosting
            locks you into ClickHouse as a dependency, and there is no option to
            swap in a different database backend.
          </p>

          <ProsConsTable
            pros={[
              "Open source (MIT) with self-hosting support.",
              "Playground experience for manual prompt engineering",
              "Strong analytics backend with ClickHouse",
            ]}
            cons={[
              "Self-hosting requires ClickHouse expertise and requires running 5+ services.",
              "Key features like SSO, RBAC, and advanced evaluation are gated behind paid plans.",
              "Steep operational overhead and frequent architecture changes in the past.",
            ]}
          />

          <div className="best-for">
            <strong>Best for:</strong> Teams already running ClickHouse
            infrastructure who primarily need tracing and a prompt playground.
            Be prepared to add separate tools (e.g., LiteLLM for AI gateway,
            third-party eval frameworks) as your agent stack matures.
          </div>

          <div className="product-faq">
            <details>
              <summary>Is Langfuse fully open source?</summary>
              <p>
                The core is MIT-licensed, but enterprise features in the{" "}
                <code>ee</code> folders have separate licensing. Some
                capabilities like SSO and advanced RBAC require a paid plan.
              </p>
            </details>
            <details>
              <summary>What languages does Langfuse support?</summary>
              <p>
                Native SDKs are available for Python and TypeScript. Other
                languages require building custom wrappers around the REST API.
              </p>
            </details>
            <details>
              <summary>Can I use a database other than ClickHouse?</summary>
              <p>
                No. Langfuse's self-hosted version requires ClickHouse for trace
                storage. There is no option to swap in PostgreSQL, MySQL, or
                another analytical database.
              </p>
            </details>
            <details>
              <summary>What happened with the ClickHouse acquisition?</summary>
              <p>
                Langfuse was acquired by ClickHouse, Inc. The long-term product
                roadmap and investment level remain to be seen. Check the
                official Langfuse blog for the latest updates.
              </p>
            </details>
          </div>

          {/* ───── 3. LangSmith ───── */}
          <h2 id="langsmith" data-toc="3. LangSmith">
            3. LangSmith - LangChain's Own Observability Platform
          </h2>
          <p>
            <strong>
              <Link to="https://www.langchain.com/langsmith/observability">
                LangSmith
              </Link>
            </strong>{" "}
            is the commercial observability platform built by LangChain. It
            provides detailed tracing, evaluation, and monitoring capabilities
            with strong support for LangChain and LangGraph applications, though
            teams using other frameworks may find the experience less polished.
          </p>
          <div className="screenshot-wrap">
            <img
              src={LangsmithTracingUI}
              alt="LangSmith tracing UI showing trace runs, inputs, outputs, and latency"
            />
          </div>

          <h4 style={{ color: "black" }}>
            Deep LangChain/LangGraph Integration
          </h4>
          <p>
            LangSmith provides the richest tracing detail for applications built
            on LangChain and LangGraph, with native agent graph visualization
            and annotation queues for structured human review. If your stack is
            LangChain-centric, the first-party experience is polished.
          </p>

          <h4 style={{ color: "black" }}>AI-powered Features</h4>
          <p>
            LangSmith provides a rich set of AI-powered features, including
            Polly AI Assistant, topic clustering, and Insights Agent, which use
            LLMs to analyze your trace data on your behalf. Some of these
            features are available only in paid plans (Plus or Enterprise).
          </p>

          <h4 style={{ color: "black" }}>Proprietary, Closed-Source Model</h4>
          <p>
            LangSmith is a proprietary SaaS platform and there is no self-hosted
            option outside enterprise contracts. The free tier is limited to
            5,000 traces per month with 14-day retention, and per-seat pricing
            ($39/seat/month on Plus) can limit access for PMs and QA.
          </p>

          <ProsConsTable
            pros={[
              "Strong integration with LangChain/LangGraph ecosystem",
              "Visual no-code agent authoring experience with LangSmith Studio",
              "AI-powered features like Polly AI Assistant, topic clustering, and insights agent for automated trace analysis",
            ]}
            cons={[
              "Proprietary, closed-source. No self-hosted option outside enterprise contracts",
              "Per-seat pricing can limit collaboration and sharing across teams",
              "Feature parity lags for integrations outside the LangChain ecosystem",
            ]}
          />

          <div className="best-for">
            <strong>Best for:</strong> Teams 100% committed to the
            LangChain/LangGraph ecosystem who are comfortable with proprietary
            SaaS pricing that scales with trace volume.
          </div>

          <div className="product-faq">
            <details>
              <summary>Does LangSmith only work with LangChain?</summary>
              <p>
                No. LangSmith supports other frameworks via OpenTelemetry
                ingestion and a <code>traceable</code> wrapper. However,
                community feedback suggests the experience is most polished for
                LangChain and LangGraph applications.
              </p>
            </details>
            <details>
              <summary>Can I self-host LangSmith?</summary>
              <p>
                Self-hosting is only available on the Enterprise tier. The free
                and Plus plans are SaaS-only with data stored on LangChain's
                infrastructure.
              </p>
            </details>
            <details>
              <summary>How does LangSmith pricing work?</summary>
              <p>
                The free tier includes 5,000 traces/month with 14-day retention.
                Plus is $39/seat/month with higher limits. Extended retention
                (400 days) is available as a paid add-on. Enterprise pricing is
                custom.
              </p>
            </details>
            <details>
              <summary>What are LangSmith's AI-powered features?</summary>
              <p>
                LangSmith offers Polly AI Assistant for natural-language trace
                debugging, topic clustering for automatic behavior
                categorization, and an insights agent that prioritizes
                improvements by frequency and impact. Some of these features are
                gated behind Plus or Enterprise plans.
              </p>
            </details>
          </div>

          {/* ───── 4. Arize Phoenix ───── */}
          <h2 id="arize-phoenix" data-toc="4. Arize Phoenix">
            4. Arize Phoenix: ML Monitoring Meets Observability
          </h2>
          <p>
            <strong>
              <Link to="https://phoenix.arize.com/">Arize Phoenix</Link>
            </strong>{" "}
            is the open source observability tool from Arize AI, a company that
            started in classical ML monitoring and is now expanding into the
            GenAI space. That monitoring heritage shows in Phoenix's strengths:
            built-in evaluation metrics, drift detection, and trace analytics.
          </p>
          <div className="screenshot-wrap">
            <img
              src={ArizePhoenixUI}
              alt="Arize Phoenix UI showing traces, evaluation metrics, and agent analysis"
            />
          </div>

          <h4 style={{ color: "black" }}>
            Built-in Evaluation Metrics
          </h4>
          <p>
            Phoenix ships with 50+ research-backed metrics covering
            faithfulness, relevance, safety, toxicity, and hallucination
            detection. Multi-step agent trajectory analysis helps teams
            understand complex agent behavior, and advanced analytics include
            trace clustering, anomaly detection, and retrieval relevancy
            visualization for RAG pipelines.
          </p>

          <h4 style={{ color: "black" }}>OpenInference and OpenTelemetry</h4>
          <p>
            Phoenix owns the{" "}
            <Link to="https://github.com/Arize-ai/openinference">
              OpenInference
            </Link>{" "}
            standard, a set of custom instrumentation SDKs for OpenTelemetry
            that provide framework-native tracing across 40+ integrations. The
            open source version (Phoenix) is available for self-hosting on a
            single node.
          </p>

          <h4 style={{ color: "black" }}>
            Source-Available, Not Fully Open Source
          </h4>
          <p>
            Phoenix uses the Elastic License 2.0 (ELv2), which restricts
            offering the software as a managed service. High-value features like
            the Alyx Copilot, online evaluations, and monitoring are gated
            behind paid plans.
            Phoenix does not offer prompt optimization, an AI gateway, or
            governance capabilities, and scaling beyond single-node deployments
            requires additional planning. The project is backed by Arize AI, so
            its long-term roadmap may be influenced by commercial priorities.
          </p>

          <ProsConsTable
            pros={[
              "Source-available Phoenix is available for self-hosting",
              "Strong set of research-backed evaluation metrics out of the box",
              "Owns OpenInference, a set of custom instrumentation SDKs for OpenTelemetry that provide framework-native tracing",
            ]}
            cons={[
              "High-value features are gated behind paid plans, such as Alyx Copilot and online evaluations.",
              "Evaluation options outside the built-in metrics are limited, such as multi-turn evaluation",
              "Elastic License 2.0 restricts the use of the software as a managed service",
            ]}
          />

          <div className="best-for">
            <strong>Best for:</strong> Research-oriented teams focused on
            evaluation metrics who want a free, single-node observability tool.
            Pairs well with MLflow for teams that need a more complete platform.
          </div>

          <div className="product-faq">
            <details>
              <summary>Is Arize Phoenix open source?</summary>
              <p>
                Phoenix is source-available under the Elastic License 2.0
                (ELv2), which allows free use but restricts offering the
                software as a managed service. This is a more restrictive
                license than Apache 2.0 or MIT.
              </p>
            </details>
            <details>
              <summary>What is OpenInference?</summary>
              <p>
                OpenInference is a set of custom instrumentation SDKs built on
                top of OpenTelemetry, maintained by Arize AI. It provides
                framework-native tracing for 40+ integrations including
                LlamaIndex, LangChain, and DSPy.
              </p>
            </details>
            <details>
              <summary>Can Phoenix scale beyond a single node?</summary>
              <p>
                The open source version is designed for single-node deployment.
                Scaling beyond that requires the commercial Arize AX platform,
                which offers managed cloud hosting with tiered pricing.
              </p>
            </details>
            <details>
              <summary>
                How does Phoenix compare to MLflow for evaluation?
              </summary>
              <p>
                Phoenix focuses on research-backed metrics (faithfulness,
                toxicity, hallucination detection) and works well for RAG
                evaluation. MLflow covers a broader evaluation surface including
                multi-turn evaluation, LLM judge alignment with human feedback,
                and automated prompt optimization based on eval results. The two
                can be used together since MLflow integrates with Phoenix as an
                eval library.
              </p>
            </details>
          </div>

          {/* ───── 5. Braintrust ───── */}
          <h2 id="braintrust" data-toc="5. Braintrust">
            5. Braintrust: Quick Analytics for Non-Technical Users
          </h2>
          <p>
            <strong>
              <Link to="https://www.braintrust.dev">Braintrust</Link>
            </strong>{" "}
            is a commercial AI observability platform designed for speed and
            ease of use, targeting teams where not everyone is deeply technical.
            Its purpose-built database (Brainstore) can efficiently analyze
            production traces, and its AI proxy provides automatic logging of
            LLM calls with minimal setup, though deeper agent-level tracing
            still requires SDK instrumentation.
          </p>
          <div className="screenshot-wrap">
            <img
              src={BraintrustTraceUI}
              alt="Braintrust trace view showing request details, spans, and scoring"
            />
          </div>
          <h4 style={{ color: "black" }}>Fast Analytics and Approachable UI</h4>
          <p>
            Braintrust's purpose-built Brainstore database is designed for AI
            workload patterns, delivering fast query performance over production
            traces. The UI is approachable for prompt iteration and output
            comparison, with 25+ built-in scorers and the ability to generate
            custom scorers from natural language descriptions.
          </p>

          <h4 style={{ color: "black" }}>AI Gateway for Unified LLM API</h4>
          <p>
            The recent addition of AI Gateway (formerly known as AI proxy) lets
            you route requests to many LLM providers with a unified LLM API. It
            provides basic features for teams to manage LLM access, such as
            caching, logging, and access control.
          </p>

          <h4 style={{ color: "black" }}>
            Proprietary SaaS with Steep Pricing Tiers
          </h4>
          <p>
            Braintrust is a proprietary SaaS platform with no self-hosted option
            and trace data stays with the vendor. The jump from free to paid
            tiers is steep ($249/mo for Pro). It does not offer built-in prompt
            optimization or a broader governance layer, and framework
            integration coverage is narrower than platforms with native
            OpenTelemetry support.
          </p>

          <ProsConsTable
            pros={[
              "Fast analytics on high-volume traces with purpose-built database",
              "Approachable UI for prompt iteration and non-technical stakeholders",
              "AI proxy provides automatic LLM call logging with minimal setup",
            ]}
            cons={[
              "Proprietary SaaS with no self-hosted option. Trace data stays with the vendor.",
              "Steep pricing jump from free to paid tiers ($249/mo for Pro)",
              "Narrow integration coverage for agent frameworks.",
            ]}
          />

          <div className="best-for">
            <strong>Best for:</strong> Teams doing fast prototyping with
            non-technical stakeholders who need approachable analytics and quick
            zero-config setup. No self-hosting option, and trace data stays with
            the vendor.
          </div>

          <div className="product-faq">
            <details>
              <summary>Can I self-host Braintrust?</summary>
              <p>
                No. Braintrust is a proprietary SaaS platform with no
                self-hosted option. All trace data is stored on Braintrust's
                infrastructure.
              </p>
            </details>
            <details>
              <summary>What is Brainstore?</summary>
              <p>
                Brainstore is Braintrust's purpose-built database designed for
                AI workload patterns. It enables fast analytical queries over
                millions of production traces.
              </p>
            </details>
            <details>
              <summary>How does Braintrust pricing work?</summary>
              <p>
                Braintrust offers a free tier with limited usage. The jump to
                Pro is $249/month, which can be steep for smaller teams. There
                is no public Enterprise pricing.
              </p>
            </details>
            <details>
              <summary>What happens to my trace data with Braintrust?</summary>
              <p>
                All trace data is stored on Braintrust's infrastructure with no
                option to self-host or bring your own storage. Data retention is
                14 days on the Starter plan and 30 days on Pro. Teams that need
                full control over trace data ownership should consider an
                open source alternative.
              </p>
            </details>
          </div>

          {/* How to Choose */}
          <h2 id="how-to-choose" data-toc="How to Choose the Right Tool">
            How to Choose the Right Tool
          </h2>
          <p>
            All five tools on this list can capture traces. The difference lies
            in what happens after you collect that data, and how much control
            you retain over it. Before picking a tool, consider these criteria:
          </p>

          <h4 style={{ color: "black" }}>Trace Data Ownership</h4>
          <p>
            Traces are among the most valuable assets an AI team generates. They
            encode how your agent reasons, what tools it calls, and where it
            fails. Ask yourself: who owns this data? Can you store it on your
            own infrastructure? Can you query it with your own tools, or are you
            locked into a vendor's retention window and export format?
          </p>

          <h4 style={{ color: "black" }}>
            From Traces to Production-Grade Agents
          </h4>
          <p>
            Observability is only the first step. To ship reliable agents you
            also need evaluation, governance, and an AI gateway, ideally within
            the same platform so insights flow naturally from traces to
            improvements. A tool that only does tracing will force you to stitch
            together separate solutions for each of these stages, increasing
            complexity and maintenance cost.
          </p>

          <h4 style={{ color: "black" }}>Flexibility and Portability</h4>
          <p>
            Agent frameworks evolve fast. The tool you choose should not tie you
            to a single framework, a single database, or a single vendor. Native
            OpenTelemetry support, a permissive open source license, and simple
            self-hosting options all protect you from lock-in as your stack
            changes.
          </p>

          {/* Our Recommendation */}
          <h2 id="recommendation" data-toc="Recommendation">
            Our Recommendation
          </h2>
          <p>
            For teams who care about trace data ownership and want to get the
            most value from that data to build production-grade agents,{" "}
            <strong>
              <Link to="/">MLflow</Link>
            </strong>{" "}
            is our top recommendation. It is the only tool on this list that is
            fully open source under the Apache 2.0 license, backed by the Linux
            Foundation, and offers observability, evaluation, governance, and an
            AI gateway in a single platform, with no enterprise paywall on any
            feature.{" "}
            <Link to="https://mlflow.org/docs/latest/genai/tracing/quickstart/">
              Get started with the Tracing Quickstart
            </Link>
            .
          </p>

          <h4 style={{ color: "black" }}>Alternatives Worth Considering</h4>
          <p>
            <strong>Langfuse</strong> is a reasonable self-hosted alternative if
            your team is already invested in ClickHouse, but it covers tracing
            and prompt management only; the stack tends to grow as you add
            LiteLLM and separate evaluation tools. <strong>LangSmith</strong>{" "}
            provides the deepest LangChain/LangGraph integration, though it is
            proprietary and pricing scales with volume.{" "}
            <strong>Braintrust</strong> suits fast prototyping with
            non-technical stakeholders, but it is a proprietary SaaS with no
            self-hosting option.
          </p>

          {/* Global FAQ */}
          <h2 id="faq" data-toc="FAQ">
            Frequently Asked Questions
          </h2>

          <h3>What is agent observability?</h3>
          <p>
            Agent observability is end-to-end visibility into every step an AI
            agent takes in production: LLM calls, tool invocations, retrieval
            steps, planning decisions, and the cascading effects between them.
            Unlike traditional APM that monitors latency and errors, agent
            observability tracks output quality, faithfulness, safety, and
            behavioral drift. Learn more on the{" "}
            <Link to="/ai-observability">AI Observability</Link> page.
          </p>

          <h3>Why is open source important for observability?</h3>
          <p>
            Traces are among the most valuable data an AI team generates. Open
            source ensures you own that data, can self-host on your own
            infrastructure, avoid vendor lock-in, and maintain full transparency
            into how your observability stack works. Learn more on the{" "}
            <Link to="/ai-platform">AI Platform</Link> page.
          </p>

          <h3>What is OpenTelemetry?</h3>
          <p>
            OpenTelemetry is an open source project that provides a
            vendor-neutral standard for collecting, processing, and exporting
            telemetry data. It is widely used across observability tools, and
            choosing OpenTelemetry-compatible platforms helps keep your trace
            data portable. Learn more on the{" "}
            <Link to="https://opentelemetry.io/">OpenTelemetry</Link> website.
          </p>

          <h3>What are other important concepts in LLMOps?</h3>
          <p>
            There are other important concepts in LLMOps, such as prompt
            management, cost control, and governance. See the full guide on the{" "}
            <Link to="/llmops">LLMOps</Link> page.
          </p>

          <h3>How long does it take to adopt an observability tool?</h3>
          <p>
            Adoption is usually quick, especially with a tool like MLflow that
            is designed to be easy to use and self-host. For example, to
            instrument an application built with the OpenAI Agents SDK, you just
            need to add a single `mlflow.openai.autolog()` call to your
            application code. This is why most teams start with observability as
            a first step in their LLMOps journey.
          </p>

          <h3>
            What is the difference between traditional APM and agent
            observability?
          </h3>
          <p>
            Traditional APM focuses on monitoring application performance and
            health, including latency, errors, and throughput. Agent
            observability focuses on agent behavior, including output quality,
            tool usage, and planning decisions. It gives you a more complete
            view of how the agent behaves in production.
          </p>

          {/* Related Resources */}
          <h2>Related Resources</h2>
          <div className="related-resources">
            <ul>
              <li>
                <Link to="/llmops">What is LLMOps?</Link>
              </li>
              <li>
                <Link to={`${MLFLOW_GENAI_DOCS_URL}tracing/quickstart`}>
                  Tracing & Observability Quickstart
                </Link>
              </li>
              <li>
                <Link to="/langfuse-alternative">
                  MLflow vs Langfuse: Detailed Comparison
                </Link>
              </li>
              <li>
                <Link to="/langsmith-alternative">
                  MLflow vs LangSmith: Detailed Comparison
                </Link>
              </li>
              <li>
                <Link to="/arize-phoenix-alternative">
                  MLflow vs Arize Phoenix: Detailed Comparison
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
