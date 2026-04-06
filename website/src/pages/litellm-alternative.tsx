import { useState } from "react";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../components/Header/Header";
import { ArticleSidebar } from "../components/ArticleSidebar/ArticleSidebar";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import { CopyButton } from "../components/CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";
import MlflowLogo from "@site/static/img/litellm-alternative/mlflow-logo.png";
import LitellmLogoImg from "@site/static/img/litellm-alternative/litellm-logo.png";
import LitellmUI from "@site/static/img/litellm-alternative/litellm-ui.png";
import MlflowEvalUI from "@site/static/img/litellm-alternative/mlflow-eval-ui.png";
import MlflowGatewayUI from "@site/static/img/litellm-alternative/mlflow-gateway-ui.png";

const tracingExamples: { label: string; mlflow: string; litellm: string }[] = [
  {
    label: "Gateway Tracing",
    mlflow: `# Enable "Usage Tracking" on the endpoint
# in the MLflow AI Gateway UI. Done.
#
# Every gateway request is automatically
# traced with latency, token usage, cost,
# and error details. No code changes.
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/gateway/v1")

# Invoke the client as usual
`,
    litellm: `import litellm

# Observability requires configuring an
# external callback (e.g. Datadog, Langfuse).

litellm.success_callback = ["datadog"]
litellm.failure_callback = ["datadog"]

# Also requires:
# - Datadog agent running
# - DD_API_KEY configured
# - Separate dashboard setup`,
  },
  {
    label: "Agent Tracing",
    mlflow: `import mlflow

mlflow.langgraph.autolog()

# Every node, edge, and tool call
# is traced automatically, from agent
# orchestration down to LLM requests.
#
# Agent traces and gateway traces are
# linked automatically via distributed
# tracing - no extra code needed.`,
    litellm: `# LiteLLM focuses on LLM API call
# proxying. Agent-level tracing (tool
# calls, multi-step reasoning) requires
# a separate tool.
#
# Client-side and proxy-side traces are
# independent -- connecting them requires
# manual correlation across systems.`,
  },
];

const gatewayFeatures: [string, string, string][] = [
  ["Feature", "MLflow", "LiteLLM"],
  ["Multi-provider Routing", "✅", "✅"],
  ["OpenAI-compatible API", "✅", "✅"],
  ["Rate Limiting", "✅", "✅"],
  ["Cost Tracking", "✅", "✅"],
  ["Fallbacks & Load Balancing", "✅", "✅"],
  ["Passthrough / Native Provider APIs", "✅", "✅"],
  ["Guardrails", "✅", "Enterprise only"],
  ["Observability", "✅", <>❌<br/> (external tools required)</>],
];

function ComparisonTable({ rows }: { rows: string[][] }) {
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

function LitellmCodeLabel() {
  return (
    <span className="code-logo-inline">
      <img src={LitellmLogoImg} alt="LiteLLM" className="code-logo-icon" />
      LiteLLM
    </span>
  );
}

function CodeBlock({ code, label }: { code: string; label: string }) {
  return (
    <div className="code-side">
      <div className="code-side-header">
        <span className="code-side-label">
          {label === "MLflow" ? <MlflowCodeLabel /> : <LitellmCodeLabel />}
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
  tabs: { label: string; mlflow: string; litellm: string }[];
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
        <CodeBlock code={tabs[active].litellm} label="LiteLLM" />
      </div>
    </div>
  );
}

export default function LitellmAlternative() {
  return (
    <>
      <Head>
        <title>
          Open Source LiteLLM AI Gateway Alternative? MLflow vs LiteLLM | MLflow
        </title>
        <meta
          name="description"
          content="Compare MLflow and LiteLLM for AI Gateway, LLM tracing, and evaluation. Learn why teams choose MLflow for enterprise-grade security, complete platform capabilities, and Linux Foundation governance."
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
            padding-left: 24px !important;
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
            text-align: center;
            padding: 12px 16px;
            border-bottom: 2px solid #e5e7eb;
            color: #1a1a1a;
            font-weight: 600;
            background: #f9fafb;
          }
          .comparison-table thead th:first-child {
            text-align: left;
          }
          .comparison-table tbody td {
            padding: 12px 16px;
            border-bottom: 1px solid #f0f0f0;
            color: #505050;
            vertical-align: top;
            text-align: center;
          }
          .comparison-table tbody td.feature-cell {
            color: #1a1a1a;
            font-weight: 500;
            text-align: left;
          }
          .comparison-table tbody td.diff-cell {
            color: #16a34a;
            font-weight: 600;
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
          <h1>Open Source LiteLLM AI Gateway Alternative? LiteLLM vs MLflow</h1>
          <p className="subtitle">
            This article is for teams looking for a secure, enterprise-ready,
            open-source alternative to LiteLLM. We compare the two across
            governance, security, performance, tracing, and complete AI platform
            capabilities to help you decide which is the right fit.
          </p>

          {/* What is LiteLLM? */}
          <h2>What is LiteLLM?</h2>
          <div className="screenshot-wrap">
            <img
              src={LitellmUI}
              alt="LiteLLM proxy UI showing virtual key creation and model management"
            />
          </div>
          <p>
            <strong>
              <Link href="https://www.litellm.ai/">LiteLLM</Link>
            </strong>{" "}
            is an open source AI gateway maintained by BerriAI, a Y
            Combinator-backed startup. It provides a unified, OpenAI-compatible
            interface to over 100 LLM providers, with built-in features like
            cost tracking, rate limiting, and automatic fallbacks. LiteLLM also
            offers a Python SDK for direct client integration.
          </p>

          {/* What is MLflow? */}
          <h2>What is MLflow?</h2>
          <div className="screenshot-wrap">
            <img src={MlflowGatewayUI} alt="MLflow gateway UI" />
          </div>
          <p>
            <strong>
              <Link to="/">MLflow</Link>
            </strong>{" "}
            is an open source AI engineering platform for agents, LLMs, and
            models that enables teams of all sizes to debug, evaluate, monitor,
            and optimize production-quality AI applications while controlling
            costs and managing access to models and data. With over 30 million
            monthly downloads, thousands of organizations rely on MLflow each
            day to ship AI to production with confidence.
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
                  Need a <strong>secure gateway solution</strong> trusted by
                  thousands of enterprises
                </li>
                <li>
                  Want <strong>native observability</strong> integrated with
                  your gateway
                </li>
                <li>
                  Care about performance and want a{" "}
                  <strong>low-overhead</strong> gateway solution
                </li>
              </ul>
            </div>
            <div className="tldr-card">
              <h3>
                <img src={LitellmLogoImg} alt="LiteLLM" className="tldr-logo" />
                Choose LiteLLM if you...
              </h3>
              <ul>
                <li>
                  Need a <strong>dedicated LLM proxy</strong> for model routing
                </li>
                <li>
                  Use <strong>long-tail LLM providers</strong> that only LiteLLM
                  supports
                </li>
                <li>
                  Already using <strong>LiteLLM client</strong> and is satisfied
                  with its experience and stability
                </li>
              </ul>
            </div>
          </div>

          {/* Open Source */}
          <h2 id="open-source-governance">Open Source</h2>
          <p>
            <strong>LiteLLM</strong> is open source under the MIT license,
            maintained by{" "}
            <strong>
              <Link href="https://www.ycombinator.com/companies/berriai">
                BerriAI
              </Link>
            </strong>
            , a Y Combinator-backed startup. While the core SDK is freely
            available, enterprise features such as SSO, audit logging, and
            advanced admin controls are only available for paid customers with
            enterprise contract. Part of the source code is also under a
            separate license.
          </p>
          <p>
            <strong>MLflow</strong> is open source under Apache 2.0 and is{" "}
            <strong>
              backed by the{" "}
              <Link href="https://www.linuxfoundation.org/press/press-release/the-mlflow-project-joins-linux-foundation">
                Linux Foundation
              </Link>
            </strong>
            , the premier open source software foundation who also owns Linux,
            Kubernetes, and Pytorch. MLflow has been powering production AI
            since 2018 and maintains full feature parity between its open source
            release and managed offerings. With over 30 million monthly
            downloads and thousands of enterprise users, MLflow is one of the
            most widely deployed AI platforms.
          </p>

          {/* Security & Reliability */}
          <h2 id="security-reliability">Security &amp; Reliability</h2>
          <p>
            For teams deploying AI in production, the security and reliability
            of the tools in their stack are not optional. When an AI gateway
            sits between your applications and LLM providers, it becomes a
            critical piece of infrastructure that handles API keys, model
            access, and sensitive data.
          </p>
          <p>
            In March 2026, <strong>LiteLLM</strong> experienced a{" "}
            <Link href="https://securitylabs.datadoghq.com/articles/litellm-compromised-pypi-teampcp-supply-chain-campaign/">
              supply chain incident
            </Link>{" "}
            where compromised packages were briefly published to PyPI. BerriAI
            responded promptly and engaged Mandiant for forensic analysis. While
            such incidents can affect any open source project, this highlights
            the importance of evaluating the{" "}
            security practices and governance model of tools in
            your AI infrastructure.
          </p>
          <p>
            <strong>MLflow</strong> benefits from{" "}
            <strong>dedicated security team</strong> from Databricks and nearly a
            decade of hardening for enterprise deployments. With thousands of
            enterprise users worldwide, MLflow has a proven track record of
            reliability and security at scale. The Linux Foundation governance
            provides additional assurance that security practices meet
            enterprise standards, and the large contributor community means more
            eyes on the code and faster identification of potential issues.
          </p>

          {/* AI Gateway */}
          <h2 id="ai-gateway">AI Gateway Capabilities</h2>
          <p>
            Both MLflow and LiteLLM offer AI Gateway capabilities for routing
            requests to multiple LLM providers, managing costs, and enforcing
            usage policies. Most major providers (OpenAI, Anthropic, Google,
            Azure, AWS Bedrock, and more) are supported by both gateways. This
            is the primary overlap between the two tools.
          </p>
          <p>
            <strong>LiteLLM</strong> is purpose-built as a{" "}
            <Link href="https://docs.litellm.ai/docs/simple_proxy">
              gateway proxy
            </Link>
            ,
            offering broad provider support (100+), virtual key management, and
            an OpenAI-compatible API format that enables applications to switch
            providers without code changes. It offers rate limiting, cost
            tracking, and automatic fallbacks. However, LiteLLM operates{" "}
            <strong>
              in isolation from the rest of the AI development stack
            </strong>
            . To gain observability into gateway traffic, teams must configure
            external callback handlers. To evaluate model quality or optimize
            prompts, they need entirely separate tools.
          </p>
          <p>
            <strong>MLflow</strong> offers a built-in{" "}
            <Link to={`${MLFLOW_GENAI_DOCS_URL}gateway/`}>AI Gateway</Link> with
            similar routing, rate limiting, cost tracking, and fallback
            capabilities, but with a critical advantage:{" "}
            <strong>
              native integration with tracing, evaluation, and prompt management
            </strong>
            . When requests flow through MLflow's gateway, they are
            automatically captured in traces, enabling teams to connect cost and
            usage data with evaluation results and prompt performance, all in a
            single platform.
          </p>
          <ComparisonTable rows={gatewayFeatures} />

          {/* Performance */}
          <h2 id="performance">Performance</h2>
          <p>
            For production AI applications, gateway performance directly impacts
            user experience. Every millisecond of overhead added by the gateway
            is multiplied across millions of requests.
          </p>
          <p>
            We benchmarked both gateways with a 50 ms simulated provider delay
            (4 workers, 50 concurrent users). The overhead column shows gateway
            latency after subtracting the simulated delay. The results show that
            MLflow's AI Gateway adds{" "}
            <strong>approximately half the overhead</strong> and delivers{" "}
            <strong>67% higher throughput</strong> compared to LiteLLM.
          </p>
          <div className="comparison-table-wrap">
            <table className="comparison-table">
              <thead>
                <tr>
                  <th rowSpan={2}>Metric</th>
                  <th colSpan={2}>MLflow</th>
                  <th colSpan={2}>LiteLLM</th>
                  <th rowSpan={2}>Overhead Diff</th>
                </tr>
                <tr>
                  <th>Latency</th>
                  <th>Overhead</th>
                  <th>Latency</th>
                  <th>Overhead</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="feature-cell">P50</td>
                  <td>78.6 ms</td>
                  <td>28.6 ms</td>
                  <td>106.7 ms</td>
                  <td>56.7 ms</td>
                  <td className="diff-cell">-50%</td>
                </tr>
                <tr>
                  <td className="feature-cell">P99</td>
                  <td>184.2 ms</td>
                  <td>134.2 ms</td>
                  <td>388.8 ms</td>
                  <td>338.8 ms</td>
                  <td className="diff-cell">-60%</td>
                </tr>
                <tr>
                  <td className="feature-cell">Throughput</td>
                  <td colSpan={2}>598 req/s</td>
                  <td colSpan={2}>358 req/s</td>
                  <td className="diff-cell">+67%</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Tracing & Observability */}
          <h2 id="tracing-observability">Observability & Monitoring</h2>
          <p>
            Tracing is essential for understanding how AI applications behave in
            production. For gateway deployments, teams need visibility into
            every request flowing through the system, including latency, token
            usage, costs, and errors.
          </p>
          <p>
            <strong>LiteLLM</strong> does not include built-in tracing. To gain
            observability into gateway traffic, teams must configure an external
            tool such as Datadog or Langfuse as a callback handler. This means
            operating and maintaining a separate observability system alongside
            the gateway. LiteLLM also has no support for distributed tracing, so
            client-side agent traces and proxy-side request logs remain
            disconnected, making end-to-end debugging difficult.
          </p>
          <p>
            <strong>MLflow</strong>'s AI Gateway provides{" "}
            <strong>native tracing out of the box</strong>. When{" "}
            <Link to={`${MLFLOW_GENAI_DOCS_URL}gateway/usage-tracking/`}>
              usage tracking
            </Link>{" "}
            is enabled on an endpoint, every gateway request is automatically
            logged as a trace with full detail: latency, token consumption,
            cost, and errors, with no code changes required. MLflow also
            supports{" "}
            <strong>
              <Link
                to={`${MLFLOW_GENAI_DOCS_URL}tracing/app-instrumentation/distributed-tracing/`}
              >
                distributed tracing
              </Link>
            </strong>{" "}
            via the W3C TraceContext standard, which allows client-side agent
            traces to automatically link to gateway traces. This gives teams
            end-to-end visibility from agent orchestration through the gateway
            down to the LLM provider, all in a single trace view.
          </p>
          <p>
            Beyond the gateway, MLflow provides a <strong>one-line</strong>{" "}
            <code>autolog()</code> API for <strong>30+ frameworks</strong>,
            including OpenAI, LangGraph, DSPy, Anthropic, LangChain, Pydantic
            AI, CrewAI, and many more. Traces capture full span-level detail
            across the entire application stack and can be viewed, searched, and
            analyzed directly in the MLflow UI.
          </p>
          <CodeTabs tabs={tracingExamples} />

          {/* Complete AI Platform */}
          <h2 id="ai-platform">Beyond a Gateway: Complete AI Platform</h2>
          <p>
            A gateway is only one piece of a production AI stack. Teams also
            need observability, evaluation, prompt management, and model
            governance. <strong>LiteLLM</strong> does not offer any of these
            capabilities. Teams using LiteLLM must adopt and integrate separate
            tools for each of these needs, creating operational complexity and
            tool sprawl.
          </p>
          <p>
            <strong>MLflow</strong> is a{" "}
            <Link to="/ai-platform">complete AI engineering platform</Link> that
            covers the entire lifecycle:
          </p>
          <ul>
            <li>
              <Link to={`${MLFLOW_GENAI_DOCS_URL}tracing/`}>
                <strong>Observability</strong>
              </Link>{" "}
              with one-line <code>autolog()</code> for 30+ frameworks and
              built-in distributed tracing across client and gateway
            </li>
            <li>
              <Link to={`${MLFLOW_GENAI_DOCS_URL}evaluation/`}>
                <strong>Agent evaluation</strong>
              </Link>{" "}
              with built-in scorers, integration with leading evaluation
              libraries (RAGAS, DeepEval, Phoenix, TruLens, Guardrails AI),
              multi-turn evaluation, online evaluation, and LLM judge alignment
              with human feedback
            </li>
            <li>
              <Link to={`${MLFLOW_GENAI_DOCS_URL}prompt-registry/`}>
                <strong>Prompt engineering</strong>
              </Link>{" "}
              with a versioned prompt registry, lineage tracking, and{" "}
              <Link
                to={`${MLFLOW_GENAI_DOCS_URL}prompt-registry/optimize-prompts/`}
              >
                automatic optimization
              </Link>{" "}
              that improves prompts based on evaluation results
            </li>
            <li>
              <strong>Fine-tuning and reinforcement learning</strong> support
              with Transformers, PEFT, Unsloth, and TRL
            </li>
          </ul>
          <div className="screenshot-wrap">
            <img
              src={MlflowEvalUI}
              alt="MLflow evaluation UI showing scorers, results, and detailed assessment views"
            />
          </div>

          {/* Summary */}
          <h2>Summary</h2>
          <p>
            <strong>
              Choose <Link href="https://www.litellm.ai/">LiteLLM</Link>
            </strong>{" "}
            if you need a <strong>dedicated LLM proxy</strong> for model
            routing, use long-tail LLM providers that only LiteLLM supports, or
            are already integrated with <strong>LiteLLM's client SDK</strong> and
            proxy in your stack.
          </p>
          <p>
            <strong>
              Choose <Link to="/">MLflow</Link>
            </strong>{" "}
            if you need a <strong>secure AI gateway</strong> trusted by
            enterprises, with <strong>native observability</strong> and
            evaluation integrated with your gateway. MLflow covers the entire AI
            lifecycle: gateway, tracing, evaluation, prompt optimization, all in
            one platform. Teams who build production-grade LLM applications and
            agents should choose MLflow.
          </p>

          {/* Related Resources */}
          <h2>Related Resources</h2>
          <ul>
            <li>
              <Link to="/ai-gateway">What is AI Gateway?</Link>
            </li>
            <li>
              <Link to="/ai-platform">What is AI Platform?</Link>
            </li>
            <li>
              <Link to={`${MLFLOW_GENAI_DOCS_URL}gateway/`}>
                MLflow AI Gateway Guide
              </Link>
            </li>
            <li>
              <Link to={`${MLFLOW_GENAI_DOCS_URL}tracing/quickstart/`}>
                MLflow Tracing Quickstart
              </Link>
            </li>
            <li>
              <Link to={MLFLOW_GENAI_DOCS_URL}>MLflow Documentation</Link>
            </li>
            <li>
              <Link href="https://docs.litellm.ai/docs/">
                LiteLLM Documentation
              </Link>
            </li>
            <li>
              <Link to="/llmops">LLMOps Guide</Link>
            </li>
          </ul>
        </div>

        <ArticleSidebar />
      </div>
    </>
  );
}
