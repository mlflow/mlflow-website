// Research date: 2026-03-31
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../components/Header/Header";
import { ArticleSidebar } from "../components/ArticleSidebar/ArticleSidebar";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import { CopyButton } from "../components/CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";

/* ───────── images ───────── */
import MlflowEvalUI from "@site/static/img/langfuse-alternative/mlflow-eval-ui.png";
import ArizePhoenixUI from "@site/static/img/arize-phoenix-alternative/arize-phoenix-ui.png";
import LangsmithEvalUI from "@site/static/img/langsmith-alternative/langsmith-eval-ui.png";
import RagasEvalUI from "@site/static/img/top-5-eval/ragas-eval-ui.png";
import DeepEvalUI from "@site/static/img/top-5-eval/deepeval-eval-ui.png";

/* ───────── data ───────── */

const quickComparisonRows: [string, string, string, string, string, string][] =
  [
    ["Capability", "MLflow", "DeepEval", "Ragas", "Arize Phoenix", "LangSmith"],
    ["Open Source", "✔️", "✔️", "✔️", "Partial (ELv2)", "No"],
    [
      "PyPI Downloads",
      "30M+/mo",
      "1.9M+/mo",
      "1M+/mo",
      "1M+/mo",
      "65M+/mo \u00b9",
    ],
    ["Dataset Management", "✔️", "SDK-only", "No", "✔️", "✔️"],
    ["Multi-Turn Evaluation", "✔️", "✔️", "✔️", "Limited", "✔️"],
    ["Conversation Simulation", "✔️", "✔️", "No", "No", "No"],
    ["Human Feedback Collection", "✔️", "SDK-only", "No", "✔️", "✔️"],
    [
      "LLM Judge Alignment",
      "✔️  (automated tuning)",
      "No",
      "No",
      "No",
      "Manual tuning with UI",
    ],
    ["Online Monitoring", "✔️", "No", "No", "✔️", "✔️"],
    ["Visualization", "✔️", "Requires Confident AI (paid)", "No", "✔️", "✔️"],
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
            <CopyButton code={code} />
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

export default function Top5AgentEvaluationFrameworks() {
  return (
    <>
      <Head>
        <title>Top 5 Agent Evaluation Frameworks in 2026 | MLflow</title>
        <meta
          name="description"
          content="Compare the best agent evaluation frameworks for testing, scoring, and improving AI agents. See how MLflow, DeepEval, Ragas, Arize Phoenix, and LangSmith stack up on metrics, multi-turn support, and CI/CD integration."
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
            content: '\u2212';
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
          <h1>Top 5 Agent Evaluation Frameworks in 2026</h1>
          <p className="subtitle">
            Shipping an AI agent without evaluation is like deploying code
            without tests. As agents grow more complex, with multi-step
            reasoning, tool use, and autonomous decision-making, you need
            evaluation frameworks that can score not just the final output but
            every step along the way. In this guide, we compare the top five
            agent evaluation frameworks and help you choose the right one for
            your team.
          </p>

          <div className="tldr-box">
            <span className="tldr-label">TL;DR</span>
            <p>
              <strong>
                <Link to="/">MLflow</Link>
              </strong>{" "}
              is our top pick. It has the broadest metric coverage, supports
              both rule-based and LLM judge custom metrics, and lets human
              reviewers label results so that automated judges improve
              automatically from that feedback.
            </p>
            <p>
              <strong>Alternatives:</strong>{" "}
              <Link to="https://deepeval.com/">DeepEval</Link> for pytest-style
              CI/CD testing, <Link to="https://docs.ragas.io/">Ragas</Link> for
              RAG-focused metrics,{" "}
              <Link to="https://phoenix.arize.com/">Arize Phoenix</Link> for
              teams extending their existing ML observability to LLM evaluation.
              All three also integrate natively with MLflow so their metrics can
              be used as plugins.
            </p>
          </div>

          {/* What to Look For */}
          <h2 id="what-to-look-for" data-toc="What to Look For">
            What to Look For in an Agent Evaluation Framework
          </h2>
          <p>
            Every framework on this list can score LLM outputs. The real
            question is how deeply they can evaluate agent behavior, and how
            well they integrate with the rest of your development workflow.
            Before comparing tools, here are the four capabilities that separate
            production-grade evaluation from basic output checking.
          </p>

          <h3>1. Flexible custom metrics</h3>
          <p>
            Built-in metrics are good start, but almost every real project
            requires some sort of custom evaluation criteria. The framework must
            make it easy to define both rule-based checks (format validation,
            tool call verification) and LLM judges (for nuanced dimensions like
            helpfulness and safety). Lacking this capability significantly
            limits the value of evaluation.
          </p>

          <h3>2. Seamless path from manual feedback to automated evaluation</h3>
          <p>
            Human feedback is the ground truth for agent quality. But manually
            reviewing every response does not scale. Look for frameworks that
            let you collect human labels on a sample of outputs and then
            automatically improve your automated judges to match those human
            assessments. This lets you start with human review and gradually
            shift to automated evaluation without sacrificing accuracy.
          </p>

          <h3>3. Multi-turn and conversation-level evaluation</h3>
          <p>
            Real agents are conversational. A single-turn eval setup that tests
            one request and one response misses most of how agents actually
            behave in production: clarifying questions, follow-ups, context
            carried across turns, and recovering from earlier mistakes. The
            framework should be able to evaluate a full conversation as a unit,
            score quality across turns, and ideally simulate synthetic
            conversations for test coverage beyond your labeled dataset.
          </p>

          <h3>4. A feedback loop between production and development</h3>
          <p>
            Test datasets are necessary but not sufficient. Production traffic
            behaves differently from your test set, models drift, and user
            behavior shifts over time. Look for frameworks that close the gap:
            production traces should be convertible into evaluation datasets,
            quality regressions should surface as actionable signals, and the
            cycle from "found a problem in production" to "verified the fix in
            staging" should be short.
          </p>

          <h3>5. Intuitive visualization</h3>
          <p>
            Numbers alone are not enough. When scores drop or an edge case
            surfaces, you need to quickly understand why. Look for frameworks
            that display trace timelines, per-turn scores, and metric breakdowns
            in a clear UI rather than requiring you to dig through JSON logs or
            build your own dashboards.
          </p>

          {/* Quick Comparison Table */}
          <h2 id="quick-comparison" data-toc="Quick Comparison">
            Quick Comparison
          </h2>
          <QuickComparisonTable rows={quickComparisonRows} />
          <p style={{ fontSize: "13px", marginTop: "-28px" }}>
            {"\u00b9"} LangSmith's PyPI count is undetermined because it is an
            automatic dependency of the <code>langchain</code> package.
          </p>

          {/* ───── 1. MLflow ───── */}
          <h2 id="mlflow" data-toc="1. MLflow">
            1. MLflow - Widest Metric Coverage Including Multi-Turn and Human
            Alignment
          </h2>
          <p>
            <strong>
              <Link to="/">MLflow</Link>
            </strong>{" "}
            is the most widely deployed open-source AI engineering platform, and
            its{" "}
            <Link to={`${MLFLOW_GENAI_DOCS_URL}eval-monitor/`}>
              evaluation system
            </Link>{" "}
            is designed specifically for the agent development loop. Unlike
            standalone evaluation libraries that score outputs in isolation,
            MLflow's{" "}
            <Link to={`${MLFLOW_GENAI_DOCS_URL}concepts/scorers/`}>
              scorer framework
            </Link>{" "}
            evaluates full execution traces, including tool calls, reasoning
            chains, and planning decisions. Combined with{" "}
            <Link to={`${MLFLOW_GENAI_DOCS_URL}tracing/`}>tracing</Link>,{" "}
            <Link
              to={`${MLFLOW_GENAI_DOCS_URL}prompt-registry/optimize-prompts/`}
            >
              prompt optimization
            </Link>
            , and <Link to="/ai-gateway">governance</Link>, MLflow provides the
            complete evaluation-to-improvement pipeline in one tool.
          </p>
          <div className="screenshot-wrap">
            <img
              src={MlflowEvalUI}
              alt="MLflow evaluation UI showing scorer results, metrics, and trace-level assessment"
            />
          </div>

          <h4 style={{ color: "black" }}>
            Trace-Aware Scorers That Evaluate the Full Agent Reasoning Loop
          </h4>
          <p>
            MLflow's evaluation API,{" "}
            <Link
              to={`${MLFLOW_GENAI_DOCS_URL}eval-monitor/running-evaluation/agents/`}
            >
              <code>mlflow.genai.evaluate()</code>
            </Link>
            , is designed to evaluate agents as they actually run. Scorers
            receive the complete execution trace, not just the final output, so
            they can assess tool selection, plan quality, logical consistency,
            execution efficiency, and plan adherence across the entire reasoning
            loop. MLflow includes built-in Agent GPA (Goal-Plan-Action) scorers
            for common agent evaluation patterns, and you can write custom
            scorers in Python for any domain-specific criteria. The evaluation
            harness runs your agent and scorers in parallel, recording all
            results as traces and feedback in the MLflow tracking server.
          </p>

          <h4 style={{ color: "black" }}>
            LLM Judge Alignment with Research-Backed Algorithms
          </h4>
          <p>
            Most evaluation frameworks let you define LLM judges but give you no
            way to verify they are actually calibrated to human judgment.
            MLflow's{" "}
            <Link
              to={`${MLFLOW_GENAI_DOCS_URL}prompt-registry/optimize-prompts/`}
            >
              judge alignment
            </Link>{" "}
            is built on research-backed algorithms including GEPA and MemAlign,
            which optimize judge prompts against your human labels so that
            automated scores track what reviewers actually care about. The
            result is automated evaluation you can trust, not just run.
          </p>

          <h4 style={{ color: "black" }}>
            Widest Metric Coverage with Native Library Integrations
          </h4>
          <p>
            MLflow natively integrates with the leading evaluation libraries as
            pluggable scorers: Ragas, DeepEval, Arize Phoenix, TruLens, and
            Guardrails AI. This means you can use the best metrics from each
            library within <code>mlflow.genai.evaluate()</code> without building
            custom glue code. Combined with MLflow's own built-in scorers and
            the ability to write custom scorers with the{" "}
            <Link to={`${MLFLOW_GENAI_DOCS_URL}concepts/scorers/`}>
              <code>@scorer</code> decorator
            </Link>
            , MLflow provides the widest metric coverage of any evaluation
            framework on this list. All scorer results are tracked, versioned,
            and visualized in the MLflow UI alongside traces and experiments.
          </p>

          <ProsConsTable
            pros={[
              "Fully open source (Apache 2.0) with Linux Foundation governance. No feature gating.",
              "Widest metric coverage: native integration with Ragas, DeepEval, Phoenix, TruLens, and Guardrails as pluggable scorers",
              "Built-in human feedback collection and automated LLM judge alignment",
            ]}
            cons={[
              "Broader platform means more initial concepts to learn than a single-purpose eval library",
              "Teams doing quick ad-hoc evaluation may not need the full platform capabilities",
            ]}
          />

          <div className="best-for">
            <strong>Best for:</strong> Teams that need the widest metric
            coverage for agent evaluation, with native integration for leading
            eval libraries, a unified interface for custom and LLM judge
            metrics, and built-in human feedback alignment. The only fully
            open-source platform that connects evaluation to prompt optimization
            and governance.
          </div>

          <div className="product-faq">
            <details>
              <summary>
                How does MLflow evaluate agent traces, not just outputs?
              </summary>
              <p>
                MLflow's scorer framework receives the complete execution trace
                from{" "}
                <Link to={`${MLFLOW_GENAI_DOCS_URL}tracing/`}>
                  MLflow Tracing
                </Link>
                , including every tool call, LLM invocation, and planning step.
                Built-in Agent GPA scorers evaluate tool selection, plan
                quality, logical consistency, and execution efficiency across
                the full agent trajectory. Custom scorers can access any part of
                the trace to implement domain-specific evaluation logic.
              </p>
            </details>
            <details>
              <summary>Can I use MLflow with other eval libraries?</summary>
              <p>
                Yes. MLflow integrates with Ragas, DeepEval, Arize Phoenix,
                TruLens, and Guardrails AI as evaluation libraries. You can use
                their metrics as scorers within{" "}
                <code>mlflow.genai.evaluate()</code>, combining the best metrics
                from multiple libraries while keeping results centralized in
                MLflow.
              </p>
            </details>
            <details>
              <summary>Does MLflow evaluation work in CI/CD?</summary>
              <p>
                Yes. <code>mlflow.genai.evaluate()</code> can be called from any
                Python script, including CI/CD pipelines. Evaluation results are
                logged to the MLflow tracking server, and you can set pass/fail
                thresholds programmatically to gate deployments based on
                evaluation scores.
              </p>
            </details>
            <details>
              <summary>
                What is prompt optimization and how does it connect to
                evaluation?
              </summary>
              <p>
                MLflow's{" "}
                <Link
                  to={`${MLFLOW_GENAI_DOCS_URL}prompt-registry/optimize-prompts/`}
                >
                  prompt optimization
                </Link>{" "}
                uses algorithms like GEPA and MIPRO to automatically find better
                prompts based on your evaluation criteria. After running
                evaluation and identifying quality issues, you feed the results
                into the optimizer, which explores prompt variations and selects
                the ones that score highest on your metrics.
              </p>
            </details>
          </div>

          {/* ───── 2. DeepEval ───── */}
          <h2 id="deepeval" data-toc="2. DeepEval">
            2. DeepEval - Pytest-Native Evaluation with 50+ Metrics
          </h2>
          <p>
            <strong>
              <Link to="https://deepeval.com/">DeepEval</Link>
            </strong>{" "}
            is an open-source LLM evaluation framework built by Confident AI
            that brings a pytest-native testing experience to agent evaluation.
            With 50+ research-backed metrics and a familiar testing interface,
            DeepEval makes it easy to add LLM evaluation to existing CI/CD
            workflows. The framework covers agents, chatbots, RAG, single-turn,
            multi-turn, and safety evaluation, all from a single library.
          </p>
          <div className="screenshot-wrap">
            <img
              src={DeepEvalUI}
              alt="DeepEval documentation showing evaluation framework features and getting started guide"
            />
          </div>

          <h4 style={{ color: "black" }}>
            Testing LLM Agents Like You Test Regular Code
          </h4>
          <p>
            DeepEval's defining feature is its pytest-native interface. You
            write evaluation tests using the same patterns and tooling that
            Python developers already know: <code>assert_test()</code> calls,
            test discovery, fixtures, and familiar CLI output. This means
            evaluation integrates naturally into CI/CD pipelines. Teams can run
            agent evaluation as part of their standard test suite, catching
            regressions on every pull request without introducing new tooling or
            workflows.
          </p>

          <h4 style={{ color: "black" }}>
            50+ Pre-built Metrics Covering Agents, RAG, and Safety
          </h4>
          <p>
            DeepEval ships with a broad library of 50+ metrics covering tool
            selection accuracy, planning quality, faithfulness, reasoning
            coherence, hallucination detection, answer relevancy, and safety.
            Metrics use LLM judges and NLP models that can run locally, keeping
            evaluation costs predictable. The framework covers single-turn,
            multi-turn, and agentic evaluation in one package.
          </p>

          <h4 style={{ color: "black" }}>
            Limited Platform Integration Without Confident AI
          </h4>
          <p>
            DeepEval itself is a Python library, not a platform. To visualize
            evaluation results, manage datasets, or collaborate across teams,
            you need the{" "}
            <Link to="https://www.confident-ai.com/">Confident AI</Link>{" "}
            platform, which starts at $19.99/user/month. DeepEval does not
            include tracing, prompt optimization, or governance capabilities.
            Teams using DeepEval for evaluation will need separate tools for
            observability and the rest of the agent development lifecycle.
          </p>

          <ProsConsTable
            pros={[
              "Open source (Apache 2.0) with pytest-native testing interface",
              "50+ research-backed metrics covering agents, RAG, chatbots, and safety",
              "50+ pre-built metrics covering agents, RAG, chatbots, and safety",
            ]}
            cons={[
              "Visualization and team collaboration require Confident AI ($19.99+/user/month)",
              "No built-in tracing, prompt optimization, or governance capabilities",
              "Not trace-aware; evaluation runs on provided data, not production traces",
            ]}
          />

          <div className="best-for">
            <strong>Best for:</strong> Teams that want to add agent evaluation
            to existing pytest CI/CD workflows with a rich library of pre-built
            metrics. Pairs well with a tracing platform like MLflow for
            production observability.
          </div>

          <div className="product-faq">
            <details>
              <summary>Is DeepEval really free?</summary>
              <p>
                The DeepEval library is free and open source under Apache 2.0.
                You can run all 50+ metrics locally without any paid
                subscription. The Confident AI platform, which adds
                visualization, dataset management, and team collaboration, has a
                free tier limited to 2 seats and 5 test runs per week. Paid
                plans start at $19.99/user/month.
              </p>
            </details>
            <details>
              <summary>
                How does DeepEval handle agent-specific evaluation?
              </summary>
              <p>
                DeepEval supports span-level evaluation that scores each step of
                an agent independently. You can evaluate tool selection
                accuracy, planning quality, step-level faithfulness, and
                reasoning coherence. The framework also supports multi-turn
                evaluation for conversational agents.
              </p>
            </details>
            <details>
              <summary>Can I use DeepEval with MLflow?</summary>
              <p>
                Yes. MLflow integrates with DeepEval as an evaluation library.
                You can use DeepEval's metrics as scorers within{" "}
                <code>mlflow.genai.evaluate()</code>, combining DeepEval's rich
                metric library with MLflow's trace-aware evaluation and platform
                capabilities.
              </p>
            </details>
            <details>
              <summary>
                What is the difference between DeepEval and Confident AI?
              </summary>
              <p>
                DeepEval is the open-source evaluation library that provides the
                metrics and testing logic. Confident AI is the commercial
                platform built by the same team that adds visualization, dataset
                management, production monitoring, and team collaboration on top
                of DeepEval.
              </p>
            </details>
          </div>

          {/* ───── 3. Ragas ───── */}
          <h2 id="ragas" data-toc="3. Ragas">
            3. Ragas - Research-Backed RAG and Agent Evaluation
          </h2>
          <p>
            <strong>
              <Link to="https://docs.ragas.io/">Ragas</Link>
            </strong>{" "}
            is an open-source evaluation framework that started as the standard
            for RAG evaluation and has expanded to cover agent evaluation as
            well. Born from an{" "}
            <Link to="https://arxiv.org/abs/2309.15217">
              EACL 2024 research paper
            </Link>
            , Ragas provides research-validated metrics for faithfulness, answer
            relevancy, context precision, agent goal accuracy, and tool call
            accuracy. It is a lightweight library with no platform dependency,
            making it easy to integrate into any evaluation workflow.
          </p>
          <div className="screenshot-wrap">
            <img
              src={RagasEvalUI}
              alt="Ragas documentation showing agent evaluation metrics including goal accuracy and tool call accuracy"
            />
          </div>

          <h4 style={{ color: "black" }}>
            The Academic Standard for RAG Evaluation
          </h4>
          <p>
            Ragas established many of the evaluation metrics that other
            frameworks have since adopted. Its core RAG metrics (faithfulness,
            answer relevancy, context precision, context recall) are
            well-documented in peer-reviewed research and widely cited in the
            community. For teams evaluating RAG pipelines, Ragas provides the
            most thoroughly validated set of metrics available, with clear
            mathematical definitions and known failure modes documented in the
            research literature.
          </p>

          <h4 style={{ color: "black" }}>
            Agent Evaluation with Goal and Tool Accuracy
          </h4>
          <p>
            Ragas has expanded beyond RAG to support agent evaluation with
            metrics like AgentGoalAccuracy (did the agent achieve the user's
            goal?), ToolCallAccuracy (did the agent call the right tools with
            the right parameters?), and TopicAdherence (did the agent stay on
            topic?). Multi-turn evaluation is supported through the
            MultiTurnSample class, which represents conversational interactions
            between humans, AI, and tools. The AspectCritic metric provides
            flexible evaluation of multi-turn conversations against custom
            criteria.
          </p>

          <h4 style={{ color: "black" }}>A Library, Not a Platform</h4>
          <p>
            Ragas is a Python library with no UI, no tracing backend, and no
            platform layer. This is both a strength and a limitation. It is easy
            to integrate into any workflow and has no vendor lock-in, but teams
            will need to build their own infrastructure for visualizing results,
            managing datasets, and tracking evaluation over time. Ragas metrics
            use LLM calls for scoring, which adds cost and latency to each
            evaluation run. The framework also does not provide trace-aware
            evaluation; you need to extract and format data from your tracing
            system before passing it to Ragas.
          </p>

          <ProsConsTable
            pros={[
              "Open source (Apache 2.0) with research-validated metrics from peer-reviewed papers",
              "Strong RAG evaluation metrics (faithfulness, relevancy, context precision)",
              "Lightweight library with no platform dependency or vendor lock-in",
            ]}
            cons={[
              "No UI, tracing, or platform layer; teams must build their own infrastructure",
              "Not trace-aware; requires manual data extraction from tracing systems",
              "LLM-based scoring adds cost and latency; no local model option",
            ]}
          />

          <div className="best-for">
            <strong>Best for:</strong> Teams that need academically validated
            RAG evaluation metrics and want a lightweight library they can
            integrate into existing pipelines. Pairs well with MLflow for
            tracing and platform capabilities.
          </div>

          <div className="product-faq">
            <details>
              <summary>Is Ragas only for RAG evaluation?</summary>
              <p>
                No. While Ragas started as a RAG evaluation framework, it now
                includes agent-specific metrics like AgentGoalAccuracy,
                ToolCallAccuracy, and TopicAdherence. It also supports
                multi-turn conversation evaluation. However, its RAG metrics
                remain the most mature and well-validated part of the library.
              </p>
            </details>
            <details>
              <summary>Can I use Ragas with MLflow?</summary>
              <p>
                Yes. MLflow integrates with Ragas as an evaluation library. You
                can use Ragas metrics as scorers within{" "}
                <code>mlflow.genai.evaluate()</code>, combining Ragas's
                research-backed metrics with MLflow's trace-aware evaluation and
                platform capabilities.
              </p>
            </details>
            <details>
              <summary>How much does Ragas cost to run?</summary>
              <p>
                The Ragas library itself is free under Apache 2.0. However, most
                metrics use LLM calls for scoring, so each evaluation run incurs
                API costs from your LLM provider. The cost depends on the number
                of test cases, the metrics you use, and the LLM provider
                pricing.
              </p>
            </details>
            <details>
              <summary>Does Ragas support custom metrics?</summary>
              <p>
                Yes. Ragas supports custom metrics through LLM-based judges. You
                can define custom evaluation criteria using natural language and
                use the AspectCritic metric for flexible, criteria-based
                evaluation of any aspect of agent behavior.
              </p>
            </details>
          </div>

          {/* ───── 4. Arize Phoenix ───── */}
          <h2 id="arize-phoenix" data-toc="4. Arize Phoenix">
            4. Arize Phoenix - Observability-First Evaluation
          </h2>
          <p>
            <strong>
              <Link to="https://phoenix.arize.com/">Arize Phoenix</Link>
            </strong>{" "}
            is an observability tool built by Arize AI that includes a strong
            evaluation layer. Phoenix combines distributed tracing with 50+
            built-in evaluation metrics, so you can score traces directly within
            the same platform you use for debugging. Its evaluation system
            supports LLM-based evaluators, code-based checks, and human labels,
            with integration support for third-party libraries like Ragas and
            DeepEval.
          </p>
          <div className="screenshot-wrap">
            <img
              src={ArizePhoenixUI}
              alt="Arize Phoenix UI showing traces, evaluation metrics, and agent analysis"
            />
          </div>

          <h4 style={{ color: "black" }}>
            Evaluate Traces Directly Within Your Observability Tool
          </h4>
          <p>
            Phoenix's key advantage for evaluation is that it can score traces
            and spans directly within the observability UI. You do not need to
            export data to a separate evaluation tool. Built-in evaluators cover
            hallucination detection, faithfulness, relevance, safety, and
            toxicity, and you can attach evaluation scores to individual spans
            for fine-grained analysis. The evaluation system supports both
            pre-built and custom evaluators, and integrates with third-party
            libraries like Ragas, DeepEval, and Cleanlab.
          </p>

          <h4 style={{ color: "black" }}>
            Familiar Ground for Teams Coming from Traditional ML Monitoring
          </h4>
          <p>
            Arize is a long-standing ML observability platform, and Phoenix
            carries that lineage. Teams that already use Arize for monitoring
            classical ML models will find the evaluation workflow familiar:
            attach evaluators to traces, score data quality dimensions, and
            track drift over time. The transition from traditional ML monitoring
            to LLM evaluation requires less ramp-up than switching to a
            purpose-built LLM evaluation tool from scratch.
          </p>

          <h4 style={{ color: "black" }}>
            Source-Available License and Platform Gaps
          </h4>
          <p>
            Phoenix uses the Elastic License 2.0 (ELv2), which restricts
            offering the software as a managed service. High-value features like
            the Alyx Copilot and online evaluations are gated behind paid plans
            on the commercial Arize AX platform. Phoenix does not offer prompt
            optimization or governance capabilities, and multi-turn evaluation
            support is more limited than dedicated evaluation libraries. The
            free tier retains data for only 7 days, and scaling beyond
            single-node deployments requires the commercial platform.
          </p>

          <ProsConsTable
            pros={[
              "Evaluate traces directly within the observability UI, no data export needed",
              "50+ built-in evaluation metrics with support for Ragas, DeepEval, and Cleanlab",
              "Natural fit for teams already using Arize for traditional ML monitoring",
            ]}
            cons={[
              "Elastic License 2.0 restricts use as a managed service",
              "High-value features (Alyx Copilot, online eval) gated behind paid plans",
              "No multi-turn or conversation-level evaluation support",
            ]}
          />

          <div className="best-for">
            <strong>Best for:</strong> Teams already using Arize for traditional
            ML monitoring who want to extend their existing observability
            workflow to cover LLM and agent evaluation. Phoenix metrics are also
            available as pluggable scorers in MLflow.
          </div>

          <div className="product-faq">
            <details>
              <summary>Is Arize Phoenix open source?</summary>
              <p>
                Phoenix is source-available under the Elastic License 2.0
                (ELv2), which allows free use but restricts offering the
                software as a managed service. This is a more restrictive
                license than Apache 2.0 or MIT. The commercial Arize AX platform
                adds features not available in the open-source version.
              </p>
            </details>
            <details>
              <summary>
                Can Phoenix evaluate agent traces, not just outputs?
              </summary>
              <p>
                Yes. Phoenix can attach evaluation scores to individual spans
                within a trace, so you can evaluate tool calls, retrieval steps,
                and reasoning independently. This provides more granular
                evaluation than output-only scoring.
              </p>
            </details>
            <details>
              <summary>
                How does Phoenix compare to MLflow for evaluation?
              </summary>
              <p>
                Both Phoenix and MLflow support trace-aware evaluation. Phoenix
                focuses on built-in metrics within the observability UI, while
                MLflow provides a broader evaluation platform with scorer
                framework, multi-turn evaluation, LLM judge alignment with human
                feedback, and automated prompt optimization based on evaluation
                results. MLflow also integrates with Phoenix as an evaluation
                library.
              </p>
            </details>
            <details>
              <summary>What is the free tier data retention?</summary>
              <p>
                The free self-hosted Phoenix retains data for 7 days. The
                commercial Arize AX platform offers 15 days on Pro and longer
                retention on Enterprise plans.
              </p>
            </details>
          </div>

          {/* ───── 5. LangSmith ───── */}
          <h2 id="langsmith" data-toc="5. LangSmith">
            5. LangSmith - LangChain's Commercial Evaluation Platform
          </h2>
          <p>
            <strong>
              <Link to="https://www.langchain.com/evaluation">LangSmith</Link>
            </strong>{" "}
            is the commercial evaluation platform built by LangChain. It
            provides multiple evaluator types (human, heuristic, LLM-as-judge,
            pairwise comparison), agent trajectory evaluation, and both offline
            and online evaluation modes. LangSmith is most polished for teams
            building on LangChain and LangGraph, though it supports other
            frameworks through OpenTelemetry ingestion.
          </p>
          <div className="screenshot-wrap">
            <img
              src={LangsmithEvalUI}
              alt="LangSmith evaluation UI showing evaluation runs, scores, and dataset management"
            />
          </div>

          <h4 style={{ color: "black" }}>
            Agent Trajectory Evaluation and Annotation Queues
          </h4>
          <p>
            LangSmith can capture the full trajectory of an agent's steps, tool
            calls, and reasoning, and define evaluators that score intermediate
            decisions. This helps teams debug complex agent workflows and
            pinpoint where things went wrong. Annotation queues provide
            structured human review, where domain experts can label traces and
            build evaluation datasets from real production data.
          </p>

          <h4 style={{ color: "black" }}>
            Polly AI Assistant for Automated Trace Analysis
          </h4>
          <p>
            LangSmith includes Polly, an AI assistant that analyzes trace data
            in natural language. You can ask Polly to summarize failure
            patterns, identify common error types, and prioritize improvements
            by frequency and impact, without writing any evaluation code.
            Complementing this are topic clustering and an insights agent that
            automatically categorize agent behavior across production runs.
            These features are available on Plus and Enterprise plans.
          </p>

          <h4 style={{ color: "black" }}>
            Proprietary Platform with Per-Seat Pricing
          </h4>
          <p>
            LangSmith is a proprietary SaaS platform with no self-hosted option
            outside enterprise contracts. The free tier is limited to 5,000
            traces per month with 14-day retention. The Plus plan costs
            $39/seat/month, and extended retention (400 days) is a paid add-on
            at $5/1k traces. Per-seat pricing can limit access for PMs, QA
            engineers, and other stakeholders who need to review evaluation
            results but may not justify a full seat cost.
          </p>

          <ProsConsTable
            pros={[
              "Agent trajectory evaluation with structured human annotation queues",
              "Polly AI assistant for natural-language trace analysis without writing eval code",
              "Deep integration with LangChain and LangGraph ecosystem",
            ]}
            cons={[
              "Proprietary SaaS with no self-hosted option outside enterprise contracts",
              "Per-seat pricing ($39/seat/month) limits collaboration across teams",
              "Feature parity lags for integrations outside the LangChain ecosystem",
            ]}
          />

          <div className="best-for">
            <strong>Best for:</strong> Teams 100% committed to the
            LangChain/LangGraph ecosystem who are comfortable with proprietary
            SaaS pricing and do not require self-hosting.
          </div>

          <div className="product-faq">
            <details>
              <summary>
                Does LangSmith evaluation work outside LangChain?
              </summary>
              <p>
                LangSmith supports other frameworks via OpenTelemetry ingestion
                and a <code>traceable</code> wrapper. However, the evaluation
                experience is most polished for LangChain and LangGraph
                applications. Teams using other frameworks may find some
                features less integrated.
              </p>
            </details>
            <details>
              <summary>Can I self-host LangSmith?</summary>
              <p>
                Self-hosting is only available on the Enterprise tier. The free,
                Plus, and Team plans are SaaS-only with data stored on
                LangChain's infrastructure.
              </p>
            </details>
            <details>
              <summary>How does LangSmith pricing work for evaluation?</summary>
              <p>
                The free tier includes 5,000 traces/month with 14-day retention.
                Plus is $39/seat/month with higher trace limits. Extended
                retention (400 days) costs $5/1k traces as a paid add-on.
                Enterprise pricing is custom. Evaluation runs count toward your
                trace quota.
              </p>
            </details>
            <details>
              <summary>
                What types of evaluators does LangSmith support?
              </summary>
              <p>
                LangSmith supports human evaluation through annotation queues,
                heuristic checks (output validation, code compilation),
                LLM-as-judge evaluators that score against custom criteria, and
                pairwise comparison evaluators for A/B testing agent versions.
                Custom evaluators can be written in Python or TypeScript.
              </p>
            </details>
          </div>

          {/* How to Choose */}
          <h2 id="how-to-choose" data-toc="How to Choose">
            How to Choose the Right Framework
          </h2>
          <p>
            All five frameworks on this list can evaluate agent outputs. The
            difference lies in how flexible they are for your specific needs,
            and how well they support the full evaluation lifecycle from custom
            metrics to human feedback alignment.
          </p>

          <h4 style={{ color: "black" }}>
            How Much Metric Coverage Do You Actually Need?
          </h4>
          <p>
            Libraries like DeepEval and Ragas ship with broad pre-built metric
            sets, but pre-built metrics rarely cover everything. Evaluate
            whether the framework makes it easy to add custom rule-based checks
            and LLM judges, and whether those custom metrics are treated as
            first-class objects with version management and reuse. The depth of
            custom metric support matters more than the number of pre-built ones
            as your evaluation suite matures.
          </p>

          <h4 style={{ color: "black" }}>
            Do You Need a Library or a Platform?
          </h4>
          <p>
            If you just need metrics you can run in a script or CI pipeline,
            DeepEval and Ragas are excellent standalone libraries. If you need
            evaluation connected to tracing, visualization, online monitoring,
            and human feedback, you need a platform. Consider whether you want
            to assemble those capabilities yourself or use something that
            provides them out of the box.
          </p>

          <h4 style={{ color: "black" }}>
            How Will You Calibrate Your Automated Judges?
          </h4>
          <p>
            LLM judges are only as reliable as their alignment with human
            expectations. Without a way to collect human labels on a sample of
            outputs and adjust your judges accordingly, you risk optimizing for
            metrics that do not reflect real quality. Check whether the
            framework provides built-in human feedback collection and a
            mechanism to improve judges automatically from those labels.
          </p>

          {/* Our Recommendation */}
          <h2 id="recommendation" data-toc="Recommendation">
            Our Recommendation
          </h2>
          <p>
            For teams building production agents,{" "}
            <strong>
              <Link to="/">MLflow</Link>
            </strong>{" "}
            is our top recommendation. It provides the widest metric coverage
            through native integration with DeepEval, Ragas, Arize Phoenix,
            TruLens, and Guardrails AI as pluggable scorers. Its unified{" "}
            <code>@scorer</code> decorator and <code>make_judge()</code> API
            make it easy to define, version, and manage custom metrics. And its
            built-in human feedback collection with the <code>align()</code> API
            for automatic LLM judge alignment is a capability you will not find
            in any other framework on this list. All of this is fully open
            source under the Apache 2.0 license, backed by the Linux Foundation.
          </p>

          <h4 style={{ color: "black" }}>Alternatives Worth Considering</h4>
          <p>
            <strong>DeepEval</strong> is a good choice for teams that want
            pytest integration with a large pre-built metric library.{" "}
            <strong>Ragas</strong> is the standard for academically-validated
            RAG metrics. <strong>Arize Phoenix</strong> suits teams already
            running Arize for traditional ML who want to extend the same
            workflow to LLM evaluation. All three integrate natively with MLflow
            as pluggable metric libraries.
          </p>

          {/* Global FAQ */}
          <h2 id="faq" data-toc="FAQ">
            Frequently Asked Questions
          </h2>

          <h3>What is agent evaluation?</h3>
          <p>
            Agent evaluation is the process of systematically scoring how well
            an AI agent performs its tasks. Unlike traditional LLM evaluation,
            which focuses on output quality, agent evaluation also assesses
            intermediate steps: tool selection, reasoning chains, planning
            quality, and the overall trajectory from goal to completion. Good
            agent evaluation catches problems that output-only scoring misses,
            like an agent that arrives at the right answer through incorrect
            reasoning. Learn more on our{" "}
            <Link to="/llm-evaluation">LLM evaluation</Link> page.
          </p>

          <h3>
            What is the difference between output evaluation and trace-aware
            evaluation?
          </h3>
          <p>
            Output evaluation scores the final result of an agent against a
            reference or quality criteria. Trace-aware evaluation goes deeper,
            scoring every step the agent took, including tool calls, LLM
            invocations, and planning decisions. Trace-aware evaluation can
            identify the specific step where an agent went wrong, while output
            evaluation can only tell you that the final result was incorrect.
            See how MLflow implements trace-aware evaluation in the{" "}
            <Link to={`${MLFLOW_GENAI_DOCS_URL}eval-monitor/`}>
              Evaluation and Monitoring docs
            </Link>
            .
          </p>

          <h3>Can I use multiple evaluation frameworks together?</h3>
          <p>
            Yes, but managing each library's own interface, data format, and
            result storage quickly becomes a mess. The practical approach is to
            use a platform that integrates with multiple metric libraries
            through a single unified interface, so you get
            academically-validated RAG metrics from Ragas and pytest-native
            agent metrics from DeepEval without juggling separate workflows.
          </p>

          <h3>How do I align LLM judges with human feedback?</h3>
          <p>
            LLM judge alignment is the process of calibrating automated judges
            so their scores match what human reviewers actually care about. You
            collect a sample of human labels, then use an optimization algorithm
            to adjust the judge prompt until its outputs correlate with those
            human assessments. Research algorithms like{" "}
            <a href="https://www.databricks.com/blog/memalign-building-better-llm-judges-human-feedback-scalable-memory">
              MemAlign
            </a>{" "}
            and GEPA formalize this process, making judge calibration
            reproducible and measurable rather than ad hoc.
          </p>

          <h3>How do I evaluate multi-turn agent conversations?</h3>
          <p>
            Multi-turn evaluation scores agent behavior across a full
            conversation, not just a single request-response pair. This means
            tracking whether the agent carries context correctly across turns,
            asks good clarifying questions, recovers from earlier mistakes, and
            reaches the user's goal by the end of the session. Among the
            frameworks on this list, MLflow, DeepEval, and Ragas all support
            multi-turn evaluation; Arize Phoenix does not. See the{" "}
            <a href="https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/multi-turn/">
              multi-turn evaluation guide
            </a>{" "}
            for practical examples.
          </p>

          <h3>
            Should I use LLM judges or deterministic metrics for agent
            evaluation?
          </h3>
          <p>
            Both have their place. Deterministic metrics (exact match, regex,
            code compilation) are fast, cheap, and reproducible, making them
            ideal for CI/CD gates. LLM judges handle nuanced quality dimensions
            like helpfulness, safety, and faithfulness that are hard to capture
            with rules. Most teams use deterministic metrics for basic
            correctness checks and LLM judges for higher-level quality
            assessment.
          </p>

          <h3>How often should I run agent evaluations?</h3>
          <p>
            At minimum, run offline evaluation on every code change that touches
            agent logic: prompt changes, tool additions, and model swaps. For
            production agents, online evaluation on a sample of live traffic
            catches quality drift that offline tests miss, because production
            queries behave differently from curated test sets.
          </p>

          {/* Related Resources */}
          <h2>Related Resources</h2>
          <div className="related-resources">
            <ul>
              <li>
                <Link to="/llm-evaluation">What is LLM Evaluation?</Link>
              </li>
              <li>
                <Link to={`${MLFLOW_GENAI_DOCS_URL}eval-monitor/quickstart/`}>
                  Evaluation Quickstart
                </Link>
              </li>
              <li>
                <Link to="/top-5-agent-observability-tools">
                  Top 5 Agent Observability Tools
                </Link>
              </li>
              <li>
                <Link to="/arize-phoenix-alternative">
                  MLflow vs Arize Phoenix: Detailed Comparison
                </Link>
              </li>
              <li>
                <Link to="/langsmith-alternative">
                  MLflow vs LangSmith: Detailed Comparison
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
