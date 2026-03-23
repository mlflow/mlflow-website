import { type ReactNode, useState } from "react";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../components/Header/Header";
import { MLFLOW_DOCS_URL, MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import { CopyButton } from "../components/CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";
import MlflowUIImg from "@site/static/img/arize-phoenix-alternative/mlflow-ui.png";
import PhoenixUIImg from "@site/static/img/arize-phoenix-alternative/arize-phoenix-ui.png";
import MlflowLogo from "@site/static/img/arize-phoenix-alternative/mlflow-logo.png";
import PhoenixLogo from "@site/static/img/arize-phoenix-alternative/arize_phoenix_logo.png";

const SEO_TITLE =
  "Arize Phoenix Alternative for LLMs & Agents | MLflow Agent Platform";
const SEO_DESCRIPTION =
  "Compare MLflow vs Arize Phoenix for LLM observability, agent observability, tracing, and evaluation with a comprehensive, open source agent engineering and ops platform.";
const CANONICAL_URL = "https://mlflow.org/arize-phoenix-alternative";

const tracingExamples: { label: string; mlflow: string; phoenix: string }[] = [
  {
    label: "LangChain",
    mlflow: `import mlflow

mlflow.langchain.autolog()

# That's it. Chain and tool calls are
# captured automatically.`,
    phoenix: `from phoenix.otel import register
from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
)

tracer_provider = register(project_name="my-app")
LangChainInstrumentor().instrument(
    tracer_provider=tracer_provider
)

# Continue running your app as usual`,
  },
  {
    label: "OpenAI",
    mlflow: `import mlflow

mlflow.openai.autolog()

# That's it. All OpenAI calls are
# captured automatically.`,
    phoenix: `from phoenix.otel import register
from openinference.instrumentation.openai import (
    OpenAIInstrumentor,
)

tracer_provider = register(project_name="my-app")
OpenAIInstrumentor().instrument(
    tracer_provider=tracer_provider
)

# Continue running your app as usual`,
  },
  {
    label: "DSPy",
    mlflow: `import mlflow

mlflow.dspy.autolog()

# That's it. Every DSPy module call
# is captured automatically.`,
    phoenix: `from phoenix.otel import register
from openinference.instrumentation.dspy import (
    DSPyInstrumentor,
)

tracer_provider = register(project_name="my-app")
DSPyInstrumentor().instrument(
    tracer_provider=tracer_provider
)

# Continue running your app as usual`,
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

function PhoenixCodeLabel() {
  return (
    <span className="code-logo-inline">
      <img src={PhoenixLogo} alt="Arize Phoenix" className="code-logo-icon" />
      Arize Phoenix
    </span>
  );
}

function CodeBlock({ code, label }: { code: string; label: string }) {
  return (
    <div className="code-side">
      <div className="code-side-header">
        <span className="code-side-label">
          {label === "MLflow" ? <MlflowCodeLabel /> : <PhoenixCodeLabel />}
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
  );
}

function CodeTabs({
  tabs,
}: {
  tabs: { label: string; mlflow: string; phoenix: string }[];
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
        <CodeBlock code={tabs[active].phoenix} label="Arize Phoenix" />
      </div>
    </div>
  );
}

const architectureTable: [string, string, string][] = [
  ["Dimension", "MLflow", "Arize Phoenix"],
  [
    "Primary Scope",
    "End-to-end AI engineering platform",
    "LLM observability and evaluation workbench",
  ],
  [
    "Core Data Plane",
    "Tracking server + SQL backend + artifact store",
    "Collector + UI + SQLite or PostgreSQL",
  ],
  [
    "Operational Focus",
    "Unified platform for agents, LLMs, and ML models",
    "Trace-centric debugging and iteration loop",
  ],
];

const capabilityRows: {
  feature: string;
  mlflow: boolean | string;
  phoenix: boolean | string;
}[] = [
  { feature: "Built-in LLM Judges", mlflow: true, phoenix: true },
  { feature: "Custom Metrics", mlflow: true, phoenix: true },
  { feature: "Versioning Metrics", mlflow: true, phoenix: false },
  {
    feature: "Aligning Judges with Human Feedback",
    mlflow: true,
    phoenix: false,
  },
  { feature: "Multi-Turn Evaluation", mlflow: true, phoenix: false },
  { feature: "Visualization & Comparison", mlflow: true, phoenix: false },
  {
    feature: "Online Evaluation",
    mlflow: true,
    phoenix: "Arize AX (paid SaaS) only",
  },
  {
    feature: "Evaluation Library Integration",
    mlflow: "RAGAS, DeepEval, TruLens, Guardrails AI",
    phoenix: "Custom integrations only",
  },
  {
    feature: "License",
    mlflow: "Apache 2.0",
    phoenix: "Elastic License 2.0",
  },
];

export default function ArizePhoenixAlternative() {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(null);

  const faqs: { question: string; answer: ReactNode; answerText: string }[] = [
    {
      question: "What is the best open source Arize Phoenix alternative?",
      answer: (
        <>
          MLflow is a strong open source alternative when you need both{" "}
          <Link to="/llm-tracing">LLM tracing</Link> and lifecycle capabilities
          such as evaluation, prompt management, AI Gateway, and governance
          workflows.
        </>
      ),
      answerText:
        "MLflow is a strong open source alternative when you need both LLM tracing and lifecycle capabilities such as evaluation, prompt management, AI Gateway, and governance workflows.",
    },
    {
      question: "How does MLflow differ from Arize Phoenix?",
      answer: (
        <>
          Arize Phoenix is primarily a trace-first observability and evaluation
          workspace. MLflow covers observability plus broader workflows across{" "}
          <Link to={MLFLOW_DOCS_URL}>tracking</Link>, evaluation, prompt
          management, AI Gateway, and governance for AI agents and LLM
          applications.
        </>
      ),
      answerText:
        "Arize Phoenix is primarily a trace-first observability and evaluation workspace. MLflow covers observability plus broader workflows across tracking, evaluation, prompt management, AI Gateway, and governance for AI agents and LLM applications.",
    },
    {
      question: "Does MLflow support OpenTelemetry traces?",
      answer: (
        <>
          Yes. MLflow supports OpenTelemetry-aligned tracing workflows and
          integrations for multiple frameworks. See{" "}
          <Link to={`${MLFLOW_GENAI_DOCS_URL}tracing/`}>
            MLflow tracing docs
          </Link>{" "}
          for setup details.
        </>
      ),
      answerText:
        "Yes. MLflow supports OpenTelemetry-aligned tracing workflows and integrations for multiple frameworks. See MLflow tracing docs for setup details.",
    },
    {
      question:
        "Can I use MLflow for both LLM applications and traditional ML workloads?",
      answer: (
        <>
          Yes. MLflow is designed for both classical ML and AI agent / LLM
          systems, so teams can manage experiments, model versions, and
          evaluations in one platform.
        </>
      ),
      answerText:
        "Yes. MLflow is designed for both classical ML and AI agent / LLM systems, so teams can manage experiments, model versions, and evaluations in one platform.",
    },
    {
      question: "Is Arize Phoenix better for trace-centric debugging?",
      answer: (
        <>
          Many teams choose Phoenix specifically for trace-first UI workflows.
          MLflow also provides strong trace visibility, with additional
          evaluation, AI Gateway, and governance features for teams that need
          broader coverage.
        </>
      ),
      answerText:
        "Many teams choose Phoenix specifically for trace-first UI workflows. MLflow also provides strong trace visibility, with additional evaluation, AI Gateway, and governance features for teams that need broader coverage.",
    },
    {
      question: "Does MLflow include an AI Gateway?",
      answer: (
        <>
          Yes. MLflow includes a built-in{" "}
          <Link to="/ai-gateway">AI Gateway</Link> for governing LLM access
          across your organization, with rate limiting, fallbacks, usage
          tracking, and credential management. Arize Phoenix does not offer a
          gateway capability.
        </>
      ),
      answerText:
        "Yes. MLflow includes a built-in AI Gateway for governing LLM access across your organization, with rate limiting, fallbacks, usage tracking, and credential management. Arize Phoenix does not offer a gateway capability.",
    },
    {
      question: "How do I evaluate LLM and agent quality in MLflow?",
      answer: (
        <>
          Use <Link to="/genai/evaluations">MLflow evaluations</Link> for
          offline and online quality checks, and combine with tracing for
          debugging regressions and prompt changes.
        </>
      ),
      answerText:
        "Use MLflow evaluations for offline and online quality checks, and combine with tracing for debugging regressions and prompt changes.",
    },
    {
      question: "Is MLflow vendor-neutral for enterprise use?",
      answer: (
        <>
          MLflow is backed by the{" "}
          <Link to="https://www.linuxfoundation.org/">Linux Foundation</Link>, a
          non-profit vendor-neutral organization. It is licensed under Apache
          2.0 with full feature parity between its open source release and
          managed offerings, ensuring long-term portability.
        </>
      ),
      answerText:
        "MLflow is backed by the Linux Foundation, a non-profit vendor-neutral organization. It is licensed under Apache 2.0 with full feature parity between its open source release and managed offerings, ensuring long-term portability.",
    },
    {
      question: "Which teams should choose Arize Phoenix vs MLflow?",
      answer: (
        <>
          Choose Phoenix if your immediate priority is dedicated trace-centric
          observability and you prefer a SaaS-first experience. Choose MLflow if
          you need a complete AI engineering platform with evaluation, AI
          Gateway, and long-term platform consolidation.
        </>
      ),
      answerText:
        "Choose Phoenix if your immediate priority is dedicated trace-centric observability and you prefer a SaaS-first experience. Choose MLflow if you need a complete AI engineering platform with evaluation, AI Gateway, and long-term platform consolidation.",
    },
    {
      question: "Where can I get started quickly with MLflow tracing?",
      answer: (
        <>
          Start with the{" "}
          <Link to={`${MLFLOW_GENAI_DOCS_URL}tracing/quickstart/`}>
            MLflow tracing quickstart
          </Link>{" "}
          and then expand to{" "}
          <Link to={`${MLFLOW_GENAI_DOCS_URL}evaluate/`}>
            evaluation workflows
          </Link>
          .
        </>
      ),
      answerText:
        "Start with the MLflow tracing quickstart and then expand to evaluation workflows.",
    },
  ];
  const faqSchema = {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    mainEntity: faqs.map((faq) => ({
      "@type": "Question",
      name: faq.question,
      acceptedAnswer: {
        "@type": "Answer",
        text: faq.answerText,
      },
    })),
  };

  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="article" />
        <link rel="canonical" href={CANONICAL_URL} />
        <script type="application/ld+json">{JSON.stringify(faqSchema)}</script>
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
            width: 42%;
          }
          .eval-checklist thead th:not(:first-child) {
            text-align: center;
            width: 29%;
          }
          .eval-checklist tbody td {
            padding: 12px 16px;
            border-bottom: 1px solid #f0f0f0;
            color: #3d3d3d;
          }
          .eval-checklist tbody td:not(:first-child) {
            text-align: center;
            font-size: 16px;
          }
          .eval-checklist tbody tr:hover {
            background: #f9fafb;
          }

          .tip-note {
            background: #f0f9ff;
            border-left: 4px solid #0194e2;
            border-radius: 4px;
            padding: 16px 20px;
            margin: 24px 0 40px 0;
            font-family: 'DM Sans', sans-serif;
            font-size: 15px;
            color: #1a1a1a;
            line-height: 1.6;
          }
          .tip-note strong {
            color: #0072b0;
          }

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
          .faq-list {
            margin: 32px 0 40px;
          }
          .faq-item {
            border-bottom: 1px solid #e5e7eb;
          }
          .faq-question {
            width: 100%;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            background: none;
            border: none;
            font-size: 18px;
            font-weight: 600;
            color: #1a1a1a;
            cursor: pointer;
            text-align: left;
          }
          .faq-chevron {
            transition: transform 0.2s ease;
            color: #6b7280;
            font-size: 16px;
            margin-left: 24px;
          }
          .faq-chevron.open {
            transform: rotate(180deg);
          }
          .faq-answer {
            padding-bottom: 20px;
            color: #3d3d3d;
            line-height: 1.7;
          }
          .faq-answer a {
            font-weight: 600;
            color: #0194e2 !important;
            text-decoration: none;
          }
          .faq-answer a:hover {
            text-decoration: underline;
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
          <h1>
            Open Source Arize Phoenix Alternative? Arize Phoenix vs MLflow
          </h1>
          <p className="subtitle">
            Arize Phoenix and MLflow are open source platforms that help teams
            build production-grade AI agents and LLM applications. In this guide,
            we compare Phoenix's tracing-focused approach with MLflow's complete
            AI engineering platform and help you decide which is the right fit.
          </p>
          <h2 id="what-is-arize-phoenix">What is Arize Phoenix?</h2>
          <div className="screenshot-wrap">
            <img
              src={PhoenixUIImg}
              alt="Arize Phoenix trace and evaluation workspace screenshot"
            />
          </div>
          <p>
            <strong>
              <Link to="https://arize.com/docs/phoenix">Arize Phoenix</Link>
            </strong>{" "}
            is an open source AI observability and evaluation tool designed for
            LLM and agent applications. Phoenix focuses on trace inspection,
            span-level debugging, annotation workflows, and experiment loops for
            prompt and quality iteration. It is especially popular with teams
            that want OpenTelemetry-native instrumentation and a dedicated
            trace-centric UI for LLM development. Phoenix OSS is primarily
            designed for local development and debugging, while Arize's
            commercial SaaS offering — Arize AX (the paid, hosted tier built on
            top of Phoenix) — targets production-scale deployments with online
            evaluations, the Alyx Copilot, and enterprise integrations.
          </p>

          <h2 id="what-is-mlflow">What is MLflow?</h2>
          <div className="screenshot-wrap">
            <img
              src={MlflowUIImg}
              alt="MLflow tracing and evaluation workspace screenshot"
            />
          </div>
          <p>
            <strong>
              <Link to="/">MLflow</Link>
            </strong>{" "}
            is an open source AI engineering platform that enables teams of all
            sizes to debug, evaluate, monitor, and optimize production-quality
            AI agents, LLM applications, and ML models while controlling costs
            and managing access to models and data. With over 30 million monthly
            downloads, thousands of organizations rely on MLflow each day to
            ship AI to production with confidence.
          </p>

          <h2 id="quick-comparison">Quick Comparison</h2>
          <div className="tldr-grid">
            <div className="tldr-card highlight">
              <h3>
                <img src={MlflowLogo} alt="MLflow" className="tldr-logo" />
                Choose MLflow if you...
              </h3>
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
                  Want a unified solution for{" "}
                  <strong>managing and governing access</strong> to LLMs via an{" "}
                  <strong>AI Gateway</strong>
                </li>
                <li>
                  Prefer a permissive <strong>Apache 2.0</strong> licensing
                  model
                </li>
              </ul>
            </div>
            <div className="tldr-card">
              <h3>
                <img
                  src={PhoenixLogo}
                  alt="Arize Phoenix"
                  className="tldr-logo"
                />
                Choose Arize Phoenix if you...
              </h3>
              <ul>
                <li>
                  Primarily need{" "}
                  <strong>LLM tracing and basic evaluation workflows</strong>
                </li>
                <li>Prefer a SaaS-first experience</li>
                <li>
                  Already have another solution for{" "}
                  <strong>managing LLM access</strong>
                </li>
                <li>
                  Are comfortable with <strong>ELv2 licensing</strong> tradeoffs
                </li>
              </ul>
            </div>
          </div>

          <h2 id="open-source-licensing">Open Source &amp; Licensing</h2>
          <p>
            <strong>MLflow</strong> is an open source project backed by the{" "}
            <strong>
              <Link to="https://www.linuxfoundation.org/">
                Linux Foundation
              </Link>
            </strong>
            , a non-profit vendor-neutral organization, ensuring long-term
            community stewardship with no single company controlling its
            direction. MLflow is licensed under Apache 2.0 and maintains full
            feature parity between its open source release and managed
            offerings. With adoption by 60%+ of the Fortune 500, MLflow is one
            of the most widely deployed AI platforms in the enterprise.
          </p>
          <p>
            <strong>Arize Phoenix</strong> is distributed under Elastic License
            2.0. While Phoenix is free to self-host, ELv2 includes restrictions
            for offering the software as a managed hosted service. Arize is
            largely focused on their commercial SaaS offering (Arize AX), and
            some production capabilities such as online evaluations, the Alyx
            Copilot, and enterprise integrations are{" "}
            <Link to="https://arize.com/products-phoenix-versus-arize-ax">
              only available in the paid SaaS tier
            </Link>
            . Phoenix OSS is primarily designed for local development and
            debugging, while Arize AX targets production-scale deployments.
          </p>

          <h2 id="tracing-observability">Tracing &amp; Observability</h2>
          <p>
            Both platforms align around OpenTelemetry-compatible tracing. MLflow
            and Phoenix both support OpenTelemetry, but they differ in
            instrumentation ergonomics and production monitoring capabilities.
          </p>
          <p>
            <strong>MLflow</strong> auto-instruments 30+ frameworks with a{" "}
            <strong>one-line unified</strong> <code>autolog()</code> API,
            including OpenAI, LangGraph, DSPy, Anthropic, LangChain, Pydantic
            AI, CrewAI, and many more. MLflow uses the native OpenTelemetry data
            model (Trace + Span + Events) and supports both OTLP ingest and
            export so teams can integrate with broader observability
            infrastructure. Teams also track{" "}
            <Link to={`${MLFLOW_GENAI_DOCS_URL}tracing/token-usage-cost/`}>
              token usage and cost
            </Link>{" "}
            to connect observability with spend.
          </p>
          <p>
            <strong>Phoenix</strong> uses OpenInference/OpenTelemetry
            instrumentation with explicit tracer registration. The open source
            version of Phoenix does not offer online monitoring as part of its
            observability stack — teams that need production monitoring must
            upgrade to the{" "}
            <Link to="https://arize.com/docs/phoenix">paid SaaS version</Link>.
          </p>
          <CodeTabs tabs={tracingExamples} />

          <h2 id="evaluation-experimentation">
            Evaluation &amp; Experimentation
          </h2>
          <p>
            Evaluation is where the gap between MLflow and Arize Phoenix is most
            pronounced. Phoenix offers trace-driven evaluation loops — inspect
            traces, annotate outputs, build datasets, and run experiments — but
            high-value features such as online evaluations require the paid Arize AX tier, and the open
            source evaluation capabilities are less mature at scale.
          </p>
          <p>
            <strong>MLflow</strong> provides production-grade evaluation backed
            by a dedicated research team. It supports a rich set of built-in
            scorers, integration with leading evaluation libraries (RAGAS,
            DeepEval, TruLens, Guardrails AI), and advanced
            capabilities like multi-turn evaluation, online evaluation, and
            aligning LLM judges with human feedback. If your team needs to move
            beyond vibe checks to rigorous quality assurance, MLflow is
            purpose-built for it.
          </p>
          <table className="eval-checklist">
            <thead>
              <tr>
                <th>Capability</th>
                <th>MLflow</th>
                <th>Arize Phoenix</th>
              </tr>
            </thead>
            <tbody>
              {capabilityRows.map((row, i) => (
                <tr key={i}>
                  <td>{row.feature}</td>
                  <td>
                    {typeof row.mlflow === "string"
                      ? row.mlflow
                      : row.mlflow
                        ? "✅"
                        : "❌"}
                  </td>
                  <td>
                    {typeof row.phoenix === "string"
                      ? row.phoenix
                      : row.phoenix
                        ? "✅"
                        : "❌"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <h2 id="architecture-operation">Architecture &amp; Operation</h2>
          <p>
            Beyond individual features, MLflow and Arize Phoenix differ
            significantly in deployment model and operational scope.
          </p>
          <ComparisonTable rows={architectureTable} />
          <div className="tip-note">
            <strong>Running at enterprise scale?</strong> Teams on Databricks
            can use managed MLflow with platform integrations for governance,
            operational scale, and production monitoring. Trace and model data
            become queryable in governed data systems and are tied to enterprise
            access controls — while remaining fully portable with open source
            MLflow.
          </div>

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
            <strong>Arize Phoenix</strong> does not offer a gateway capability.
            To manage costs and model access, teams using Phoenix must bolt on a
            separate tool such as LiteLLM, PortKey, or build a custom gateway
            solution.
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
            . Teams can switch providers, add guardrails, or enforce usage
            policies without changing application code.
          </p>

          <h2 id="summary">Summary</h2>
          <p>
            <strong>Arize Phoenix is a solid observability tool</strong>, but
            tracing is only one piece of the puzzle. Its limited open source
            feature set and absence of an AI Gateway mean that teams inevitably
            need additional tools to build a complete AI engineering stack.{" "}
            <strong>Choose Phoenix</strong> if tracing and basic evaluation are
            all you need and you prefer a SaaS-first experience.
          </p>
          <p>
            <strong>MLflow is a complete AI engineering platform.</strong> It
            covers tracing, production-grade evaluation, prompt optimization, an
            AI Gateway, and governance, all backed by the Linux Foundation with
            full open source feature parity. <strong>Choose MLflow</strong> if
            you need a vendor-neutral platform that goes beyond observability to
            help you actually improve and ship AI agents and LLM applications
            with confidence.
          </p>

          <h2 id="faq">FAQ</h2>
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
          <div className="related-resources">
            <ul>
              <li>
                <Link to="https://docs.arize.com/phoenix">
                  Arize Phoenix Documentation
                </Link>
              </li>
              <li>
                <Link to={MLFLOW_DOCS_URL}>MLflow Documentation</Link>
              </li>
              <li>
                <Link to={`${MLFLOW_GENAI_DOCS_URL}tracing/`}>
                  MLflow Tracing Documentation
                </Link>
              </li>
              <li>
                <Link to={`${MLFLOW_GENAI_DOCS_URL}evaluate/`}>
                  MLflow Evaluation
                </Link>
              </li>
              <li>
                <Link to="/ai-gateway">MLflow AI Gateway</Link>
              </li>
            </ul>
          </div>
        </div>

        <aside className="article-sidebar">
          <p className="toc-title">ON THIS PAGE</p>
          <ul>
            <li>
              <a href="#quick-comparison">Quick Comparison</a>
            </li>
            <li>
              <a href="#open-source-licensing">Open Source &amp; Licensing</a>
            </li>
            <li>
              <a href="#tracing-observability">Tracing &amp; Observability</a>
            </li>
            <li>
              <a href="#evaluation-experimentation">
                Evaluation &amp; Experimentation
              </a>
            </li>
            <li>
              <a href="#architecture-operation">
                Architecture &amp; Operation
              </a>
            </li>
            <li>
              <a href="#ai-gateway">AI Gateway</a>
            </li>
            <li>
              <a href="#faq">FAQ</a>
            </li>
          </ul>
          <hr className="toc-divider" />
          <p className="toc-title">RESOURCES</p>
          <ul>
            <li>
              <a href={MLFLOW_DOCS_URL}>Documentation</a>
            </li>
            <li>
              <a href="/ai-gateway">AI Gateway</a>
            </li>
            <li>
              <a href="/slack">Slack</a>
            </li>
            <li>
              <a href="https://github.com/mlflow/mlflow">GitHub</a>
            </li>
          </ul>
        </aside>
      </div>
    </>
  );
}
