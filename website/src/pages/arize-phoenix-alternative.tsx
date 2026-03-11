import { type ReactNode, useState } from "react";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../components/Header/Header";
import { MLFLOW_DOCS_URL, MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import { CopyButton } from "../components/CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";

const MLFLOW_UI_SRC =
  "/img/arize-phoenix-alternative/mlflow-ui.png";
const PHOENIX_UI_SRC =
  "/img/arize-phoenix-alternative/arize-phoenix-ui.png";
const MLFLOW_LOGO_SRC =
  "/img/arize-phoenix-alternative/mlflow-logo.png";
const PHOENIX_LOGO_SRC =
  "/img/arize-phoenix-alternative/arize-phoenix-logo-placeholder.svg";
const SEO_TITLE =
  "Arize Phoenix Alternative for LLMs & Agents | MLflow Agent Platform";
const SEO_DESCRIPTION =
  "Compare MLflow vs Arize Phoenix for LLM observability, agent observability, tracing, and evaluation with a comprehensive, open-source agent engineering and ops platform.";
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

tracer_provider = register(project_name="support-bot")
LangChainInstrumentor().instrument(
    tracer_provider=tracer_provider
)

# Continue running your app as usual`,
  },
  {
    label: "OpenAI",
    mlflow: `import mlflow

mlflow.openai.autolog()

# That's it. Requests and responses
# are traced automatically.`,
    phoenix: `from phoenix.otel import register
from openinference.instrumentation.openai import (
    OpenAIInstrumentor,
)

tracer_provider = register(project_name="chat-app")
OpenAIInstrumentor().instrument(
    tracer_provider=tracer_provider
)`,
  },
  {
    label: "DSPy",
    mlflow: `import mlflow

mlflow.dspy.autolog()

# That's it. DSPy module executions
# are traced automatically.`,
    phoenix: `from phoenix.otel import register
from openinference.instrumentation.dspy import (
    DSPyInstrumentor,
)

tracer_provider = register(project_name="rag-evals")
DSPyInstrumentor().instrument(
    tracer_provider=tracer_provider
)`,
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
      <img src={MLFLOW_LOGO_SRC} alt="MLflow" className="code-logo-icon" />
      MLflow
    </span>
  );
}

function PhoenixCodeLabel() {
  return (
    <span className="code-logo-inline">
      <img
        src={PHOENIX_LOGO_SRC}
        alt="Arize Phoenix"
        className="code-logo-icon"
      />
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
    "End-to-end AI/ML lifecycle platform",
    "LLM observability and evaluation workbench",
  ],
  [
    "Core Data Plane",
    "Tracking server + SQL backend + artifact store",
    "Collector + UI + SQLite or PostgreSQL",
  ],
  [
    "Model Registry",
    "Built-in registry with lineage and versioning",
    "Not a core product surface",
  ],
  [
    "Deployment Surface",
    "Model packaging + serving patterns",
    "Focuses on tracing/evals, not model serving",
  ],
  [
    "Operational Focus",
    "Unified platform for training, models, and GenAI",
    "Trace-centric debugging and iteration loop",
  ],
];

const capabilityRows: {
  feature: string;
  mlflow: boolean | string;
  phoenix: boolean | string;
}[] = [
  { feature: "Training Experiment Tracking", mlflow: true, phoenix: false },
  { feature: "Model Registry & Version Aliases", mlflow: true, phoenix: false },
  { feature: "Model Packaging Standard", mlflow: true, phoenix: false },
  {
    feature: "OpenTelemetry Ingest for LLM Traces",
    mlflow: true,
    phoenix: true,
  },
  { feature: "Human Annotation Workflows", mlflow: true, phoenix: true },
  {
    feature: "Primary Strength",
    mlflow: "Lifecycle breadth",
    phoenix: "Trace-first LLM iteration",
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
      question: "What is the best open-source Arize Phoenix alternative?",
      answer: (
        <>
          MLflow is a strong open-source alternative when you need both{" "}
          <Link to="/llm-tracing">LLM tracing</Link> and lifecycle capabilities
          such as model registry, packaging, and deployment workflows.
        </>
      ),
      answerText:
        "MLflow is a strong open-source alternative when you need both LLM tracing and lifecycle capabilities such as model registry, packaging, and deployment workflows.",
    },
    {
      question: "How does MLflow differ from Arize Phoenix?",
      answer: (
        <>
          Arize Phoenix is primarily a trace-first observability and evaluation
          workspace. MLflow covers observability plus broader workflows across{" "}
          <Link to={MLFLOW_DOCS_URL}>tracking</Link>, registry, deployment, and
          GenAI quality operations.
        </>
      ),
      answerText:
        "Arize Phoenix is primarily a trace-first observability and evaluation workspace. MLflow covers observability plus broader workflows across tracking, registry, deployment, and GenAI quality operations.",
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
      question: "Can I use MLflow for both LLM and traditional ML workloads?",
      answer: (
        <>
          Yes. MLflow is designed for both classical ML and GenAI systems, so
          teams can manage experiments, model versions, and GenAI evaluations in
          one platform.
        </>
      ),
      answerText:
        "Yes. MLflow is designed for both classical ML and GenAI systems, so teams can manage experiments, model versions, and GenAI evaluations in one platform.",
    },
    {
      question: "Is Arize Phoenix better for trace-centric debugging?",
      answer: (
        <>
          Many teams choose Phoenix specifically for trace-first UI workflows.
          MLflow also provides strong trace visibility, with additional
          lifecycle and governance features for teams that need broader
          coverage.
        </>
      ),
      answerText:
        "Many teams choose Phoenix specifically for trace-first UI workflows. MLflow also provides strong trace visibility, with additional lifecycle and governance features for teams that need broader coverage.",
    },
    {
      question: "Does MLflow include model registry and deployment workflows?",
      answer: (
        <>
          Yes. MLflow includes model registry and packaging capabilities in
          addition to observability. This is a key reason teams adopt MLflow as
          a full AI engineering system.
        </>
      ),
      answerText:
        "Yes. MLflow includes model registry and packaging capabilities in addition to observability. This is a key reason teams adopt MLflow as a full AI engineering system.",
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
          MLflow uses an open-source model with broad ecosystem adoption. Teams
          can self-host or use managed environments while preserving portability
          across infrastructure choices.
        </>
      ),
      answerText:
        "MLflow uses an open-source model with broad ecosystem adoption. Teams can self-host or use managed environments while preserving portability across infrastructure choices.",
    },
    {
      question: "Which teams should choose Arize Phoenix vs MLflow?",
      answer: (
        <>
          Choose Phoenix if your immediate priority is dedicated trace-centric
          observability and annotation loops. Choose MLflow if you also need
          lifecycle governance, registry, and long-term platform consolidation.
        </>
      ),
      answerText:
        "Choose Phoenix if your immediate priority is dedicated trace-centric observability and annotation loops. Choose MLflow if you also need lifecycle governance, registry, and long-term platform consolidation.",
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
            GenAI evaluation workflows
          </Link>
          .
        </>
      ),
      answerText:
        "Start with the MLflow tracing quickstart and then expand to GenAI evaluation workflows.",
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
          .quick-nav {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 20px;
            margin: 12px 0 32px;
            background: #f9fafb;
          }
          .quick-nav p {
            margin-bottom: 12px !important;
          }
          .quick-nav ul {
            margin: 0 !important;
            padding-left: 20px !important;
          }
          .quick-nav li {
            margin-bottom: 4px;
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
            Whether you build LLM applications with OpenAI and Claude or
            multi-step agents with LangGraph and CrewAI, observability is a core
            production requirement. This guide compares Arize Phoenix and MLflow
            across LLM observability, agent observability, evaluation, and
            lifecycle operations so you can choose the right platform.
          </p>
          <div className="quick-nav">
            <p>
              <strong>Quick Navigation</strong>
            </p>
            <ul>
              <li>
                <a href="#quick-comparison">Quick Comparison</a>
              </li>
              <li>
                <a href="#llm-observability">LLM Observability</a>
              </li>
              <li>
                <a href="#agent-observability">Agent Observability</a>
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
                <a href="#faq">FAQ</a>
              </li>
            </ul>
          </div>

          <h2>What is Arize Phoenix?</h2>
          <div className="screenshot-wrap">
            <img
              src={PHOENIX_UI_SRC}
              alt="Arize Phoenix trace and evaluation workspace placeholder screenshot"
            />
          </div>
          <p>
            <strong>
              <Link to="https://arize.com/docs/phoenix">Arize Phoenix</Link>
            </strong>{" "}
            is an open-source AI observability and evaluation tool designed for
            LLM and agent applications. Phoenix focuses on trace inspection,
            span-level debugging, annotation workflows, and experiment loops for
            prompt and quality iteration. It is especially popular with teams
            that want OpenTelemetry-native instrumentation and a dedicated
            trace-centric UI for LLM development.
          </p>

          <h2>What is MLflow?</h2>
          <div className="screenshot-wrap">
            <img
              src={MLFLOW_UI_SRC}
              alt="MLflow tracing and evaluation workspace placeholder screenshot"
            />
          </div>
          <p>
            <strong>
              <Link to="/">MLflow</Link>
            </strong>{" "}
            is an open-source AI engineering platform that covers the full
            lifecycle: experiment tracking, model packaging, model registry,
            deployment integration, and{" "}
            <Link to="/genai/observability">GenAI observability</Link>. Teams
            use MLflow to build, monitor, and improve ML and LLM systems in one
            unified platform while keeping infrastructure and governance choices
            flexible.
          </p>

          <h2 id="quick-comparison">Quick Comparison</h2>
          <div className="tldr-grid">
            <div className="tldr-card highlight">
              <h3>
                <img src={MLFLOW_LOGO_SRC} alt="MLflow" className="tldr-logo" />
                Choose MLflow if you...
              </h3>
              <ul>
                <li>
                  Need a <strong>complete lifecycle platform</strong> across
                  tracking, registry, deployment, and GenAI
                </li>
                <li>
                  Want <strong>one system</strong> for both classical ML and LLM
                  applications
                </li>
                <li>
                  Require <strong>model governance</strong> and controlled
                  promotion workflows
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
                  src={PHOENIX_LOGO_SRC}
                  alt="Arize Phoenix"
                  className="tldr-logo"
                />
                Choose Arize Phoenix if you...
              </h3>
              <ul>
                <li>
                  Primarily need <strong>LLM tracing + eval workflows</strong>
                </li>
                <li>
                  Want a <strong>trace-first UI</strong> for rapid debugging and
                  prompt iteration
                </li>
                <li>
                  Do not need built-in{" "}
                  <strong>model registry/deployment</strong>
                </li>
                <li>
                  Are comfortable with <strong>ELv2 licensing</strong> tradeoffs
                </li>
              </ul>
            </div>
          </div>

          <h2 id="llm-observability">LLM Observability</h2>
          <p>
            LLM observability focuses on prompt execution, model responses,
            latency, token usage, and quality trends over time. Teams use it to
            debug degraded responses, identify expensive paths, and detect
            quality regressions before they impact production users.
          </p>
          <p>
            Phoenix is strong for trace-first inspection and annotation loops.
            MLflow supports this workflow while also connecting traces to model
            lineage and evaluation history. For teams that need deeper quality
            control, pair tracing with{" "}
            <Link to="/genai/evaluations">LLM evaluation workflows</Link>.
          </p>

          <h2 id="agent-observability">Agent Observability</h2>
          <p>
            Agent observability extends LLM observability to multi-step systems
            with tool calls, retrieval hops, and branching control flow. This is
            essential for debugging failures that happen across several steps.
          </p>
          <p>
            MLflow supports agent-oriented tracing and integrates with common
            frameworks such as LangGraph, LangChain, and CrewAI while preserving
            a lifecycle view across experiments and deployments. Teams building
            production agents can combine observability with{" "}
            <Link to="/llm-evaluation">agent evaluation</Link> to drive safer
            releases.
          </p>

          <h2 id="open-source-licensing">Open Source &amp; Licensing</h2>
          <p>
            Both tools are open-source projects, but the licensing model is an
            important strategic difference. <strong>MLflow</strong> is licensed
            under Apache 2.0 and backed by broad open governance in the Linux
            Foundation ecosystem, making it easier to adopt in environments that
            prioritize permissive licensing and long-term portability.
          </p>
          <p>
            <strong>Arize Phoenix</strong> is distributed under Elastic License
            2.0. It is free to self-host, but ELv2 includes restrictions for
            offering the software itself as a managed hosted service. For many
            internal use cases this is fine, but teams building customer-facing
            hosted products should evaluate the implications early.
          </p>

          <h2 id="lifecycle-scope">
            Lifecycle Scope: Tracking, Registry &amp; Deployment
          </h2>
          <p>
            The largest difference is platform scope. <strong>MLflow</strong> is
            designed as a full lifecycle system: it tracks training and
            evaluation runs, packages models in a standard format, manages
            versions in a registry, and supports production deployment patterns.
            See <Link to={MLFLOW_DOCS_URL}>MLflow docs</Link> for end-to-end
            lifecycle capabilities.
          </p>
          <p>
            <strong>Arize Phoenix</strong> focuses on observability and
            evaluation for LLM/agent systems. That makes Phoenix highly
            effective for quality iteration loops, but it is not intended to
            replace a training experiment tracker or model registry.
          </p>
          <ComparisonTable rows={architectureTable} />

          <h2 id="tracing-observability">Tracing &amp; Observability</h2>
          <p>
            Both platforms now align around OpenTelemetry-compatible tracing.
            Phoenix is built around trace exploration, while MLflow has expanded
            rapidly with native tracing plus OTLP ingest/export so teams can
            integrate with broader observability infrastructure.
          </p>
          <p>
            Instrumentation ergonomics differ in day-to-day workflows.
            <strong> MLflow</strong> emphasizes one-line <code>autolog()</code>
            APIs across many frameworks, while <strong>Phoenix</strong> commonly
            uses OpenInference/OpenTelemetry instrumentation with explicit
            tracer registration. Teams also track{" "}
            <Link to={`${MLFLOW_GENAI_DOCS_URL}tracing/token-usage-cost/`}>
              token usage and cost
            </Link>{" "}
            to connect observability with spend.
          </p>
          <CodeTabs tabs={tracingExamples} />

          <h2 id="evaluation-experimentation">
            Evaluation &amp; Experimentation
          </h2>
          <p>
            Phoenix is strong at trace-driven LLM evaluation loops: inspect
            traces, annotate outputs, build datasets, run experiments, and
            compare prompt or system changes. This workflow is a core Phoenix
            strength and is one reason teams adopt it for agent iteration.
          </p>
          <p>
            MLflow combines LLM evaluation with broader ML evaluation and
            lifecycle capabilities. This allows teams to connect online quality
            signals, offline evaluations, model versions, and deployment context
            inside one platform rather than stitching multiple systems together.
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

          <h2 id="governance-security">Governance &amp; Security</h2>
          <p>
            For self-hosted deployments, both platforms support authentication
            and role-based control patterns, but they are optimized for
            different outcomes. Phoenix emphasizes secure access and data
            controls for observability projects. MLflow emphasizes governance
            across experiments, models, and prompts, and can be extended with
            plugins and enterprise infrastructure patterns.
          </p>
          <p>
            If your roadmap includes strict enterprise governance and auditable
            lifecycle controls, MLflow is usually the stronger long-term center
            of gravity. If your immediate goal is accelerating LLM debugging and
            feedback loops, Phoenix can deliver value quickly.
          </p>

          <h2 id="databricks-managed-mlflow">Databricks Managed MLflow</h2>
          <p>
            Teams running on Databricks can use managed MLflow with platform
            integrations for governance, operational scale, and production
            monitoring workflows. This is especially relevant when you want
            trace and model data to be queryable in governed data systems and
            tied to enterprise access controls.
          </p>
          <p>
            This managed path often strengthens MLflow's advantage for larger
            organizations that need unified governance and reliability while
            preserving API portability with open-source MLflow.
          </p>

          <h2>Summary</h2>
          <p>
            <strong>Arize Phoenix</strong> is a strong product for trace-first
            LLM observability and evaluation workflows. If your main challenge
            is debugging agent behavior and iterating on prompts with annotation
            loops, Phoenix is a compelling option.
          </p>
          <p>
            <strong>MLflow</strong> is a broader AI engineering platform. It
            covers LLM tracing and evaluation while also delivering training
            experiment tracking, model packaging, model registry, and deployment
            integration. Choose MLflow when you need a single platform that can
            support both current LLM workloads and long-term ML/AI lifecycle
            needs.
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
                <Link to="https://arize.com/docs/phoenix">
                  Arize Phoenix Overview
                </Link>
              </li>
              <li>
                <Link to="https://docs.arize.com/phoenix">
                  Arize Phoenix Docs
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
                  MLflow GenAI Evaluation
                </Link>
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
              <a href="#lifecycle-scope">Lifecycle Scope</a>
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
              <a href="#faq">FAQ</a>
            </li>
          </ul>
          <hr className="toc-divider" />
          <p className="toc-title">RESOURCES</p>
          <ul>
            <li>
              <a href="/llm-evaluation">MLflow FAQ</a>
            </li>
            <li>
              <a href={MLFLOW_DOCS_URL}>Documentation</a>
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
