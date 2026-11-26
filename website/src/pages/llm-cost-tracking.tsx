import { useState } from "react";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../components/Header/Header";
import { SocialLinksFooter } from "../components/SocialLinksFooter/SocialLinksFooter";
import { ArticleSidebar } from "../components/ArticleSidebar/ArticleSidebar";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import CostDashboard from "@site/static/img/llm-cost-tracking/mlflow-demo-cost-breakdown.png";
import { CopyButton } from "../components/CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";

const SEO_TITLE =
  "LLM Cost Tracking with MLflow: Token Usage and Cost per Trace";
const SEO_DESCRIPTION =
  "LLM cost tracking measures token usage and spend for every LLM call and agent step. See how MLflow Tracing records tokens and USD cost per span and trace.";

const faqs: {
  question: string;
  answer: React.ReactNode;
  answerText?: string;
}[] = [
  {
    question: "What is LLM cost tracking?",
    answer:
      "LLM cost tracking is the practice of recording how many tokens each LLM call consumes and what those tokens cost, then attributing that spend to requests, agent steps, models, users, or features. It answers questions that a monthly provider invoice cannot, such as which step of an agent is the most expensive or which model change raised the bill. MLflow Tracing captures token usage on every LLM span and estimates cost in USD from model pricing.",
  },
  {
    question: "How do I track LLM token usage and cost with MLflow?",
    answer:
      "Enable tracing for your LLM library, for example mlflow.openai.autolog(), and run your application. MLflow records input, output, and total tokens on each LLM span and aggregates them on the trace. With MLflow 3.10 or later, MLflow also estimates cost in USD per span and per trace. Read the values with trace.info.token_usage and trace.info.cost, or view them in the MLflow UI.",
  },
  {
    question: "Which MLflow version do I need for cost tracking?",
    answer:
      "Token usage tracking requires MLflow 3.2.0 or later. Cost tracking requires MLflow 3.10.0 or later, and the token usage and cost documentation recommends installing the genai extra on the tracking server (pip install 'mlflow[genai]>=3.10.0'). Prices come from LiteLLM's price table when LiteLLM is installed, and from MLflow's built-in model catalog otherwise.",
  },
  {
    question: "How does MLflow calculate the cost of an LLM call?",
    answer:
      "MLflow multiplies the input and output token counts recorded on an LLM span by the per-token price of the model, using its price table (LiteLLM's when installed, otherwise MLflow's built-in catalog). The result is stored on the span as mlflow.llm.cost with input_cost, output_cost, and total_cost in USD, and summed into trace-level cost. The value is an estimate based on list pricing, so it can differ from your provider invoice if you have negotiated rates or discounts.",
  },
  {
    question: "Why does my trace show tokens but no cost?",
    answer:
      "Cost is missing when the model name recorded on the span has no entry in the pricing table, so trace.info.cost is None while the token counts are still recorded. In that case, set the mlflow.chat.tokenUsage and mlflow.llm.cost attributes on the span yourself, which overrides the calculated values.",
  },
  {
    question: "Can I see cost per agent step or per tool call?",
    answer:
      "Yes. Cost is stored on each LLM span, and every span sits inside the trace tree of your agent. You can open a trace in the MLflow UI to see token usage and cost next to each LLM call in a multi-step agent, or iterate over trace.data.spans in Python and read span.llm_cost to attribute spend to individual steps.",
  },
  {
    question: "Can MLflow track cost across multiple models and providers?",
    answer:
      "Yes. Token usage and cost tracking works across the tracing integrations that return usage data, and the experiment dashboard breaks cost down by model or provider. Some providers or models do not return token usage, in which case the usage and cost fields are empty and you can set them manually.",
  },
  {
    question: "Does MLflow AI Gateway track cost?",
    answer:
      "Yes. When usage tracking is enabled on an AI Gateway endpoint, each request is logged as a trace, and the AI Gateway usage dashboard charts requests, latency, errors, token usage, cost breakdown, and cost over time. Budget policies can alert through a webhook or reject requests with HTTP 429 when spend passes a daily, weekly, or monthly limit. Token and cost charts can be empty for models whose usage or pricing is unavailable.",
  },
  {
    question: "Is MLflow LLM cost tracking free?",
    answer:
      "Yes. MLflow is open source under the Apache 2.0 license and backed by the Linux Foundation. Token and cost tracking is included at no cost, with no per-trace fees. You pay only for your own infrastructure and for the LLM API calls your application makes. Managed MLflow is also available on Databricks.",
  },
];

const faqJsonLd = {
  "@context": "https://schema.org",
  "@type": "FAQPage",
  mainEntity: faqs.map((faq) => ({
    "@type": "Question",
    name: faq.question,
    acceptedAnswer: {
      "@type": "Answer",
      text: faq.answerText || faq.answer,
    },
  })),
};

const softwareJsonLd = {
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  name: "MLflow",
  applicationCategory: "DeveloperApplication",
  operatingSystem: "Cross-platform",
  offers: {
    "@type": "Offer",
    price: "0",
    priceCurrency: "USD",
  },
  description:
    "Open-source platform for LLM tracing, token usage and cost tracking, evaluation, and deployment.",
  url: "https://mlflow.org",
  license: "https://www.apache.org/licenses/LICENSE-2.0",
};

function CodeBlock({
  code,
  language = "python",
}: {
  code: string;
  language?: string;
}) {
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

const tableStyle: React.CSSProperties = {
  width: "100%",
  borderCollapse: "collapse",
  margin: "0 0 32px 0",
  fontSize: "15px",
};
const cellStyle: React.CSSProperties = {
  border: "1px solid #e5e7eb",
  padding: "10px 14px",
  textAlign: "left",
  verticalAlign: "top",
  color: "#3d3d3d",
  lineHeight: 1.5,
};
const headStyle: React.CSSProperties = {
  ...cellStyle,
  background: "#f9fafb",
  color: "#1a1a1a",
  fontWeight: 600,
};

export default function LLMCostTracking() {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta
          property="og:url"
          content="https://mlflow.org/llm-cost-tracking"
        />
        <link rel="canonical" href="https://mlflow.org/llm-cost-tracking" />
        <script type="application/ld+json">{JSON.stringify(faqJsonLd)}</script>
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
          <h1>LLM Cost Tracking</h1>

          <p>
            <strong>LLM cost tracking</strong> is the practice of recording how
            many tokens every LLM call uses and what those tokens cost, then
            attributing that spend to individual requests, agent steps, models,
            and features. MLflow Tracing captures token usage on each LLM call
            and, from MLflow 3.10, estimates the cost in USD per span and per
            trace, so you can see where your LLM budget goes in the MLflow UI or
            query it from Python. No separate billing tool is required.
          </p>

          <p style={{ color: "#6b7280", fontSize: "14px" }}>
            Last reviewed: November 26, 2026
          </p>

          <div style={{ margin: "40px 0", textAlign: "center" }}>
            <img
              src={CostDashboard}
              alt="MLflow experiment dashboard with the Cost Breakdown donut chart by model and the Cost Over Time chart, plus latency, errors, token usage, and tokens per trace, from MLflow demo data"
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
              The MLflow experiment dashboard showing the Cost Breakdown donut
              by model and the Cost Over Time chart, from MLflow demo data
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
                <a href="#what-is">What is LLM cost tracking?</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#how-mlflow-tracks-cost">
                  How MLflow tracks tokens and cost
                </a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#quickstart">Track cost in three steps</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#analyze">Analyze cost by trace, step, and model</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#ai-gateway">Cost control with AI Gateway</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#manual">When automatic cost is unavailable</a>
              </li>
              <li style={{ marginBottom: "0" }}>
                <a href="#faq">FAQ</a>
              </li>
            </ul>
          </div>

          <h2 id="what-is">What Is LLM Cost Tracking?</h2>

          <p>
            LLM providers bill by tokens: the input tokens you send and the
            output tokens the model generates, each at a model-specific price.
            LLM cost tracking turns those token counts into a per-call cost and
            keeps it next to the prompt, response, and latency of that call.
          </p>

          <p>
            A provider invoice tells you what you spent in total. Cost tracking
            tells you <em>why</em>: which agent step, prompt version, model, or
            user drove the spend, and whether a change made it better or worse.
          </p>

          <h3>Why it matters for agents</h3>

          <p>
            A single user request to an agent can fan out into many LLM calls
            for planning, tool selection, retrieval, and answer generation. Cost
            per request therefore varies widely and is hard to predict from the
            prompt alone. Tracking cost on every span of the trace makes the
            expensive step visible, so you can fix it instead of guessing.
          </p>

          <h3>Token usage vs. cost</h3>

          <p>
            <strong>Token usage</strong> is a count reported by the model
            response. <strong>Cost</strong> is derived: token counts multiplied
            by the price of the model. Token usage is the reliable signal, and
            cost is an estimate that depends on pricing data being available for
            the model you call.
          </p>

          <h2 id="how-mlflow-tracks-cost">
            How MLflow Tracks Token Usage and Cost
          </h2>

          <p>
            <Link href="/llm-tracing">MLflow Tracing</Link> records each LLM
            call as a span. For supported integrations, MLflow stores the token
            counts and the estimated cost as span attributes, and aggregates
            them on the parent trace.
          </p>

          <table style={tableStyle}>
            <thead>
              <tr>
                <th style={headStyle}>Level</th>
                <th style={headStyle}>Token usage</th>
                <th style={headStyle}>Cost (USD)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={cellStyle}>Span (one LLM call)</td>
                <td style={cellStyle}>
                  <code>mlflow.chat.tokenUsage</code>: input, output, total
                </td>
                <td style={cellStyle}>
                  <code>mlflow.llm.cost</code>: input, output, total
                </td>
              </tr>
              <tr>
                <td style={cellStyle}>Trace (whole request)</td>
                <td style={cellStyle}>
                  <code>trace.info.token_usage</code>
                </td>
                <td style={cellStyle}>
                  <code>trace.info.cost</code>
                </td>
              </tr>
            </tbody>
          </table>

          <h3>Requirements</h3>

          <table style={tableStyle}>
            <thead>
              <tr>
                <th style={headStyle}>Feature</th>
                <th style={headStyle}>MLflow version</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={cellStyle}>Token usage tracking</td>
                <td style={cellStyle}>3.2.0 or later</td>
              </tr>
              <tr>
                <td style={cellStyle}>Cost tracking</td>
                <td style={cellStyle}>3.10.0 or later</td>
              </tr>
            </tbody>
          </table>

          <p>
            The documentation recommends installing the <code>genai</code> extra
            on the tracking server to compute cost. Prices come from LiteLLM's
            model price data when LiteLLM is installed and from MLflow's
            built-in model catalog otherwise, so cost is an estimate at list
            prices. Some providers or models do not return token usage, and in
            those cases the usage and cost fields are empty. See the{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/token-usage-cost/"}>
              token usage and cost documentation
            </Link>{" "}
            for the full details.
          </p>

          <h2 id="quickstart">Track LLM Cost in Three Steps</h2>

          <p>
            <strong>
              1. Install MLflow 3.10 or later with the genai extra and start a
              server.
            </strong>
          </p>

          <CodeBlock
            language="bash"
            code={`pip install 'mlflow[genai]>=3.10.0'
mlflow server`}
          />

          <p>
            <strong>2. Turn on tracing for your LLM library.</strong> This
            example uses OpenAI; MLflow provides{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/integrations/"}>
              one-line integrations for many other providers and frameworks
            </Link>
            .
          </p>

          <CodeBlock
            code={`import mlflow
from openai import OpenAI

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("cost-tracking-demo")

mlflow.openai.autolog()

client = OpenAI()
client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize MLflow in one sentence."}],
)`}
          />

          <p>
            <strong>3. Read token usage and cost from the trace.</strong>
          </p>

          <CodeBlock
            code={`trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

usage = trace.info.token_usage  # {"input_tokens", "output_tokens", "total_tokens"}
cost = trace.info.cost          # {"input_cost", "output_cost", "total_cost"} in USD

print(usage)
if cost:
    print(f"Total cost: \${cost['total_cost']:.6f}")`}
          />

          <h2 id="analyze">Analyze Cost by Trace, Step, and Model</h2>

          <h3>See cost in the MLflow UI</h3>

          <p>
            Open a trace to see token usage and cost next to each LLM call in
            the span tree. The experiment dashboard aggregates the same data
            into charts for traces, latency, errors, token usage, tokens per
            trace, cost breakdown, and cost over time, which you can split by
            model or provider.
          </p>

          <h3>Find your most expensive traces</h3>

          <p>
            Use <code>mlflow.search_traces</code> to pull traces into Python and
            aggregate. This example totals spend across the first 500 traces
            that the search returns and lists the five most expensive of them.
            For a full experiment, page through results with{" "}
            <code>MlflowClient.search_traces</code>.
          </p>

          <CodeBlock
            code={`import mlflow

traces = mlflow.search_traces(
    locations=["<experiment_id>"],
    return_type="list",
    max_results=500,
)

priced = [t for t in traces if t.info.cost]
total_cost = sum(t.info.cost["total_cost"] for t in priced)
print(f"{len(priced)} priced traces, total cost: \${total_cost:.4f}")

for t in sorted(priced, key=lambda t: t.info.cost["total_cost"], reverse=True)[:5]:
    print(t.info.trace_id, f"\${t.info.cost['total_cost']:.6f}")`}
          />

          <h3>Attribute spend to agent steps and models</h3>

          <p>
            Cost lives on each LLM span, so you can group it by span name to see
            which step of an agent is the most expensive, or by model name to
            compare models.
          </p>

          <CodeBlock
            code={`from collections import defaultdict

by_model = defaultdict(float)
by_step = defaultdict(float)

for trace in traces:
    for span in trace.data.spans:
        if cost := span.llm_cost:
            by_model[span.model_name] += cost["total_cost"]
            by_step[span.name] += cost["total_cost"]

print(dict(by_model))
print(dict(by_step))`}
          />

          <p>
            Pair cost with quality. A cheaper model is only a saving if{" "}
            <Link href="/llm-evaluation">LLM evaluation</Link> shows the answers
            stay good, and evaluation results attach to the same traces that
            carry the cost.
          </p>

          <h2 id="ai-gateway">Cost Control with MLflow AI Gateway</h2>

          <p>
            If your applications call models through the{" "}
            <Link href="/ai-gateway">MLflow AI Gateway</Link>, you can track and
            cap spend at the gateway instead of in each application. With usage
            tracking enabled on an endpoint, every request is logged as a trace,
            and the{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL + "governance/ai-gateway/usage-tracking"
              }
            >
              usage dashboard
            </Link>{" "}
            charts requests, latency, errors, token usage, cost breakdown, and
            cost over time for each endpoint.
          </p>

          <p>
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "governance/ai-gateway/budget-alerts-limits"
              }
            >
              Budget policies
            </Link>{" "}
            set a USD threshold over a daily, weekly, or monthly window. When
            the threshold is passed, a policy either fires a webhook alert or
            rejects new requests with HTTP 429.
          </p>

          <h2 id="manual">When Automatic Cost Is Unavailable</h2>

          <p>
            Cost is missing when the model name on the span has no entry in the
            pricing table. Token counts can still be correct. In these cases,
            set the token usage and cost attributes on the span yourself:
          </p>

          <CodeBlock
            code={`import mlflow


@mlflow.trace
def my_llm_call():
    span = mlflow.get_current_active_span()

    # ... call your model here ...

    span.set_attribute(
        "mlflow.chat.tokenUsage",
        {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    )
    span.set_attribute(
        "mlflow.llm.cost",
        {"input_cost": 0.0001, "output_cost": 0.0002, "total_cost": 0.0003},
    )
    return "response"`}
          />

          <h2>How MLflow Compares with Other Ways to Track LLM Cost</h2>

          <table style={tableStyle}>
            <thead>
              <tr>
                <th style={headStyle}>Approach</th>
                <th style={headStyle}>What you get</th>
                <th style={headStyle}>Limits</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={cellStyle}>Provider billing dashboard</td>
                <td style={cellStyle}>
                  Exact invoiced spend per account, project, or API key
                </td>
                <td style={cellStyle}>
                  No link to prompts, agent steps, or quality; one provider at a
                  time
                </td>
              </tr>
              <tr>
                <td style={cellStyle}>Custom logging in your code</td>
                <td style={cellStyle}>Whatever fields you decide to log</td>
                <td style={cellStyle}>
                  You maintain the price tables, schema, and charts
                </td>
              </tr>
              <tr>
                <td style={cellStyle}>MLflow Tracing</td>
                <td style={cellStyle}>
                  Tokens and estimated cost per span and trace, next to the
                  prompt, response, latency, and evaluation scores
                </td>
                <td style={cellStyle}>
                  Cost is an estimate from list pricing, and depends on the
                  provider returning usage
                </td>
              </tr>
            </tbody>
          </table>

          <div className="info-box">
            <p>
              <Link href="/genai" style={{ color: "#007bff" }}>
                <strong>MLflow</strong>
              </Link>{" "}
              is an open-source{" "}
              <strong>
                AI engineering platform for agents, LLMs, and ML models
              </strong>
              . Backed by the Linux Foundation and licensed under Apache 2.0,
              MLflow provides token and cost tracking with no per-trace fees.{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/token-usage-cost/"}>
                Get started →
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
              <Link href="/llm-tracing">LLM Tracing</Link>
            </li>
            <li>
              <Link href="/ai-observability">AI Observability</Link>
            </li>
            <li>
              <Link href="/ai-gateway">AI Gateway</Link>
            </li>
            <li>
              <Link href="/ai-monitoring">AI Monitoring</Link>
            </li>
            <li>
              <Link href="/llm-evaluation">LLM Evaluation</Link>
            </li>
            <li>
              <Link href="/blog/agent-costs-mlflow-gateway">
                How to Prevent Runaway Agent Costs with MLflow AI Gateway
              </Link>
            </li>
            <li>
              <Link href="/top-5-agent-observability-tools">
                Top 5 Agent Observability Tools Comparison
              </Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/token-usage-cost/"}>
                Token Usage and Cost Tracking Documentation
              </Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL}>
                MLflow for Agents and LLMs Documentation
              </Link>
            </li>
          </ul>
        </div>

        <ArticleSidebar />
        <SocialLinksFooter />
      </div>
    </>
  );
}
