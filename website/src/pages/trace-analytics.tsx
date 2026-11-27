import { useState } from "react";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../components/Header/Header";
import { SocialLinksFooter } from "../components/SocialLinksFooter/SocialLinksFooter";
import { ArticleSidebar } from "../components/ArticleSidebar/ArticleSidebar";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import { CopyButton } from "../components/CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";
import DashboardImg from "@site/static/img/trace-analytics/mlflow-demo-dashboard.png";
import IssuesImg from "@site/static/img/trace-analytics/mlflow-demo-issues.png";

const SEO_TITLE = "Trace Analytics for LLM and Agent Applications | MLflow";
const SEO_DESCRIPTION =
  "Trace analytics turns LLM and agent traces into answers: search and filter traces, chart latency, tokens, and errors, group sessions, and detect issues with MLflow.";

const faqs: { question: string; answer: string }[] = [
  {
    question: "What is trace analytics?",
    answer:
      "Trace analytics is the practice of searching, aggregating, and mining the traces that an LLM application or agent produces, so you can find slow, expensive, failing, or low-quality requests and understand why they happened. A trace records every step of one request: prompts, model calls, tool calls, retrieved documents, latency, and token usage. Analytics works across thousands of those traces instead of reading them one at a time.",
  },
  {
    question: "How do I analyze LLM traces with MLflow?",
    answer:
      "Turn on MLflow Tracing for your application, then use the Traces tab in the MLflow UI to filter and inspect traces, the Dashboard to chart latency, errors, token usage, and cost over time, and Sessions to follow multi-turn conversations. For custom analysis, call mlflow.search_traces() to get a pandas DataFrame or a list of Trace objects and analyze them in Python.",
  },
  {
    question: "How do I search and filter traces in MLflow?",
    answer:
      "Use the filter bar on the Traces tab or the filter_string argument of mlflow.search_traces(). The SQL-like syntax supports trace status, execution time, trace name, tags, metadata, span name and type, linked prompts, feedback, and expectations, combined with AND. The OR keyword is not supported.",
  },
  {
    question: "Can MLflow show latency, token usage, and errors over time?",
    answer:
      "Yes. The experiment Dashboard has a Usage tab with request counts, latency percentiles, error counts and rates, token usage, and cost, plus a Quality tab for scorer results and a Tool calls tab for agent tool usage. All tabs have a time range and a time unit selector. The charts need a SQL-backed tracking store such as SQLite, PostgreSQL, or MySQL.",
  },
  {
    question: "How do I group traces into conversations or users?",
    answer:
      "Set a session ID and a user ID on each trace with mlflow.update_current_trace(session_id=..., user=...), or wrap auto-instrumented code in mlflow.tracing.context(session_id=..., user=...). The Traces tab then shows the Session and User columns, you can filter on them, and mlflow.search_sessions() returns complete multi-turn sessions. The session_id and user arguments need MLflow 3.11.0 or later; earlier versions set the mlflow.trace.session and mlflow.trace.user metadata keys directly.",
  },
  {
    question:
      "How do I find quality problems in traces without reading each one?",
    answer:
      "Attach assessments to traces: LLM judge scores, code-based checks, and human feedback. Then filter by them, for example feedback.rating = 'positive', or chart them on the Quality tab. MLflow also has automatic issue detection, which uses AI to analyze traces and groups related failures into named issues with evidence, so you can start from the pattern instead of the individual trace.",
  },
  {
    question: "Does trace analytics work for agents and RAG applications?",
    answer:
      "Yes. Traces capture nested spans for each LLM call, tool call, and retriever step, so you can filter by span name or type, chart tool call success rate and latency on the Tool calls tab, and compare traces for different agent or retrieval configurations. The same search and dashboard features apply to any framework MLflow Tracing supports.",
  },
  {
    question: "Do I need to send my trace data to a third party?",
    answer:
      "No. MLflow is open source under the Apache 2.0 license, and you can self-host the tracking server on your own infrastructure with a SQL database of your choice. Trace data stays where you put it, and you can query it with the UI or the Python API.",
  },
  {
    question: "Is MLflow trace analytics free?",
    answer:
      "Yes. Tracing, trace search, the dashboard, sessions, assessments, and issue detection UI are part of open source MLflow at no license cost. You pay only for your own infrastructure and for any LLM calls that LLM judges or issue detection make. Managed MLflow is also available on Databricks and other platforms.",
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

const softwareJsonLd = {
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  name: "MLflow",
  applicationCategory: "DeveloperApplication",
  operatingSystem: "Cross-platform",
  offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
  description:
    "Open-source platform for LLM tracing, trace analytics, agent evaluation, and observability.",
  url: "https://mlflow.org",
  license: "https://www.apache.org/licenses/LICENSE-2.0",
};

function CodeBlock({ code }: { code: string }) {
  return (
    <div
      className="rounded-lg border border-white/10 overflow-hidden"
      style={{ backgroundColor: CODE_BG, margin: "8px 0 24px" }}
    >
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
        <span className="text-xs text-white/50 font-mono">python</span>
        <CopyButton code={code} />
      </div>
      <div className="p-3 overflow-x-auto">
        <Highlight theme={customNightOwl} code={code} language="python">
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

function Figure({
  src,
  alt,
  caption,
}: {
  src: string;
  alt: string;
  caption: string;
}) {
  return (
    <figure style={{ margin: "40px 0", textAlign: "center" }}>
      <img
        src={src}
        alt={alt}
        loading="lazy"
        style={{
          width: "100%",
          borderRadius: "8px",
          boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
        }}
      />
      <figcaption
        style={{ fontSize: "14px", color: "#6b7280", marginTop: "12px" }}
      >
        {caption}
      </figcaption>
    </figure>
  );
}

const cellStyle = { padding: "14px 16px", verticalAlign: "top" as const };

function Table({ head, rows }: { head: string[]; rows: string[][] }) {
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
            {head.map((h) => (
              <th
                key={h}
                style={{
                  ...cellStyle,
                  textAlign: "left",
                  fontWeight: 600,
                  color: "#1a1a1a",
                }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row[0]}
              style={{
                borderBottom: "1px solid #e2e8f0",
                backgroundColor: "#ffffff",
              }}
            >
              {row.map((cell, i) => (
                <td
                  key={i}
                  style={{
                    ...cellStyle,
                    fontWeight: i === 0 ? 600 : 400,
                    color: i === 0 ? "#1a1a1a" : "#3d3d3d",
                  }}
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

export default function TraceAnalytics() {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://mlflow.org/trace-analytics" />
        <link rel="canonical" href="https://mlflow.org/trace-analytics" />
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
          <h1>Trace Analytics for LLM and Agent Applications</h1>

          <p>
            Trace analytics is the practice of searching, aggregating, and
            mining the traces your LLM application or agent produces to find
            slow, expensive, failing, or low-quality requests.{" "}
            <Link href="/genai">MLflow</Link> provides it in one open-source
            platform: trace search, an LLM trace dashboard for latency, errors,
            tokens, and cost, session grouping, quality assessments, and
            automatic issue detection.
          </p>

          <p style={{ fontSize: "14px", color: "#6b7280", marginTop: "-8px" }}>
            Last updated: 2026-11-26
          </p>

          <p>
            A single trace shows you what happened in one request. Analytics
            shows you what is happening across all of them. Once an agent
            handles thousands of requests a day, you cannot read traces one by
            one, so you need to filter them, aggregate them, and let tools point
            you at the patterns worth reading.
          </p>

          <Figure
            src={DashboardImg}
            alt="MLflow experiment Dashboard showing trace counts, latency percentiles, errors, token usage, and cost charts for an LLM application"
            caption="The MLflow Dashboard charts trace volume, latency, errors, and token usage over time. Demo data."
          />

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
                <a href="#what-is-trace-analytics">What is trace analytics?</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#questions">What questions it answers</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#search">How to search and filter traces</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#dashboard">
                  Dashboards for latency, tokens, and cost
                </a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#sessions">Sessions and users</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#assessments">Measuring quality</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#issue-detection">Automatic issue detection</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#platforms">Choosing a platform</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#getting-started">Getting started</a>
              </li>
              <li style={{ marginBottom: "0" }}>
                <a href="#faq">FAQ</a>
              </li>
            </ul>
          </div>

          <h2 id="what-is-trace-analytics">What Is Trace Analytics?</h2>

          <p>
            A <Link href="/llm-tracing">trace</Link> is the record of one
            request through your application: the user input, each LLM call,
            tool call, and retrieval step as nested spans, plus timing, token
            counts, errors, and any tags or feedback attached to it. Trace
            analytics treats the collection of traces as a dataset you can
            query.
          </p>

          <p>
            It sits between two related practices.{" "}
            <Link href="/llm-observability">AI observability</Link> is the
            broader goal of seeing what your system does; trace analytics is the
            part that answers questions across many traces.{" "}
            <Link href="/llm-evaluation">Evaluation</Link> scores quality on a
            fixed test set; trace analytics works on real production traffic and
            can feed failures back into that test set.
          </p>

          <h2 id="questions">What Questions Can Trace Analytics Answer?</h2>

          <p>
            Most teams start with a handful of recurring questions. Each one
            maps to a feature in MLflow:
          </p>

          <Table
            head={["Question", "Where to look in MLflow"]}
            rows={[
              [
                "Which requests are slow or failing?",
                "Filter on trace status and execution time in the Traces tab, or chart latency percentiles and error rate on the Dashboard.",
              ],
              [
                "Where do tokens and cost go?",
                "Token usage and cost charts on the Dashboard, and per-span token counts in the trace viewer.",
              ],
              [
                "Which tools does my agent call, and do they work?",
                "The Tool calls tab shows call counts, latency, and error rate for each tool.",
              ],
              [
                "How do multi-turn conversations go?",
                "Session and User columns in the Traces tab, and mlflow.search_sessions() in Python.",
              ],
              [
                "Which responses are low quality?",
                "Judge scores, code checks, and human feedback attached to traces as assessments, charted on the Quality tab.",
              ],
              [
                "What are the recurring failure patterns?",
                "Automatic issue detection groups related failures into named issues with evidence.",
              ],
            ]}
          />

          <h2 id="search">How Do I Search and Filter Traces?</h2>

          <p>
            The Traces tab in the MLflow UI lists every trace in an experiment
            with its input, output, duration, state, token count, session, and
            assessment results. The filter bar uses the same SQL-like syntax as
            the Python API. You can filter on trace status, execution time,
            trace name, tags, metadata, span name and type, linked prompt
            versions, feedback, and expectations, and combine conditions with{" "}
            <code>AND</code>.
          </p>

          <CodeBlock
            code={`import mlflow

mlflow.set_experiment("my-agent")

# Slow production failures, newest first
traces = mlflow.search_traces(
    filter_string="""
        tag.environment = 'production'
        AND trace.status = 'ERROR'
        AND trace.execution_time_ms > 500
    """,
    order_by=["timestamp_ms DESC"],
    max_results=100,
)`}
          />

          <p>
            <code>mlflow.search_traces()</code> returns a pandas DataFrame by
            default, or a list of <code>Trace</code> objects with{" "}
            <code>return_type="list"</code>. The DataFrame has columns such as{" "}
            <code>state</code>, <code>execution_duration</code>,{" "}
            <code>inputs</code>, <code>outputs</code>, <code>tags</code>, and{" "}
            <code>assessments</code>, so ordinary pandas code gives you custom
            analytics:
          </p>

          <CodeBlock
            code={`df = mlflow.search_traces(max_results=1000)

print(df["state"].value_counts())
print(df["execution_duration"].quantile([0.5, 0.95, 0.99]))`}
          />

          <p>
            See the{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/search-traces"}>
              search traces documentation
            </Link>{" "}
            for the full list of fields and operators.
          </p>

          <h2 id="dashboard">
            How Do I Build an LLM Trace Dashboard for Latency, Tokens, and Cost?
          </h2>

          <p>
            The experiment Dashboard aggregates traces over a time range you
            choose, with a selectable time unit. It has three tabs:
          </p>

          <ul>
            <li>
              <strong>Usage:</strong> request counts, latency, errors, token
              usage, tokens per trace, and cost.
            </li>
            <li>
              <strong>Quality:</strong> a summary and one chart per assessment
              type, built from the scorers and feedback attached to your traces.
            </li>
            <li>
              <strong>Tool calls:</strong> total calls, average latency, success
              rate, and per-tool usage and error rate for agents.
            </li>
          </ul>

          <p>
            Token usage and cost are recorded automatically for supported LLM
            providers, so the charts fill in without extra instrumentation. The
            dashboard needs a SQL-backed tracking store (SQLite, PostgreSQL, or
            MySQL); the default file store does not render the charts. See the{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL + "tracing/observe-with-traces/dashboard"
              }
            >
              dashboard documentation
            </Link>{" "}
            and the{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/token-usage-cost/"}>
              token usage and cost guide
            </Link>
            .
          </p>

          <h2 id="sessions">How Do I Track Sessions and Users?</h2>

          <p>
            Chat applications and agents handle conversations, not isolated
            requests. Tag each trace with a session ID and a user ID and MLflow
            groups them, so you can look at a whole conversation, filter by
            user, and analyze multi-turn behavior.
          </p>

          <CodeBlock
            code={`import mlflow

@mlflow.trace
def chat(message: str, user_id: str, session_id: str):
    mlflow.update_current_trace(session_id=session_id, user=user_id)
    return my_agent(message)

# Or wrap auto-instrumented code
with mlflow.tracing.context(session_id="session-123", user="user-456"):
    agent.invoke("What is the capital of France?")

# Retrieve complete multi-turn sessions
sessions = mlflow.search_sessions(max_results=50)`}
          />

          <p>
            The <code>session_id</code> and <code>user</code> arguments need
            MLflow 3.11.0 or later. On older versions, set the{" "}
            <code>mlflow.trace.session</code> and <code>mlflow.trace.user</code>{" "}
            metadata keys instead. Details are in the{" "}
            <Link
              href={MLFLOW_GENAI_DOCS_URL + "tracing/track-users-sessions/"}
            >
              user and session tracking guide
            </Link>
            .
          </p>

          <h2 id="assessments">How Do I Measure Quality in Traces?</h2>

          <p>
            Latency and errors tell you whether a request ran. They do not tell
            you whether the answer was good. MLflow stores quality signals on
            the trace itself as <strong>assessments</strong>: feedback from LLM
            judges, code-based checks, or human reviewers, and expectations that
            record ground truth. Once attached, assessments are searchable and
            chartable like any other trace field.
          </p>

          <CodeBlock
            code={`from mlflow.entities import AssessmentSource, AssessmentSourceType

mlflow.log_feedback(
    trace_id=trace_id,
    name="rating",
    value="positive",
    rationale="Answer matched the retrieved policy document.",
    source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id="reviewer@example.com",
    ),
)

# Later: find every trace a reviewer rated positively
good = mlflow.search_traces(filter_string="feedback.rating = 'positive'")`}
          />

          <p>
            <Link href="/llm-as-a-judge">LLM judges</Link> can score traces
            automatically, and review queues let you route traces to human
            reviewers. Read more in the{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "assessments/feedback"}>
              feedback collection guide
            </Link>
            .
          </p>

          <h2 id="issue-detection">How Does Automatic Issue Detection Work?</h2>

          <p>
            Filters and charts help when you know what to look for. Automatic
            issue detection helps when you do not. It analyzes a selection of
            traces with an LLM, groups related failures into named issues such
            as incorrect information or ungrounded claims, and attaches the
            traces and span-level evidence behind each one. You review an issue
            list instead of thousands of traces, and each issue links back to
            the traces that support it.
          </p>

          <Figure
            src={IssuesImg}
            alt="MLflow Issues page listing detected issues for an LLM application, with severity, category, and evidence from traces"
            caption="Detected issues group related failures and link to evidence in the traces. Demo data."
          />

          <p>
            Issue detection runs from the Traces tab and uses the model you
            choose, so it incurs LLM calls. See the{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL + "eval-monitor/ai-insights/detect-issues"
              }
            >
              automatic issue detection guide
            </Link>{" "}
            and{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "eval-monitor/ai-insights/ai-issue-discovery"
              }
            >
              AI issue discovery
            </Link>
            .
          </p>

          <h2 id="platforms">
            What Should I Look for in an LLM Trace Analytics Platform?
          </h2>

          <p>
            Many tools offer GenAI observability and LLM monitoring. When you
            compare platforms for LLM trace analytics, these criteria matter
            most, and they are where an open-source platform like MLflow is
            designed to help:
          </p>

          <Table
            head={["Criterion", "What to check", "In MLflow"]}
            rows={[
              [
                "Ownership of data",
                "Can you self-host and keep traces in your own database?",
                "Open source under Apache 2.0, self-hosted on SQLite, PostgreSQL, or MySQL.",
              ],
              [
                "Framework coverage",
                "Does tracing work across the LLM providers and agent frameworks you use?",
                "Automatic tracing integrations plus manual tracing for any Python code.",
              ],
              [
                "Search depth",
                "Can you filter on spans, tags, sessions, and quality scores as well as time?",
                "SQL-like filters on status, duration, spans, tags, metadata, feedback, and expectations.",
              ],
              [
                "Quality signals",
                "Do judge scores and human feedback live next to the trace?",
                "Assessments attach to traces and chart on the Quality tab.",
              ],
              [
                "Evaluation loop",
                "Can production failures become test cases?",
                "Traces feed MLflow evaluation datasets and scorers in the same platform.",
              ],
            ]}
          />

          <p>
            For a side-by-side look at specific tools, see{" "}
            <Link href="/top-5-agent-observability-tools">
              top agent observability tools
            </Link>
            .
          </p>

          <h2 id="getting-started">
            How Do I Get Started With Trace Analytics?
          </h2>

          <p>
            Trace analytics needs traces first. Enable tracing with one line for
            your framework, then open the experiment in the MLflow UI:
          </p>

          <CodeBlock
            code={`import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")  # SQL store for the Dashboard
mlflow.set_experiment("my-agent")
mlflow.openai.autolog()  # or langchain, anthropic, and other integrations

# Run your application, then start the UI:
#   mlflow server --backend-store-uri sqlite:///mlflow.db`}
          />

          <p>
            Open the experiment, then use Traces to search, Dashboard for
            trends, and Issues to review detected problems. The{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/quickstart/"}>
              tracing quickstart
            </Link>{" "}
            walks through the setup, and{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/prod-tracing"}>
              production tracing
            </Link>{" "}
            covers sampling and deployment.
          </p>

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
              <Link href="/articles/llm-trace-analytics">
                LLM Trace Analytics: Where to Find It and What to Track
              </Link>
            </li>
            <li>
              <Link href="/llm-tracing">LLM Tracing</Link>
            </li>
            <li>
              <Link href="/llm-observability">AI Observability</Link>
            </li>
            <li>
              <Link href="/ai-monitoring">AI Monitoring</Link>
            </li>
            <li>
              <Link href="/llm-evaluation">
                LLM Evaluation and Agent Evaluation
              </Link>
            </li>
            <li>
              <Link href="/llm-as-a-judge">LLM as a Judge</Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
                MLflow Tracing documentation
              </Link>
            </li>
            <li>
              <Link href="/genai">MLflow for Agents and LLMs Overview</Link>
            </li>
            <li>
              <Link href="/agent-observability">Agent Observability</Link>
            </li>
            <li>
              <Link href="/agent-tracing">Agent Tracing</Link>
            </li>
            <li>
              <Link href="/llm-cost-tracking">LLM Cost Tracking</Link>
            </li>
            <li>
              <Link href="/llm-guardrails">LLM Guardrails</Link>
            </li>
          </ul>
        </div>

        <ArticleSidebar />
        <SocialLinksFooter />
      </div>
    </>
  );
}
