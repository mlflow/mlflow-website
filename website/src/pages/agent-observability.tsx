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
import ToolCallsDashboard from "@site/static/img/agent-observability/mlflow-agent-observability-tool-calls.png";

// Built from parts: check-links would otherwise 4xx on this page's own URL until it is live.
const PAGE_URL = "https://mlflow.org" + "/agent-observability";

const SEO_TITLE = "Agent Observability: Monitor and Debug AI Agents | MLflow";
const SEO_DESCRIPTION =
  "Agent observability shows what your AI agents do, why they fail, and how well they perform. See how open-source MLflow traces agents, tool calls, quality, and cost.";

const faqs: { question: string; answer: string }[] = [
  {
    question: "What is agent observability?",
    answer:
      "Agent observability is the practice of capturing and analyzing what an AI agent does at runtime, including every LLM call, tool call, retrieval step, and decision, so you can debug failures, measure quality, and control cost. It builds on tracing and adds tool-level analytics, quality scoring, and issue detection. MLflow provides all of these in one open-source platform.",
  },
  {
    question: "How is agent observability different from LLM observability?",
    answer:
      "LLM observability focuses on individual model calls: prompts, completions, latency, and token usage. Agent observability covers the whole multi-step run, including which tools the agent chose, in what order, with which arguments, and whether the task was completed. An agent can make several correct-looking LLM calls and still fail the task, so you need the full execution graph, not just per-call metrics.",
  },
  {
    question: "How do I add observability to an AI agent with MLflow?",
    answer:
      "Call the autolog function for your framework, for example mlflow.openai.autolog() or mlflow.langchain.autolog() (LangChain autologging also traces LangGraph). Then wrap your own tools and agent entry point with the @mlflow.trace decorator and set span_type to TOOL or AGENT. Every run is recorded as a trace with nested spans that you can inspect in the MLflow UI.",
  },
  {
    question: "What should I monitor for an AI agent in production?",
    answer:
      "Monitor four things: errors and latency per trace, tool-call success rate and latency per tool, quality scores from automated judges (for example correctness, safety, and tool-call correctness), and token usage and cost. Add session and user identifiers to traces so you can follow multi-turn conversations and find the users affected by a failure.",
  },
  {
    question: "How do I find out which tool is failing or slow?",
    answer:
      "Open the Tool Calls tab of the experiment Dashboard. It lists total tool calls, average latency, success rate, and failed calls, with a per-tool performance summary, usage over time, latency comparison, and error rate. This view is built from the TOOL spans in your traces, so tools you instrument with span_type TOOL appear automatically.",
  },
  {
    question: "Can MLflow evaluate agent quality automatically in production?",
    answer:
      "Yes. You can register an LLM judge or scorer, such as ToolCallCorrectness, and start it with a sampling rate so that a share of incoming traces is scored automatically. Scores are attached to the traces as assessments and appear in the Quality tab of the Dashboard and in the Traces table.",
  },
  {
    question:
      "Can MLflow find problems in my agent without me knowing what to look for?",
    answer:
      "MLflow includes AI issue detection. The Detect Issues action in the Traces and Issues pages analyzes the traces in an experiment and groups recurring problems, such as incorrect information or ungrounded claims, with evidence from specific traces. Use it to decide which failures to turn into evaluation test cases.",
  },
  {
    question: "Does MLflow work with OpenTelemetry?",
    answer:
      "Yes. MLflow Tracing is compatible with OpenTelemetry, so you can export MLflow traces to other OpenTelemetry-compatible backends and ingest OpenTelemetry traces into MLflow. This lets you add agent observability without being locked into a single vendor's SDK.",
  },
  {
    question: "Which agent frameworks does MLflow support?",
    answer:
      "MLflow has one-line automatic tracing integrations for popular agent frameworks and model providers, including OpenAI (and the OpenAI Agents SDK), LangChain and LangGraph, LlamaIndex, CrewAI, AutoGen, DSPy, Pydantic AI, Anthropic, Gemini, and AWS Bedrock. Code that is not covered can be traced with the @mlflow.trace decorator or the mlflow.start_span context manager.",
  },
  {
    question: "Is MLflow agent observability free?",
    answer:
      "Yes. MLflow is open source under the Apache 2.0 license and is part of the Linux Foundation. Tracing, the dashboard, evaluation, and issue detection are included at no cost, and you control where your trace data is stored. Your own infrastructure and any LLM calls made by judges are the only costs. Managed MLflow is also available on Databricks.",
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
    "Open-source platform for agent observability: tracing, tool-call analytics, quality monitoring, and issue detection for AI agents and LLM applications.",
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
          {({ tokens, getLineProps, getTokenProps }) => (
            <pre
              style={{
                margin: 0,
                background: "transparent",
                fontSize: "13px",
                lineHeight: 1.6,
              }}
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

const cell = {
  padding: "12px 16px",
  borderBottom: "1px solid #e5e7eb",
  textAlign: "left" as const,
  verticalAlign: "top" as const,
  fontSize: "15px",
  lineHeight: 1.5,
  color: "#3d3d3d",
};

const signals: [string, string, string][] = [
  [
    "Traces",
    "Every LLM call, tool call, and retrieval step as nested spans, with inputs, outputs, latency, and errors",
    "Traces tab, trace detail view",
  ],
  [
    "Tool calls",
    "Calls, success rate, latency, and error rate for each tool, taken from TOOL spans",
    "Dashboard > Tool Calls",
  ],
  [
    "Quality",
    "Scores from LLM judges and code-based scorers, plus human feedback, attached to traces as assessments",
    "Dashboard > Quality, Traces table",
  ],
  [
    "Cost and usage",
    "Token counts and estimated cost per span, per trace, and over time",
    "Dashboard > Usage",
  ],
  [
    "Sessions and users",
    "Multi-turn conversations grouped by session ID and user ID",
    "Sessions tab",
  ],
  [
    "Issues",
    "Recurring failure patterns found automatically across traces, with evidence",
    "Issues tab",
  ],
];

export default function AgentObservability() {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="article:modified_time" content="2026-11-26" />
        <meta property="og:url" content={PAGE_URL} />
        <link rel="canonical" href={PAGE_URL} />
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
          <h1>Agent Observability</h1>

          <p
            style={{
              fontSize: "14px",
              color: "#6b7280",
              margin: "-8px 0 24px",
            }}
          >
            Last updated: 2026-11-26
          </p>

          <p>
            <strong>Agent observability</strong> is the practice of capturing
            what an AI agent does at runtime, including every LLM call, tool
            call, and decision, so you can debug failures, measure quality, and
            control cost. MLflow provides open-source agent observability
            through{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
              MLflow Tracing
            </Link>
            , a tool-call dashboard, automated quality scoring, and AI issue
            detection, all in one platform and with no per-trace fees.
          </p>

          <p>
            Agents are different from single LLM calls. They plan, call tools,
            read the results, and loop until they decide they are done. When an
            agent gives a wrong answer, the cause is usually several steps
            earlier: a tool returned stale data, a retrieval missed the right
            document, or a loop ran twice as long as it should. Agent
            observability makes those steps visible.
          </p>

          <div className="info-box">
            <p>
              <strong>In short:</strong> instrument your agent once, then use
              traces to debug a single run, the dashboard to watch tools and
              cost over time, scorers to track quality, and issue detection to
              find problems you did not know to look for.
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
                <a href="#what-is-agent-observability">
                  What is agent observability?
                </a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#what-to-observe">What to observe</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#instrument">Instrument an agent</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#tool-calls">Monitor tool calls</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#quality">Monitor quality</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#issues">Find issues automatically</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#workflow">Debug and improve loop</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#choose">Choose a tool</a>
              </li>
              <li style={{ marginBottom: "0" }}>
                <a href="#faq">FAQ</a>
              </li>
            </ul>
          </div>

          <h2 id="what-is-agent-observability" data-toc="What Is It?">
            What Is Agent Observability?
          </h2>

          <p>
            Observability means you can understand a system's internal state
            from the data it emits. For an agent, that data is the{" "}
            <Link href="/llm-tracing">trace</Link>: a tree of spans that records
            each LLM call, tool call, and retrieval, along with inputs, outputs,
            timing, token usage, and errors. Agent observability adds three
            layers on top of tracing:
          </p>

          <ul>
            <li>
              <strong>Analytics</strong> over many traces, such as tool success
              rates, latency percentiles, and cost trends.
            </li>
            <li>
              <strong>Quality measurement</strong> with LLM judges, code-based
              scorers, and human feedback, so "it ran" becomes "it answered
              correctly".
            </li>
            <li>
              <strong>Issue detection</strong> that surfaces recurring failure
              patterns across traces instead of waiting for a user to report
              them.
            </li>
          </ul>

          <p>
            If you want the model-call view only, see{" "}
            <Link href="/ai-observability">AI observability</Link>. For a
            comparison of products that offer these features, read{" "}
            <Link href="/top-5-agent-observability-tools">
              Top 5 Agent Observability Tools
            </Link>
            .
          </p>

          <h2 id="what-to-observe" data-toc="What to Observe">
            What Should You Observe in an AI Agent?
          </h2>

          <p>
            A useful agent observability setup covers six signals. Each one maps
            to a view in the MLflow UI.
          </p>

          <div style={{ overflowX: "auto", margin: "24px 0 40px" }}>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                border: "1px solid #e5e7eb",
              }}
            >
              <thead>
                <tr style={{ background: "#f9fafb" }}>
                  <th style={cell}>Signal</th>
                  <th style={cell}>What MLflow captures</th>
                  <th style={cell}>Where to see it</th>
                </tr>
              </thead>
              <tbody>
                {signals.map(([signal, captured, where]) => (
                  <tr key={signal}>
                    <td style={{ ...cell, fontWeight: 600, color: "#1a1a1a" }}>
                      {signal}
                    </td>
                    <td style={cell}>{captured}</td>
                    <td style={cell}>{where}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h2 id="instrument" data-toc="Instrument Your Agent">
            How Do You Instrument an AI Agent?
          </h2>

          <p>
            Start with automatic tracing for your framework. MLflow ships
            one-line{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL + "tracing/app-instrumentation/automatic"
              }
            >
              autolog integrations
            </Link>{" "}
            for OpenAI, LangChain and LangGraph, LlamaIndex, CrewAI, AutoGen,
            DSPy, Anthropic, and others. Then mark your own tools and agent
            entry point with the{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "tracing/app-instrumentation/manual-tracing"
              }
            >
              @mlflow.trace
            </Link>{" "}
            decorator. Setting the span type to TOOL is what makes a function
            show up in the tool-call dashboard.
          </p>

          <CodeBlock
            code={`import mlflow
from openai import OpenAI

mlflow.set_experiment("support-agent")
mlflow.openai.autolog()  # traces every OpenAI call

client = OpenAI()


@mlflow.trace(span_type="TOOL")
def lookup_order(order_id: str) -> dict:
    return {"order_id": order_id, "status": "shipped"}


@mlflow.trace(span_type="AGENT")
def support_agent(question: str, session_id: str, user_id: str) -> str:
    # Group multi-turn conversations and attribute traces to users
    mlflow.update_current_trace(session_id=session_id, user=user_id)

    order = lookup_order("A-1001")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Order data: {order}"},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content`}
          />

          <p>
            Each call to <code>support_agent</code> is now one trace. The agent
            span contains the tool span and the LLM span, with inputs, outputs,
            latency, and token usage on each. Passing a session ID lets MLflow
            group the turns of one conversation, which matters for agents that
            fail only after several turns. See{" "}
            <Link
              href={MLFLOW_GENAI_DOCS_URL + "tracing/track-users-sessions/"}
            >
              tracking users and sessions
            </Link>{" "}
            for details. Because MLflow Tracing is built on OpenTelemetry
            concepts, you can also{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "tracing/app-instrumentation/opentelemetry"
              }
            >
              export and ingest OpenTelemetry traces
            </Link>
            .
          </p>

          <h2 id="tool-calls" data-toc="Monitor Tool Calls">
            How Do You Monitor Agent Tool Calls?
          </h2>

          <p>
            Tool calls are where agents most often go wrong: the wrong tool, the
            wrong arguments, a slow API, or an error the model quietly ignores.
            The Tool Calls tab of the experiment{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL + "tracing/observe-with-traces/dashboard/"
              }
            >
              dashboard
            </Link>{" "}
            summarizes every TOOL span across your traces: total calls, average
            latency, success rate, failed calls, a per-tool performance table,
            usage and latency over time, and error rate per tool.
          </p>

          <figure style={{ margin: "40px 0" }}>
            <img
              src={ToolCallsDashboard}
              alt="MLflow dashboard Tool Calls tab showing total tool calls, average latency, success rate, a tool performance summary table, tool usage over time, and latency comparison charts for demo agent data"
              style={{
                width: "100%",
                border: "1px solid #e5e7eb",
                borderRadius: "8px",
              }}
            />
            <figcaption
              style={{
                marginTop: "12px",
                fontSize: "14px",
                color: "#6b7280",
                textAlign: "center",
              }}
            >
              The Tool Calls tab in the MLflow dashboard, shown with MLflow's
              built-in demo data.
            </figcaption>
          </figure>

          <p>
            When a tool looks slow or error-prone, click through to the traces
            that called it and read the exact inputs and outputs in the trace
            detail view. The dashboard also has Usage and Quality tabs for
            request volume, latency, errors, token usage, cost, and scorer
            results.
          </p>

          <h2 id="quality" data-toc="Monitor Quality">
            How Do You Monitor Agent Quality in Production?
          </h2>

          <p>
            Speed and error rate do not tell you whether the agent answered
            correctly. MLflow can run{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL + "eval-monitor/automatic-evaluations/"
              }
            >
              automatic evaluations
            </Link>{" "}
            on a sample of live traces. Register a scorer, choose a sampling
            rate, and MLflow attaches the scores to each trace as an assessment.
            Built-in scorers include ToolCallCorrectness and ToolCallEfficiency
            for agent behavior, plus Safety, Correctness, and Completeness for
            responses.
          </p>

          <CodeBlock
            code={`import mlflow
from mlflow.genai.scorers import ToolCallCorrectness, ScorerSamplingConfig

mlflow.set_experiment("support-agent")

tool_judge = ToolCallCorrectness(model="gateway:/my-llm-endpoint")
registered = tool_judge.register(name="tool_call_correctness")
registered.start(
    sampling_config=ScorerSamplingConfig(sample_rate=0.5),  # score 50% of traces
)`}
          />

          <p>
            Use the same scorers offline with{" "}
            <code>mlflow.genai.evaluate()</code> to test changes before you ship
            them. Learn more on the{" "}
            <Link href="/llm-evaluation">LLM and agent evaluation</Link> page
            and in the guide to{" "}
            <Link href="/llm-as-a-judge">LLM-as-a-judge</Link>.
          </p>

          <h2 id="issues" data-toc="Find Issues Automatically">
            How Do You Find Agent Issues Automatically?
          </h2>

          <p>
            Dashboards answer questions you already have. Issue detection helps
            with the ones you do not. The Detect Issues action analyzes the
            traces in an experiment and groups recurring problems into issues,
            each with a severity, the affected traces, and the evidence that led
            to it. You can read more in the{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "eval-monitor/ai-insights/ai-issue-discovery"
              }
            >
              AI issue discovery documentation
            </Link>
            .
          </p>

          <h2 id="workflow" data-toc="Debug and Improve Loop">
            What Is the Debug and Improve Loop?
          </h2>

          <p>Observability is most useful when it feeds a repeatable loop:</p>

          <ol>
            <li>
              <strong>Observe.</strong> Trace every run and watch the dashboard
              for changes in tool errors, latency, cost, or quality scores.
            </li>
            <li>
              <strong>Diagnose.</strong> Open the failing traces, find the span
              where the run went wrong, and group failures with issue detection.
            </li>
            <li>
              <strong>Evaluate.</strong> Turn failing traces into an evaluation
              dataset and write a scorer for the failure.
            </li>
            <li>
              <strong>Fix and compare.</strong> Change the prompt, tool, or
              model, re-run the evaluation, and compare versions side by side
              before deploying.
            </li>
            <li>
              <strong>Monitor.</strong> Keep an online scorer running so a
              regression shows up in the dashboard, not in a support ticket.
            </li>
          </ol>

          <h2 id="choose" data-toc="Choosing a Tool">
            How Do You Choose an Agent Observability Tool?
          </h2>

          <p>
            Whichever product you evaluate, compare it on the same questions:
          </p>

          <ul>
            <li>
              <strong>Open source and self-hostable.</strong> Can you run it
              yourself and keep trace data in your own infrastructure?
            </li>
            <li>
              <strong>Standards-based.</strong> Does it support OpenTelemetry,
              so traces are not locked into one vendor's SDK?
            </li>
            <li>
              <strong>Framework coverage.</strong> Does it trace the agent
              frameworks and model providers you already use with minimal code?
            </li>
            <li>
              <strong>Tool-level visibility.</strong> Can you see success rate
              and latency per tool, not only per request?
            </li>
            <li>
              <strong>Quality and issues in the same place.</strong> Can you
              score traces, collect feedback, and detect issues without moving
              data between tools?
            </li>
          </ul>

          <p>
            MLflow is built to meet each of these as an open-source platform.
            For a side-by-side comparison with other tools, see{" "}
            <Link href="/top-5-agent-observability-tools">
              Top 5 Agent Observability Tools
            </Link>
            .
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

          <h2>Where Can You Learn More?</h2>

          <ul>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/quickstart/"}>
                Tracing Quickstart
              </Link>
            </li>
            <li>
              <Link
                href={
                  MLFLOW_GENAI_DOCS_URL +
                  "tracing/observe-with-traces/dashboard/"
                }
              >
                Dashboard (Usage, Quality, and Tool Calls)
              </Link>
            </li>
            <li>
              <Link
                href={
                  MLFLOW_GENAI_DOCS_URL + "eval-monitor/automatic-evaluations/"
                }
              >
                Automatic Evaluations for Production Traces
              </Link>
            </li>
            <li>
              <Link href="/llm-tracing">LLM Tracing FAQ</Link>
            </li>
            <li>
              <Link href="/ai-observability">AI Observability FAQ</Link>
            </li>
            <li>
              <Link href="/ai-monitoring">AI Monitoring FAQ</Link>
            </li>
            <li>
              <Link href="/llm-evaluation">LLM and Agent Evaluation</Link>
            </li>
            <li>
              <Link href="/top-5-agent-observability-tools">
                Top 5 Agent Observability Tools Comparison
              </Link>
            </li>
            <li>
              <Link href="/genai">MLflow for Agents and LLMs Overview</Link>
            </li>
          </ul>
        </div>

        <ArticleSidebar />
        <SocialLinksFooter />
      </div>
    </>
  );
}
