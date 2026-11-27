import { useState } from "react";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../components/Header/Header";
import { SocialLinksFooter } from "../components/SocialLinksFooter/SocialLinksFooter";
import { ArticleSidebar } from "../components/ArticleSidebar/ArticleSidebar";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import AgentTraceImage from "@site/static/img/agent-tracing/mlflow-demo-trace-detail.png";
import { CopyButton } from "../components/CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";

const SEO_TITLE = "Agent Tracing: Trace Every Step of Your AI Agent | MLflow";
const SEO_DESCRIPTION =
  "Agent tracing records every LLM call, tool call, and handoff in an AI agent run. See how open-source MLflow Tracing captures agents with one line of code.";

const faqs: { question: string; answer: string }[] = [
  {
    question: "What is agent tracing?",
    answer:
      "Agent tracing records everything an AI agent does during a run as a structured trace: each LLM call, tool call, retrieval, and handoff to another agent, with inputs, outputs, timing, and token usage. The steps are nested as a tree of spans, so you can see the order of decisions the agent made and where a run went wrong. MLflow Tracing provides agent tracing as open source.",
  },
  {
    question: "How is agent tracing different from LLM tracing?",
    answer:
      "LLM tracing usually captures a single request and response. Agent tracing captures a multi-step run where each model call decides the next action, such as which tool to call. An agent trace is a tree of spans with an agent span at the root, so it can show wrong tool choices, loops, and redundant calls that a single-call trace cannot.",
  },
  {
    question: "How do I trace an AI agent with MLflow?",
    answer:
      "For supported frameworks, call the framework's autolog function once before running the agent, for example mlflow.langchain.autolog() for LangGraph or mlflow.openai.autolog() for the OpenAI Agents SDK. For a custom agent, decorate your functions with @mlflow.trace and set a span type such as AGENT or TOOL, or wrap code blocks with mlflow.start_span. Then open the experiment's Traces tab in the MLflow UI.",
  },
  {
    question: "Which agent frameworks does MLflow trace automatically?",
    answer:
      "MLflow has one-line autologging for LangGraph, the OpenAI Agents SDK, CrewAI, AutoGen, AG2, Pydantic AI, Strands Agents, smolagents, Agno, and the Claude Agent SDK for Python, plus model SDKs such as OpenAI and Anthropic. Other frameworks and languages can send traces through OpenTelemetry. The integrations page in the MLflow documentation has the current list.",
  },
  {
    question: "Can I trace multi-agent systems?",
    answer:
      "Yes. When one agent calls another, the sub-agent's run is recorded as child spans of the calling agent, so the trace shows the full hierarchy. If your agents run in separate services, MLflow's distributed tracing connects their spans into a single trace. You can also attach session and user IDs so that the turns of a conversation are grouped together.",
  },
  {
    question: "What data does an agent trace contain?",
    answer:
      "Each span records its inputs and outputs, start time and duration, status or exception, and custom attributes. Model call spans also record token usage, and the MLflow UI shows a cost estimate. Traces can carry tags, user and session metadata, and feedback from people or automated judges. Because traces can contain user content, you should review what your agent logs before sending production traffic.",
  },
  {
    question: "How do I find and debug failed agent runs?",
    answer:
      "Open the Traces tab in the MLflow UI, filter by status or text, and expand a failing trace to read each span's inputs and outputs. In code, mlflow.search_traces accepts a filter string, for example trace.status = 'ERROR', so you can pull failing runs into a notebook or a script and analyze them.",
  },
  {
    question: "Can I evaluate agents using their traces?",
    answer:
      "Yes. mlflow.genai.evaluate accepts traces as data, so you can score existing agent runs without re-running them. Scorers see the full trace, which lets them judge behavior such as tool selection and redundant tool calls, not only the final answer. See the LLM evaluation page for the built-in judges and custom scorers.",
  },
  {
    question: "Does agent tracing work in production?",
    answer:
      "Yes. The same instrumentation runs in production. For services, the lightweight mlflow-tracing package has a smaller footprint than the full MLflow install and sends traces to a shared tracking server. You can add tags for environment and version so you can compare production behavior across releases.",
  },
  {
    question: "Is MLflow agent tracing free?",
    answer:
      "Yes. MLflow is open source under the Apache 2.0 license and is a Linux Foundation project, and tracing is included at no cost. You pay only for the infrastructure you run it on and any model calls your agent makes. Managed MLflow is also available from some platform vendors if you prefer not to host it yourself.",
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
      text: faq.answer,
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
    "Open-source platform for agent tracing, LLM observability, evaluation, and experiment tracking.",
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
              style={{
                ...style,
                background: "transparent",
                margin: 0,
                fontSize: "13px",
                lineHeight: "1.6",
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

function SimpleTable({
  head,
  rows,
}: {
  head: string[];
  rows: React.ReactNode[][];
}) {
  const cell = { padding: "12px 16px", textAlign: "left" as const };
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
            {head.map((h, i) => (
              <th
                key={i}
                style={{ ...cell, fontWeight: 600, color: "#1a1a1a" }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, r) => (
            <tr key={r} style={{ borderBottom: "1px solid #e2e8f0" }}>
              {row.map((c, i) => (
                <td
                  key={i}
                  style={{
                    ...cell,
                    color: "#3d3d3d",
                    fontWeight: i === 0 ? 600 : 400,
                    verticalAlign: "top",
                  }}
                >
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function AgentTracing() {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://mlflow.org/agent-tracing" />
        <link rel="canonical" href="https://mlflow.org/agent-tracing" />
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
          <h1>Agent Tracing</h1>

          <p>
            <strong>Agent tracing</strong> records every step an AI agent takes
            (each LLM call, tool call, retrieval, and handoff) as a structured,
            nested trace with inputs, outputs, latency, and token usage.{" "}
            <Link href="/llm-tracing">MLflow Tracing</Link> is an open-source
            way to do this. It captures agent runs automatically for popular
            agent frameworks with a single line of code, and supports manual
            instrumentation for custom agents.
          </p>

          <p>
            An agent decides what to do next at run time, so the same request
            can take a different path every time. When an answer is wrong, the
            final output rarely shows why. A trace shows the whole run: which
            tools the agent picked, what each one returned, and which step went
            off track.
          </p>

          <p style={{ color: "#6b7280", fontSize: "14px" }}>
            Last updated: November 26, 2026
          </p>

          <div className="info-box">
            <p>
              <strong>In short:</strong> call{" "}
              <code>mlflow.&lt;framework&gt;.autolog()</code> once, run your
              agent, and open the trace in the MLflow UI. Every agent run
              appears as a tree of spans you can search, debug, and evaluate.
              MLflow is Apache 2.0 licensed, a Linux Foundation project, and you
              can self-host it.
            </p>
          </div>

          <div style={{ margin: "40px 0", textAlign: "center" }}>
            <img
              src={AgentTraceImage}
              alt="MLflow trace view of a multi-step agent run: a chat_agent root span with LLM calls and tool calls such as search_traces and analyze_retrieval_quality, with latency, token counts, and cost per span"
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
              A multi-step agent trace in the MLflow UI (generated with{" "}
              <code>mlflow demo</code>): the span tree on the left, the selected
              span's inputs and outputs on the right.
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
                <a href="#what-is-agent-tracing">What is agent tracing?</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#what-a-trace-captures">
                  What Does an Agent Trace Capture?
                </a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#quickstart">How Do You Trace an Agent in One Line?</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#frameworks">
                  Which Agent Frameworks Does MLflow Trace?
                </a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#custom-agents">How Do You Trace a Custom Agent?</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#multi-agent-sessions">
                  How Do You Trace Multi-Agent Systems and Sessions?
                </a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#debug-and-evaluate">
                  How Do You Debug and Evaluate Agents with Traces?
                </a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#production">Does Agent Tracing Work in Production?</a>
              </li>
              <li style={{ marginBottom: "0" }}>
                <a href="#faq">FAQ</a>
              </li>
            </ul>
          </div>

          <h2 id="what-is-agent-tracing">What Is Agent Tracing?</h2>

          <p>
            Agent tracing is the practice of capturing an AI agent's full
            execution as a trace: a tree of timed steps called{" "}
            <strong>spans</strong>. The root span is the agent run. Its children
            are the model calls, tool calls, retrievals, and sub-agent calls the
            agent made along the way, nested in the order they happened.
          </p>

          <p>
            This is the main difference from plain{" "}
            <Link href="/llm-tracing">LLM tracing</Link>. An LLM trace usually
            records one request and one response. An agent trace records a loop
            of reasoning and actions, where each model call decides the next
            tool call. Seeing that loop is what lets you debug it.
          </p>

          <SimpleTable
            head={["", "LLM call tracing", "Agent tracing"]}
            rows={[
              [
                "What is recorded",
                "One prompt and one response",
                "The whole run: every model call, tool call, retrieval, and handoff",
              ],
              [
                "Shape of the trace",
                "Usually a single span",
                "A tree of nested spans, with an agent span at the root",
              ],
              [
                "Typical question",
                "Was this response good, and what did it cost?",
                "Why did the agent pick this tool, and where did the run go wrong?",
              ],
              [
                "Failures you can find",
                "Bad prompt, slow model, high token use",
                "Wrong tool choice, bad tool arguments, loops, redundant calls, lost context between steps",
              ],
            ]}
          />

          <h2 id="what-a-trace-captures">What Does an Agent Trace Capture?</h2>

          <p>
            Every span in an MLflow trace has a type, so the UI and the search
            API can tell a model call from a tool call. The span types that
            matter most for agents are:
          </p>

          <SimpleTable
            head={["Span type", "What it represents in an agent"]}
            rows={[
              [
                "AGENT",
                "An agent run or a sub-agent, usually the root of the trace",
              ],
              [
                "LLM / CHAT_MODEL",
                "A call to a language model, with prompt, response, and token usage",
              ],
              [
                "TOOL",
                "A tool or function the agent called, with its arguments and result",
              ],
              ["RETRIEVER", "A search over a vector store or knowledge base"],
              ["CHAIN", "A step that wires other steps together"],
              ["GUARDRAIL", "A safety or policy check on an input or output"],
            ]}
          />

          <p>
            On each span MLflow records the inputs and outputs, start time and
            duration, status and any exception, and custom attributes. For model
            calls it also records token counts, and the UI shows the cost
            estimate for each call and for the whole trace. Traces can also
            carry tags, user and session IDs, and{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
              human or automated feedback
            </Link>{" "}
            attached to the trace or to a single span.
          </p>

          <h2 id="quickstart">How Do You Trace an Agent in One Line?</h2>

          <p>
            For supported frameworks, you enable tracing by calling that
            framework's <code>autolog()</code> function before you run the
            agent. This example uses LangGraph, but every framework in the next
            section works the same way.
          </p>

          <CodeBlock
            code={`import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-agent")

# One line: every graph step, model call, and tool call becomes a span
mlflow.langchain.autolog()

# Run your compiled LangGraph agent as usual
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)`}
          />

          <p>
            Start a local MLflow server with <code>mlflow server</code>, run the
            script, and open the experiment's Traces tab. Each run shows up as a
            trace you can expand span by span. The{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/quickstart/"}>
              tracing quickstart
            </Link>{" "}
            walks through the full setup.
          </p>

          <h2 id="frameworks">Which Agent Frameworks Does MLflow Trace?</h2>

          <p>
            MLflow has one-line autologging for the ten agent frameworks below,
            plus model SDKs such as OpenAI and Anthropic. Each framework has a
            dedicated integration:
          </p>

          <SimpleTable
            head={["Framework", "Enable tracing", "Guide"]}
            rows={[
              [
                "LangGraph",
                "mlflow.langchain.autolog()",
                <Link
                  href={
                    MLFLOW_GENAI_DOCS_URL +
                    "tracing/integrations/listing/langgraph/"
                  }
                >
                  LangGraph tracing
                </Link>,
              ],
              [
                "OpenAI Agents SDK",
                "mlflow.openai.autolog()",
                <Link
                  href={
                    MLFLOW_GENAI_DOCS_URL +
                    "tracing/integrations/listing/openai-agent/"
                  }
                >
                  OpenAI Agents tracing
                </Link>,
              ],
              [
                "CrewAI",
                "mlflow.crewai.autolog()",
                <Link
                  href={
                    MLFLOW_GENAI_DOCS_URL +
                    "tracing/integrations/listing/crewai/"
                  }
                >
                  CrewAI tracing
                </Link>,
              ],
              [
                "AutoGen",
                "mlflow.autogen.autolog()",
                <Link
                  href={
                    MLFLOW_GENAI_DOCS_URL +
                    "tracing/integrations/listing/autogen/"
                  }
                >
                  AutoGen tracing
                </Link>,
              ],
              [
                "AG2",
                "mlflow.ag2.autolog()",
                <Link
                  href={
                    MLFLOW_GENAI_DOCS_URL + "tracing/integrations/listing/ag2/"
                  }
                >
                  AG2 tracing
                </Link>,
              ],
              [
                "Pydantic AI",
                "mlflow.pydantic_ai.autolog()",
                <Link
                  href={
                    MLFLOW_GENAI_DOCS_URL +
                    "tracing/integrations/listing/pydantic_ai/"
                  }
                >
                  Pydantic AI tracing
                </Link>,
              ],
              [
                "Strands Agents",
                "mlflow.strands.autolog()",
                <Link
                  href={
                    MLFLOW_GENAI_DOCS_URL +
                    "tracing/integrations/listing/strands/"
                  }
                >
                  Strands tracing
                </Link>,
              ],
              [
                "smolagents",
                "mlflow.smolagents.autolog()",
                <Link
                  href={
                    MLFLOW_GENAI_DOCS_URL +
                    "tracing/integrations/listing/smolagents/"
                  }
                >
                  smolagents tracing
                </Link>,
              ],
              [
                "Agno",
                "mlflow.agno.autolog()",
                <Link
                  href={
                    MLFLOW_GENAI_DOCS_URL + "tracing/integrations/listing/agno/"
                  }
                >
                  Agno tracing
                </Link>,
              ],
              [
                "Claude Agent SDK (Python)",
                "mlflow.anthropic.autolog()",
                <Link
                  href={
                    MLFLOW_GENAI_DOCS_URL +
                    "tracing/integrations/listing/claude_agent_sdk_python/"
                  }
                >
                  Claude Agent SDK tracing
                </Link>,
              ],
            ]}
          />

          <p>
            Frameworks without a dedicated integration, and agents written in
            other languages, can send traces through{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "tracing/app-instrumentation/opentelemetry/"
              }
            >
              OpenTelemetry
            </Link>
            . The{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/integrations/"}>
              full integrations list
            </Link>{" "}
            also covers model SDKs such as OpenAI, Anthropic, Gemini, and
            Bedrock, which appear as LLM spans inside an agent trace.
          </p>

          <h2 id="custom-agents">How Do You Trace a Custom Agent?</h2>

          <p>
            If you wrote your own agent loop, add the <code>@mlflow.trace</code>{" "}
            decorator to the functions you care about and set a span type. Calls
            made inside a traced function are nested under it automatically. For
            a block of code that is not its own function, use{" "}
            <code>mlflow.start_span</code>.
          </p>

          <CodeBlock
            code={`import mlflow
from mlflow.entities import SpanType


@mlflow.trace(span_type=SpanType.TOOL)
def search_docs(query: str) -> list[str]:
    return vector_store.search(query)  # your retrieval code


@mlflow.trace(span_type=SpanType.AGENT)
def run_agent(question: str) -> str:
    docs = search_docs(question)
    with mlflow.start_span(name="draft_answer", span_type=SpanType.LLM) as span:
        span.set_inputs({"question": question, "docs": docs})
        answer = call_llm(question, docs)  # your model call
        span.set_outputs({"answer": answer})
    return answer`}
          />

          <p>
            You can mix both styles. Turn on autologging for the model SDK or
            framework you use, and decorate your own tools and orchestration
            code, so the trace shows the full run in one tree. See the{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "tracing/app-instrumentation/manual-tracing/"
              }
            >
              manual tracing guide
            </Link>{" "}
            for more options.
          </p>

          <h2 id="multi-agent-sessions">
            How Do You Trace Multi-Agent Systems and Sessions?
          </h2>

          <p>
            A sub-agent run is recorded as child spans of the agent that called
            it, so one trace shows the whole multi-agent system. In a
            multi-agent system, a supervisor agent calls other agents as tools
            or hands work off to them. Because spans nest, a sub-agent run
            appears as a child of the agent that called it, so you can see who
            did what and where time and tokens went. When agents run in separate
            services, MLflow supports{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "tracing/app-instrumentation/distributed-tracing/"
              }
            >
              distributed tracing
            </Link>{" "}
            to stitch their spans into one trace.
          </p>

          <p>
            Agents are often conversational. Attach a user ID and a session ID
            to each trace and MLflow groups the turns of one conversation, so
            you can follow a multi-turn session from the first message to the
            failure.
          </p>

          <CodeBlock
            code={`import mlflow
from mlflow.entities import SpanType


@mlflow.trace(span_type=SpanType.AGENT)
def chat(message: str, session_id: str, user_id: str) -> str:
    mlflow.update_current_trace(session_id=session_id, user=user_id)
    return agent_reply(message)  # your agent logic`}
          />

          <p>
            The{" "}
            <Link
              href={MLFLOW_GENAI_DOCS_URL + "tracing/track-users-sessions/"}
            >
              user and session tracking guide
            </Link>{" "}
            covers the details. The <code>session_id</code> and{" "}
            <code>user</code> parameters need MLflow 3.11 or later; on older
            versions, set the <code>mlflow.trace.session</code> and{" "}
            <code>mlflow.trace.user</code> metadata keys instead.
          </p>

          <h2 id="debug-and-evaluate">
            How Do You Debug and Evaluate Agents with Traces?
          </h2>

          <p>
            Open a failing trace in the MLflow UI to step through its spans,
            search traces to find similar runs, and run scorers over traces to
            measure quality.
          </p>

          <p>
            Once traces are in MLflow, three workflows build on them. First, you
            debug in the UI: open a bad run, step through the spans, and find
            the tool call with the wrong arguments or the model call that lost
            context. Second, you search: filter traces by status, span type,
            tags, or text to find the runs that matter. Third, you evaluate: run{" "}
            <Link href="/llm-as-a-judge">LLM judges</Link> and code-based
            scorers over the traces to measure quality, tool use, and
            efficiency, and track it over time as part of{" "}
            <Link href="/llm-evaluation">agent evaluation</Link>.
          </p>

          <CodeBlock
            code={`import mlflow
from mlflow.genai.scorers import ToolCallEfficiency

# Find runs that failed
failed = mlflow.search_traces(filter_string="trace.status = 'ERROR'", max_results=50)

# Score the tool-call trajectory of recent agent runs, no re-run needed
traces = mlflow.search_traces(max_results=100)
results = mlflow.genai.evaluate(data=traces, scorers=[ToolCallEfficiency()])`}
          />

          <p>
            Scorers read the whole trace, so they can judge the trajectory (for
            example, redundant tool calls) and not only the final answer. The
            guide on{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "eval-monitor/running-evaluation/traces/"
              }
            >
              evaluating traces
            </Link>{" "}
            shows the complete flow.
          </p>

          <h2 id="production">Does Agent Tracing Work in Production?</h2>

          <p>
            Yes, the same tracing code works in development and in production.
            For production services, install the lightweight{" "}
            <code>mlflow-tracing</code> package, which has a smaller footprint
            than the full MLflow install, and point it at a shared tracking
            server. Traces can be tagged with environment, version, and user
            information so you can slice production runs the way you slice any
            other telemetry. See{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/prod-tracing/"}>
              tracing in production
            </Link>{" "}
            for deployment guidance, and{" "}
            <Link href="/llm-tracing">LLM tracing</Link> for the broader
            picture.
          </p>

          <p>
            Because MLflow is open source, you choose where traces live. You can
            self-host the tracking server, run it on a managed platform, and
            export or re-route data through OpenTelemetry. There are no
            per-trace fees from MLflow itself, and your agent data stays in
            infrastructure you control.
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
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
                MLflow Tracing Documentation
              </Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/integrations/"}>
                Tracing Integrations for Agent Frameworks and LLM SDKs
              </Link>
            </li>
            <li>
              <Link href="/llm-tracing">LLM Tracing</Link>
            </li>
            <li>
              <Link href="/llm-evaluation">
                LLM Evaluation and Agent Evaluation
              </Link>
            </li>
            <li>
              <Link href="/llm-as-a-judge">LLM-as-a-Judge</Link>
            </li>
            <li>
              <Link href="/top-5-agent-observability-tools">
                Top 5 Agent Observability Tools Comparison
              </Link>
            </li>
            <li>
              <Link href="/genai">MLflow for Agents and LLMs Overview</Link>
            </li>
            <li>
              <Link href="/agent-observability">Agent Observability</Link>
            </li>
            <li>
              <Link href="/rag-tracing">RAG Tracing</Link>
            </li>
            <li>
              <Link href="/opentelemetry-llm-tracing">
                OpenTelemetry LLM Tracing
              </Link>
            </li>
            <li>
              <Link href="/trace-analytics">Trace Analytics</Link>
            </li>
            <li>
              <Link href="/llm-observability">LLM Observability</Link>
            </li>
          </ul>
        </div>

        <ArticleSidebar />
        <SocialLinksFooter />
      </div>
    </>
  );
}
