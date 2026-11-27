import { useState } from "react";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../components/Header/Header";
import { SocialLinksFooter } from "../components/SocialLinksFooter/SocialLinksFooter";
import { ArticleSidebar } from "../components/ArticleSidebar/ArticleSidebar";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import TracesListImg from "@site/static/img/llm-observability/mlflow-demo-traces-list.png";
import TraceDetailImg from "@site/static/img/llm-observability/mlflow-demo-trace-detail.png";
import DashboardImg from "@site/static/img/llm-observability/mlflow-demo-dashboard.png";
import { CopyButton } from "../components/CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";

const SEO_TITLE =
  "LLM Observability: Tracing, Evaluation & Monitoring | MLflow";
const SEO_DESCRIPTION =
  "LLM observability records every prompt, tool call, token, and cost in your LLM app so you can debug, evaluate, and monitor it. Open source with MLflow Tracing.";

const SNIPPET_AUTOLOG = `import mlflow
from openai import OpenAI

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-llm-app")

# One line: every OpenAI call is now captured as a trace
mlflow.openai.autolog()

client = OpenAI()
client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is LLM observability?"}],
)`;

const SNIPPET_MANUAL = `import mlflow
from mlflow.entities import SpanType


@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve(query: str) -> list[str]:
    return search_index(query)  # your retrieval code


@mlflow.trace
def answer(question: str) -> str:
    docs = retrieve(question)

    # Add a custom span around any block of code
    with mlflow.start_span(name="build_prompt") as span:
        prompt = f"Context: {docs}\\n\\nQuestion: {question}"
        span.set_inputs({"question": question, "num_docs": len(docs)})

    return call_llm(prompt)  # auto-traced if you enabled autolog`;

const SNIPPET_CONTEXT = `import mlflow


@mlflow.trace
def chat(message: str, session_id: str, user_id: str) -> str:
    # Group traces by conversation and user so you can filter by them later
    mlflow.update_current_trace(
        session_id=session_id,
        user=user_id,
        tags={"app_version": "2.3.1", "environment": "production"},
    )
    return run_agent(message)`;

const SNIPPET_COST = `import mlflow

trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

print(trace.info.token_usage)  # input_tokens, output_tokens, total_tokens
print(trace.info.cost)  # input_cost, output_cost, total_cost in USD`;

const SNIPPET_EVAL = `import mlflow
from mlflow.genai.scorers import Safety, RelevanceToQuery

# Pull recent production traces...
traces = mlflow.search_traces(
    experiment_ids=["<your-experiment-id>"],
    max_results=100,
    return_type="list",
)

# ...and score them with built-in LLM judges
results = mlflow.genai.evaluate(
    data=traces,
    scorers=[Safety(), RelevanceToQuery()],
)`;

const faqs: {
  question: string;
  answer: React.ReactNode;
  answerText?: string;
}[] = [
  {
    question: "What is LLM observability?",
    answer:
      "LLM observability is the practice of recording what an LLM application does on every request (prompts, model responses, retrieval results, tool calls, latency, token usage, and cost) as structured traces, then using that data to debug failures, measure output quality, and monitor production behavior. It matters because LLM applications are non-deterministic: the same input can produce different outputs, so logs and uptime metrics alone cannot explain why an answer was wrong.",
  },
  {
    question:
      "What is the difference between LLM observability and LLM monitoring?",
    answer:
      "Monitoring tells you that something changed: latency rose, errors spiked, or cost jumped. Observability lets you ask why, because every request is recorded as a trace you can open and inspect step by step. In practice you need both: dashboards and alerts for the aggregate view, and traces for root-cause analysis.",
  },
  {
    question:
      "What is the difference between LLM observability and AI observability?",
    answer:
      "LLM observability focuses on applications built on language models: prompts, completions, tokens, latency, and quality. AI observability is the broader term that also covers agents, RAG pipelines, and other AI components. The foundation is the same: traces. MLflow Tracing covers both with one SDK.",
  },
  {
    question: "What should an LLM observability tool capture?",
    answer:
      "At minimum: the full input and output of every LLM call, the intermediate steps (retrieval, tool calls, agent decisions) as nested spans, latency per step, token counts and cost, errors with context, and metadata such as user, session, and application version. Beyond capture, it should let you evaluate quality with LLM judges or human feedback and watch trends over time.",
  },
  {
    question: "How do I add observability to an LLM application with MLflow?",
    answer: (
      <>
        Install MLflow, point it at a tracking server, and call the autolog
        function for your library, for example{" "}
        <code>mlflow.openai.autolog()</code>. Every call is then captured as a
        trace with prompts, responses, latency, and token usage. For your own
        code, add the <code>@mlflow.trace</code> decorator or wrap blocks in{" "}
        <code>mlflow.start_span()</code>. See the{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/quickstart/"}>
          tracing quickstart
        </Link>
        .
      </>
    ),
    answerText:
      "Install MLflow, point it at a tracking server, and call the autolog function for your library, for example mlflow.openai.autolog(). Every call is then captured as a trace with prompts, responses, latency, and token usage. For your own code, add the @mlflow.trace decorator or wrap blocks in mlflow.start_span(). See the tracing quickstart.",
  },
  {
    question: "Does MLflow support OpenTelemetry?",
    answer: (
      <>
        Yes. MLflow Tracing is compatible with OpenTelemetry: you can send
        OpenTelemetry traces to MLflow over OTLP and export MLflow traces to
        other OpenTelemetry-compatible backends. See the{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/opentelemetry/"}>
          OpenTelemetry integration guide
        </Link>
        .
      </>
    ),
    answerText:
      "Yes. MLflow Tracing is compatible with OpenTelemetry: you can send OpenTelemetry traces to MLflow over OTLP and export MLflow traces to other OpenTelemetry-compatible backends. See the OpenTelemetry integration guide.",
  },
  {
    question: "Which LLM providers and frameworks does MLflow trace?",
    answer: (
      <>
        MLflow has one-line automatic tracing for providers such as OpenAI,
        Anthropic, Bedrock, Gemini, and Mistral, and for any OpenAI-compatible
        provider such as Cohere and Ollama (through mlflow.openai.autolog()),
        and for agent frameworks such as LangChain, LangGraph, LlamaIndex,
        CrewAI, DSPy, AutoGen, and the OpenAI Agents SDK. See the full list in
        the{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/integrations/"}>
          integrations documentation
        </Link>
        . SDKs are available for Python and TypeScript.
      </>
    ),
    answerText:
      "MLflow has one-line automatic tracing for providers such as OpenAI, Anthropic, Bedrock, Gemini, and Mistral, and for any OpenAI-compatible provider such as Cohere and Ollama (through mlflow.openai.autolog()), and for agent frameworks such as LangChain, LangGraph, LlamaIndex, CrewAI, DSPy, AutoGen, and the OpenAI Agents SDK. See the full list in the integrations documentation. SDKs are available for Python and TypeScript.",
  },
  {
    question: "How do I track LLM token usage and cost?",
    answer: (
      <>
        MLflow records token counts on every LLM span and aggregates them on the
        trace, available as <code>trace.info.token_usage</code>. Cost in USD is
        available as <code>trace.info.cost</code> when the model has a known
        price. The Overview dashboard in the MLflow UI charts token usage and
        cost over time. See{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/token-usage-cost/"}>
          token usage and cost tracking
        </Link>
        .
      </>
    ),
    answerText:
      "MLflow records token counts on every LLM span and aggregates them on the trace, available as trace.info.token_usage. Cost in USD is available as trace.info.cost when the model has a known price. The Overview dashboard in the MLflow UI charts token usage and cost over time. See token usage and cost tracking.",
  },
  {
    question: "Can I evaluate LLM output quality from my traces?",
    answer: (
      <>
        Yes. Pass traces from <code>mlflow.search_traces()</code> to{" "}
        <code>mlflow.genai.evaluate()</code> with built-in or custom{" "}
        <Link href="/llm-as-a-judge">LLM judges</Link> to score correctness,
        safety, relevance, and more. You can also configure judges to score a
        sample of production traces automatically. See{" "}
        <Link href="/llm-evaluation">LLM evaluation</Link>.
      </>
    ),
    answerText:
      "Yes. Pass traces from mlflow.search_traces() to mlflow.genai.evaluate() with built-in or custom LLM judges to score correctness, safety, relevance, and more. You can also configure judges to score a sample of production traces automatically. See LLM evaluation.",
  },
  {
    question:
      "How do I run LLM observability in production without slowing my app?",
    answer: (
      <>
        MLflow logs traces asynchronously by default in OSS deployments, so
        tracing does not block requests. For high-volume services, set{" "}
        <code>MLFLOW_TRACE_SAMPLING_RATIO</code> to export only a fraction of
        traces, and use the lightweight production tracing SDK to keep
        dependencies small. See{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/prod-tracing/"}>
          production tracing
        </Link>
        .
      </>
    ),
    answerText:
      "MLflow logs traces asynchronously by default in OSS deployments, so tracing does not block requests. For high-volume services, set MLFLOW_TRACE_SAMPLING_RATIO to export only a fraction of traces, and use the lightweight production tracing SDK to keep dependencies small. See production tracing.",
  },
  {
    question: "Is MLflow free for LLM observability?",
    answer:
      "Yes. MLflow is open source under the Apache 2.0 license and hosted by the Linux Foundation. Tracing, evaluation, and the dashboards are included at no cost, and you can self-host it and keep your data in your own infrastructure. Managed MLflow is also offered by cloud providers if you prefer not to run it yourself.",
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
    "Open-source platform for LLM observability, tracing, evaluation, and monitoring of LLM applications and agents.",
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

function Screenshot({
  src,
  alt,
  caption,
}: {
  src: string;
  alt: string;
  caption: string;
}) {
  return (
    <figure style={{ margin: "40px 0" }}>
      <img
        src={src}
        alt={alt}
        loading="lazy"
        style={{
          width: "100%",
          height: "auto",
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
        {caption}
      </figcaption>
    </figure>
  );
}

export default function LLMObservability() {
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
          content="https://mlflow.org/llm-observability"
        />
        <link rel="canonical" href="https://mlflow.org/llm-observability" />
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
          .article-container table,
          .article-container th,
          .article-container td {
            color: #3d3d3d !important;
          }
          .article-container time {
            color: #6b7280 !important;
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
          <h1>LLM Observability</h1>

          <p style={{ fontSize: "14px", color: "#6b7280", marginTop: "-16px" }}>
            Last updated: <time dateTime="2026-11-26">2026-11-26</time>
          </p>

          <p>
            <strong>LLM observability</strong> is the practice of recording what
            your LLM application does on every request (prompts, model
            responses, retrieval results, tool calls, latency, token usage, and
            cost) as structured traces, and using that data to debug failures,
            measure quality, and monitor production behavior. LLM applications
            are non-deterministic, so the same input can produce different
            outputs. Logs and uptime metrics cannot tell you why an answer was
            wrong. A trace can.
          </p>

          <p>
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
              MLflow Tracing
            </Link>{" "}
            is an open source way to get LLM observability. One line of code
            captures every LLM call, and the MLflow UI lets you search traces,
            inspect each step, track token cost, and score quality with{" "}
            <Link href="/llm-as-a-judge">LLM judges</Link>. It is compatible
            with OpenTelemetry, works with any LLM provider or agent framework,
            and stores your data where you choose. This page covers how LLM
            observability works and how to set it up. For a deep dive into the
            tracing layer itself, see{" "}
            <Link href="/llm-tracing">LLM tracing</Link>.
          </p>

          <Screenshot
            src={TracesListImg}
            alt="MLflow Traces page listing LLM requests with input, output, duration, token count, and LLM judge scores for correctness, groundedness, relevance, and safety"
            caption="The MLflow Traces page: every request with its input, output, latency, tokens, and quality scores. Demo data."
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
                <a href="#what-is">What is LLM observability?</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#what-to-capture">What to capture</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#setup">Set up LLM observability</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#debugging">Debug with traces</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#cost-latency">Track cost and latency</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#quality">Monitor quality</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#production">Run it in production</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#choosing">Choosing a tool</a>
              </li>
              <li style={{ marginBottom: "0" }}>
                <a href="#faq">FAQ</a>
              </li>
            </ul>
          </div>

          <h2 id="what-is">What Is LLM Observability?</h2>

          <p>
            Traditional observability answers "is the service up and how fast is
            it?" LLM observability adds the questions that matter for language
            model applications: what exactly did the model see, what did it
            return, which documents were retrieved, which tools were called, how
            many tokens did it cost, and was the answer any good?
          </p>

          <p>
            It is built on <strong>traces</strong>. A trace is the record of one
            request through your application. It is made of nested{" "}
            <strong>spans</strong>, one per step: an LLM call, a retrieval, a
            tool call, or a function of your own. Each span stores its inputs,
            outputs, timing, and attributes such as model name and token counts.
            Reading a trace top to bottom shows what the application did and
            where it went wrong.
          </p>

          <div style={{ overflowX: "auto", margin: "32px 0" }}>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: "15px",
              }}
            >
              <thead>
                <tr style={{ background: "#f9fafb", textAlign: "left" }}>
                  <th style={{ padding: "12px", border: "1px solid #e5e7eb" }}>
                    Practice
                  </th>
                  <th style={{ padding: "12px", border: "1px solid #e5e7eb" }}>
                    Question it answers
                  </th>
                  <th style={{ padding: "12px", border: "1px solid #e5e7eb" }}>
                    Main data
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ padding: "12px", border: "1px solid #e5e7eb" }}>
                    <strong>LLM observability</strong>
                  </td>
                  <td style={{ padding: "12px", border: "1px solid #e5e7eb" }}>
                    Why did the application behave this way?
                  </td>
                  <td style={{ padding: "12px", border: "1px solid #e5e7eb" }}>
                    Traces with spans, inputs, outputs, tokens
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: "12px", border: "1px solid #e5e7eb" }}>
                    <Link href="/ai-monitoring">LLM monitoring</Link>
                  </td>
                  <td style={{ padding: "12px", border: "1px solid #e5e7eb" }}>
                    Did something change in production?
                  </td>
                  <td style={{ padding: "12px", border: "1px solid #e5e7eb" }}>
                    Aggregated latency, errors, cost, quality scores
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: "12px", border: "1px solid #e5e7eb" }}>
                    <Link href="/llm-evaluation">LLM evaluation</Link>
                  </td>
                  <td style={{ padding: "12px", border: "1px solid #e5e7eb" }}>
                    Is the output good enough?
                  </td>
                  <td style={{ padding: "12px", border: "1px solid #e5e7eb" }}>
                    Scores from LLM judges, code checks, and human feedback
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <p>
            The three work as a loop. Traces give evaluation realistic data to
            score, evaluation turns traces into quality signals, and monitoring
            watches those signals over time and points you back to the traces
            that need attention.
          </p>

          <h2 id="what-to-capture">
            What Should You Capture in an LLM Application?
          </h2>

          <p>
            A useful trace records more than the final answer. These are the
            signals that explain most production problems:
          </p>

          <div className="grid-2">
            <div className="card">
              <h3>Prompts and responses</h3>
              <p>
                The full input and output of every LLM call, including system
                prompts, parameters such as temperature, and the model name.
              </p>
            </div>

            <div className="card">
              <h3>Intermediate steps</h3>
              <p>
                Retrieved documents, tool calls with their arguments and
                results, and agent decisions, recorded as nested spans.
              </p>
            </div>

            <div className="card">
              <h3>Latency, tokens, and cost</h3>
              <p>
                Time spent in each step, input and output token counts, and cost
                per request, so you can find slow or expensive steps.
              </p>
            </div>

            <div className="card">
              <h3>Errors and context</h3>
              <p>
                Exceptions with the span that raised them, plus user, session,
                and application version so you can filter and compare.
              </p>
            </div>
          </div>

          <h2 id="setup">How Do You Set Up LLM Observability with MLflow?</h2>

          <p>
            Start an MLflow tracking server (<code>mlflow server</code>), then
            enable automatic tracing for your LLM library. MLflow supports
            one-line tracing for{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/integrations/"}>
              many providers and frameworks
            </Link>
            , including OpenAI, Anthropic, LangChain, LangGraph, LlamaIndex,
            CrewAI, and DSPy.
          </p>

          <CodeBlock code={SNIPPET_AUTOLOG} />

          <p>
            Automatic tracing covers library calls. For your own logic, such as
            retrieval, business rules, or custom tools, add the{" "}
            <code>@mlflow.trace</code> decorator or wrap a block in{" "}
            <code>mlflow.start_span()</code>. The two approaches combine into a
            single trace.
          </p>

          <CodeBlock code={SNIPPET_MANUAL} />

          <p>
            To make traces searchable by who and what produced them, attach
            session, user, and tags to the current trace. In the UI you can then
            filter to one customer, one conversation, or one app version.
          </p>

          <CodeBlock code={SNIPPET_CONTEXT} />

          <p>
            See{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/quickstart/"}>
              the tracing quickstart
            </Link>{" "}
            for a complete walkthrough, or{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "tracing/app-instrumentation/manual-tracing"
              }
            >
              manual tracing
            </Link>{" "}
            for the full decorator and span API.
          </p>

          <h2 id="debugging">How Do You Debug LLM Applications with Traces?</h2>

          <p>
            Open any trace in the MLflow UI to see the span tree on the left and
            the inputs, outputs, and attributes of the selected span on the
            right. Typical questions this answers in minutes:
          </p>

          <ul>
            <li>
              <strong>Why was the answer wrong?</strong> Check whether the
              retrieval span returned relevant documents or the model ignored
              good context.
            </li>
            <li>
              <strong>Why is this request slow?</strong> Compare the duration of
              each span to find the slow LLM call, tool, or retrieval.
            </li>
            <li>
              <strong>Why did the agent loop or pick the wrong tool?</strong>{" "}
              Read the sequence of tool calls and the arguments passed to each.
            </li>
            <li>
              <strong>What changed since last week?</strong> Filter traces by
              app version tag and compare them.
            </li>
          </ul>

          <Screenshot
            src={TraceDetailImg}
            alt="MLflow trace detail view showing an agent trace with nested spans for LLM calls and tool calls, with duration, token count, and cost per span"
            caption="A trace in the MLflow UI: nested spans for each LLM and tool call, with duration, tokens, and cost per step. Demo data."
          />

          <h2 id="cost-latency">
            How Do You Track LLM Token Usage, Cost, and Latency?
          </h2>

          <p>
            MLflow records token counts on LLM spans and rolls them up to the
            trace. Cost in USD is calculated from token usage and model pricing
            when the model has a known price. Both are available in code and
            charted in the experiment Overview dashboard, along with request
            volume, latency percentiles, and error rates.
          </p>

          <CodeBlock code={SNIPPET_COST} />

          <Screenshot
            src={DashboardImg}
            alt="MLflow experiment Overview dashboard with charts for trace volume, latency percentiles, errors, token usage, and tokens per trace"
            caption="The MLflow Overview dashboard: trace volume, latency, errors, and token usage over time. Demo data."
          />

          <p>
            Token usage and cost depend on what the provider returns, so some
            models report no usage. See{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/token-usage-cost/"}>
              token usage and cost tracking
            </Link>{" "}
            and the{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL + "tracing/observe-with-traces/dashboard/"
              }
            >
              dashboard guide
            </Link>{" "}
            for details. To enforce rate limits and budgets rather than only
            observe spend, see the <Link href="/ai-gateway">AI Gateway</Link>.
          </p>

          <h2 id="quality">How Do You Monitor LLM Output Quality?</h2>

          <p>
            Latency and cost are easy to measure. Quality is the hard part, and
            observability is what makes it measurable: once requests are traces,
            you can run <Link href="/llm-as-a-judge">LLM judges</Link> over
            them. Pass traces to <code>mlflow.genai.evaluate()</code> with
            built-in scorers such as <code>Safety</code> and{" "}
            <code>RelevanceToQuery</code>, or write custom judges for your own
            criteria.
          </p>

          <CodeBlock code={SNIPPET_EVAL} />

          <p>
            You can also configure judges to score a sample of production traces
            automatically, so quality shows up next to latency and cost. Add
            human feedback to the same traces to calibrate the judges. See{" "}
            <Link href="/llm-evaluation">LLM evaluation</Link> and{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL + "eval-monitor/automatic-evaluations/"
              }
            >
              automatic evaluations
            </Link>
            .
          </p>

          <h2 id="production">
            How Do You Run LLM Observability in Production?
          </h2>

          <p>
            Tracing should not slow down or destabilize your application. For
            production deployments:
          </p>

          <ul>
            <li>
              <strong>Async logging.</strong> MLflow exports traces
              asynchronously by default in OSS deployments, so requests are not
              blocked on the tracking server.
            </li>
            <li>
              <strong>Sampling.</strong> Set{" "}
              <code>MLFLOW_TRACE_SAMPLING_RATIO</code> to a value below 1.0 to
              export only a fraction of traces for high-volume services.
            </li>
            <li>
              <strong>Lightweight SDK.</strong> The production tracing SDK keeps
              the dependency footprint small.
            </li>
            <li>
              <strong>Privacy.</strong> Mask sensitive fields before traces are
              stored. See{" "}
              <Link
                href={
                  MLFLOW_GENAI_DOCS_URL + "tracing/observe-with-traces/masking/"
                }
              >
                masking
              </Link>
              .
            </li>
            <li>
              <strong>Open standards.</strong> Because MLflow Tracing is
              OpenTelemetry-compatible, traces can flow to and from the rest of
              your observability stack.
            </li>
          </ul>

          <p>
            The full checklist is in the{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/prod-tracing/"}>
              production tracing guide
            </Link>
            .
          </p>

          <h2 id="choosing">
            What Are the Best LLM Observability Tools and Platforms?
          </h2>

          <p>
            The best LLM observability platform for your team is one that traces
            your stack with little code, speaks open standards, lets you own
            your data, and connects traces to quality. MLflow is an open source
            LLM observability platform that does all four. When comparing any
            tools, these questions matter most:
          </p>

          <ul>
            <li>
              <strong>Coverage.</strong> Does it trace your LLM providers and
              agent frameworks with minimal code?
            </li>
            <li>
              <strong>Standards.</strong> Does it speak OpenTelemetry, so you
              are not locked into one vendor's SDK and format?
            </li>
            <li>
              <strong>Data ownership.</strong> Can you self-host and keep
              prompts and responses in your own infrastructure?
            </li>
            <li>
              <strong>Quality, not just logs.</strong> Does it connect traces to
              evaluation, human feedback, and monitoring?
            </li>
            <li>
              <strong>Cost.</strong> What does it cost at your trace volume?
            </li>
          </ul>

          <p>
            MLflow is Apache 2.0 licensed and hosted by the Linux Foundation, so
            you can run it yourself at no license cost. For side-by-side
            comparisons, see{" "}
            <Link href="/top-5-agent-observability-tools">
              the top agent observability tools
            </Link>
            , or the <Link href="/langfuse-alternative">Langfuse</Link>,{" "}
            <Link href="/langsmith-alternative">LangSmith</Link>, and{" "}
            <Link href="/arize-phoenix-alternative">Arize Phoenix</Link>{" "}
            comparisons.
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
              <Link href="/llm-tracing">LLM Tracing</Link>
            </li>
            <li>
              <Link href="/llm-evaluation">LLM Evaluation</Link>
            </li>
            <li>
              <Link href="/llm-as-a-judge">LLM as a Judge</Link>
            </li>
            <li>
              <Link href="/ai-monitoring">AI Monitoring</Link>
            </li>
            <li>
              <Link href="/llmops">LLMOps</Link>
            </li>
            <li>
              <Link href="/ai-gateway">AI Gateway</Link>
            </li>
            <li>
              <Link href="/top-5-agent-observability-tools">
                Top 5 Agent Observability Tools
              </Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
                MLflow Tracing Documentation
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
