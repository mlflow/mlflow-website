import { useState } from "react";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../../components/Header/Header";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import TracingHero from "@site/static/img/GenAI_observability/GenAI_observability_hero.png";
import { CopyButton } from "../../components/CodeSnippet/CopyButton";
import {
  customNightOwl,
  CODE_BG,
} from "../../components/CodeSnippet/codeTheme";

const SEO_TITLE = "LLM Tracing & AI Tracing for Agents | MLflow Agent Platform";
const SEO_DESCRIPTION =
  "Learn LLM tracing, AI tracing, and agent tracing with MLflow—the comprehensive, open-source agent engineering and ops platform.";

const faqs: {
  question: string;
  answer: React.ReactNode;
  answerText?: string;
}[] = [
  {
    question: "What is LLM tracing?",
    answer:
      "LLM tracing is the practice of capturing detailed execution data for every large language model call in your application, including prompts, completions, model parameters, token counts, latency, and metadata. It creates a complete audit trail showing exactly what happened during each LLM request.",
  },
  {
    question: "How is LLM tracing different from traditional logging?",
    answer:
      "Traditional logging captures errors and discrete events. LLM tracing captures the full context of AI execution: prompts, responses, tool calls, retrieval results, model parameters, token usage, and nested spans showing multi-step reasoning. It's structured, queryable, and designed specifically for debugging non-deterministic AI systems.",
  },
  {
    question: "What is AI tracing?",
    answer:
      "AI tracing is the broader practice of capturing execution data across all AI system components—LLMs, embeddings, retrievers, agents, RAG pipelines, and more. It extends LLM tracing to cover the entire AI application stack, not just model calls.",
  },
  {
    question: "What is agent tracing?",
    answer:
      "Agent tracing captures the multi-step execution graph of autonomous agents, showing how they reason, which tools they call, how they handle errors, and how they chain multiple LLM calls together. It extends LLM tracing to reveal the complete decision-making process of agentic systems built with frameworks like LangGraph, CrewAI, or AutoGen.",
  },
  {
    question: "Do I need LLM tracing for my application?",
    answer:
      "Yes, if you're building production LLM applications or agents. LLM tracing is essential for debugging unexpected outputs, optimizing token costs, monitoring quality over time, and understanding why your AI system behaves the way it does. Without tracing, you're blind to what's actually happening inside your AI application.",
  },
  {
    question: "What's the difference between tracing and observability?",
    answer:
      "Tracing is one component of observability. Tracing captures execution data (what happened, when, and how). Observability combines tracing with evaluation (quality assessment), monitoring (metrics over time), and feedback collection to give you complete visibility into your AI system's behavior and quality.",
  },
  {
    question: "What is the best LLM tracing tool?",
    answer:
      "The best LLM tracing tool depends on your needs. MLflow is the leading open-source option, offering automatic tracing for 50+ LLM providers and agent frameworks with no vendor lock-in. MLflow is fully OpenTelemetry compatible, giving you total ownership of your trace data. Unlike proprietary tools, MLflow is Apache 2.0 licensed and backed by a community of 20,000+ GitHub stars.",
  },
  {
    question: "What LLM providers and frameworks does MLflow support?",
    answer:
      "MLflow supports any LLM provider and framework. This includes OpenAI, Anthropic (Claude), AWS Bedrock, Google Gemini, Azure OpenAI, Mistral, Cohere, AI21, Together AI, Anyscale, vLLM, Ollama, and more. For frameworks: LangChain, LangGraph, LlamaIndex, CrewAI, AutoGen, DSPy, Haystack, Semantic Kernel, Vercel AI SDK, and many others. MLflow is OpenTelemetry compatible, so it works with any language or tool.",
  },
  {
    question: "How does MLflow tracing compare to other tools?",
    answer:
      "Unlike proprietary tracing tools that lock you into a vendor's ecosystem, MLflow provides complete, open-source tracing with no vendor lock-in. It supports any LLM or agent framework, is OpenTelemetry compatible, and gives you full control over your trace data. MLflow is also available on Databricks, AWS, and other platforms.",
  },
  {
    question: "Is MLflow free for LLM tracing?",
    answer:
      "Yes. MLflow is 100% open source under the Apache 2.0 license, backed by the Linux Foundation. You can use all of its tracing features for free, including in commercial applications. There are no per-trace fees, no usage limits, and no vendor lock-in.",
  },
  {
    question: "How do I get started with LLM tracing?",
    answer: (
      <>
        Getting started with MLflow LLM tracing takes just one line of code.
        Install MLflow, call{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/quickstart/"}>
          mlflow.openai.autolog()
        </Link>{" "}
        (or the equivalent for your framework), and every LLM call is
        automatically traced. See the{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
          MLflow tracing documentation
        </Link>{" "}
        for framework-specific examples.
      </>
    ),
    answerText:
      "Getting started with MLflow LLM tracing takes just one line of code. Install MLflow, call mlflow.openai.autolog() (or the equivalent for your framework), and every LLM call is automatically traced. See the MLflow tracing documentation for framework-specific examples.",
  },
  {
    question: "Does MLflow support OpenTelemetry?",
    answer:
      "Yes. MLflow's tracing is fully compatible with OpenTelemetry, so you can export traces to any OpenTelemetry-compatible backend. This gives you total ownership and portability of your trace data without vendor lock-in.",
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
    "Open-source platform for LLM tracing, AI observability, experiment tracking, evaluation, and deployment.",
  url: "https://mlflow.org",
  license: "https://www.apache.org/licenses/LICENSE-2.0",
};

export default function LLMTracing() {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://mlflow.org/faq/llm-tracing" />
        <link rel="canonical" href="https://mlflow.org/faq/llm-tracing" />
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
          <h1>LLM Tracing and AI Tracing</h1>

          <p>
            LLM tracing is the practice of capturing detailed execution data for
            every large language model call in your application, including
            prompts, completions, model parameters, token counts, latency, and
            metadata. When tracing extends beyond individual LLM calls to cover
            entire AI system stacks (embeddings, retrievers, RAG pipelines),
            this is known as <a href="#ai-tracing">AI tracing</a>. For
            multi-step autonomous{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "agent-evaluation/"}>
              agents
            </Link>
            , tracing the complete decision-making process is known as{" "}
            <a href="#agent-tracing">agent tracing</a>.
          </p>

          <p>
            LLM tracing is a foundational component of{" "}
            <Link href="/faq/ai-observability">AI observability</Link>. While
            observability encompasses tracing, evaluation, monitoring, and
            feedback collection, tracing provides the raw execution data that
            makes all other observability capabilities possible. LLM tracing
            gives engineering teams deep visibility into what their AI
            applications are actually doing: not just whether they're running,
            but what prompts are being sent, what responses are being generated,
            how much they're costing, and where failures occur. As LLM
            applications move from prototypes to production-critical systems,
            tracing becomes essential for debugging, optimization, and quality
            assurance.
          </p>

          <p>
            Unlike traditional logging, LLM tracing captures the full context of
            AI execution in a structured, queryable format. It records{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "prompts/"}>prompts</Link>,
            model responses,{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>tool calls</Link>,
            retrieval results, token usage, and nested spans showing multi-step
            reasoning. This structured telemetry allows teams to search for
            patterns (e.g., "all traces where token usage exceeded 10,000"),
            debug specific failures, and understand the "why" behind every
            output.
          </p>

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
                src={
                  require("@site/static/img/releases/3.10.0/demo-experiment.mp4")
                    .default
                }
                type="video/mp4"
              />
              Your browser does not support the video tag.
            </video>
          </div>

          <div style={{ margin: "40px 0", textAlign: "center" }}>
            <img
              src={TracingHero}
              alt="MLflow Trace UI showing captured LLM calls with prompts, responses, and metadata"
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
              An LLM trace showing prompts, completions, token usage, latency,
              and execution metadata
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
                <a href="#llm-tracing">LLM Tracing</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#ai-tracing">AI Tracing</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#agent-tracing">Agent Tracing</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#how-to-implement">How to Implement</a>
              </li>
              <li style={{ marginBottom: "0" }}>
                <a href="#faq">FAQ</a>
              </li>
            </ul>
          </div>

          <h2>Why LLM Tracing Matters</h2>

          <p>
            LLM applications introduce unique challenges that traditional
            logging can't address:
          </p>

          <div className="grid-2">
            <div className="card">
              <h3>Debugging Non-Determinism</h3>
              <p>
                <strong>Problem:</strong> LLMs produce different outputs for the
                same input. Traditional logs can't capture the full context
                needed to debug why a specific output was generated.
              </p>
              <p>
                <strong>Solution:</strong> Tracing captures prompts, model
                parameters, and responses together, making every execution
                reproducible and debuggable.
              </p>
            </div>

            <div className="card">
              <h3>Cost Optimization</h3>
              <p>
                <strong>Problem:</strong> Token costs can spiral without
                visibility into which requests are most expensive and why.
              </p>
              <p>
                <strong>Solution:</strong>{" "}
                <Link
                  href={MLFLOW_GENAI_DOCS_URL + "tracing/token-usage-cost/"}
                >
                  Track token usage
                </Link>{" "}
                per request, identify inefficient prompts, and find
                opportunities to switch to smaller models without sacrificing
                quality.
              </p>
            </div>

            <div className="card">
              <h3>Quality Assurance</h3>
              <p>
                <strong>Problem:</strong> LLMs can produce hallucinations,
                irrelevant responses, or degraded outputs that undermine user
                trust.
              </p>
              <p>
                <strong>Solution:</strong> Trace data combined with{" "}
                <Link href={MLFLOW_GENAI_DOCS_URL + "llm-evaluate/"}>
                  automated evaluation
                </Link>{" "}
                helps detect quality issues before they reach users.
              </p>
            </div>

            <div className="card">
              <h3>Production Monitoring</h3>
              <p>
                <strong>Problem:</strong> Without tracing, it's impossible to
                know when LLM behavior changes due to model updates or prompt
                drift.
              </p>
              <p>
                <strong>Solution:</strong> Continuous tracing provides a
                baseline for{" "}
                <Link href="/faq/ai-observability">detecting regressions</Link>,
                latency spikes, and cost anomalies.
              </p>
            </div>
          </div>

          <h2 id="llm-tracing">What is LLM Tracing?</h2>

          <p>
            LLM tracing captures detailed execution data for every large
            language model call in your application. Each trace records:
          </p>

          <ul>
            <li>
              <strong>Prompts:</strong> The exact input sent to the model,
              including system messages, user messages, and few-shot examples.
            </li>
            <li>
              <strong>Completions:</strong> The full response generated by the
              model, including all candidate outputs if using n &gt; 1.
            </li>
            <li>
              <strong>Model Parameters:</strong> Temperature, top_p, max_tokens,
              stop sequences, and other configuration that influences output.
            </li>
            <li>
              <strong>Token Usage:</strong> Prompt tokens, completion tokens,
              and total tokens consumed, enabling{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/token-usage-cost/"}>
                cost tracking
              </Link>
              .
            </li>
            <li>
              <strong>Latency:</strong> Time to first token, total response
              time, and server-side processing time.
            </li>
            <li>
              <strong>Metadata:</strong> User IDs, session IDs, request IDs, and
              custom tags for filtering and analysis.
            </li>
          </ul>

          <p>
            This structured data allows teams to search for patterns (e.g., "all
            traces where latency exceeded 5 seconds"), debug specific failures,
            and understand cost drivers. Unlike logs, traces are designed to be
            queried, aggregated, and correlated across millions of requests.
          </p>

          <p>
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
              MLflow's automatic tracing
            </Link>{" "}
            captures all of this telemetry with a single line of code for 50+
            LLM providers and frameworks, storing traces locally or sending them
            to your tracking server for analysis and monitoring.
          </p>

          <h2 id="ai-tracing">What is AI Tracing?</h2>

          <p>
            AI tracing extends LLM tracing to cover the entire AI application
            stack, not just individual model calls. While LLM tracing focuses on
            capturing prompts and completions, AI tracing captures every
            component of your AI system:
          </p>

          <ul>
            <li>
              <strong>Embeddings:</strong> Text chunks embedded, embedding
              models used, vector dimensions, and computation time.
            </li>
            <li>
              <strong>Retrievers:</strong> Search queries, retrieved documents,
              similarity scores, and retrieval latency.
            </li>
            <li>
              <strong>RAG Pipelines:</strong> Document chunking, retrieval,
              re-ranking, and context assembly steps.
            </li>
            <li>
              <strong>Multi-Model Systems:</strong> Chains of different models
              (e.g., embedding model → ranking model → generation model).
            </li>
            <li>
              <strong>Custom Logic:</strong> Business logic, data
              transformations, and external API calls integrated with LLMs.
            </li>
          </ul>

          <p>
            AI tracing captures these components as a single execution graph,
            showing how data flows through your entire AI stack. This makes it
            possible to debug failures that span multiple components (e.g., "the
            retriever returned irrelevant documents, so the LLM hallucinated")
            and optimize end-to-end latency and cost.
          </p>

          <p>
            With{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
              MLflow's OpenTelemetry-compatible tracing
            </Link>
            , you can trace any component of your AI stack, not just LLM calls.
            Use{" "}
            <Link
              href={MLFLOW_GENAI_DOCS_URL + "tracing/tracing-with-decorators/"}
            >
              <code>@mlflow.trace</code> decorators
            </Link>{" "}
            to instrument custom functions, or rely on automatic tracing for
            popular frameworks like{" "}
            <Link href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langchain.html">
              LangChain
            </Link>
            ,{" "}
            <Link href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/llamaindex.html">
              LlamaIndex
            </Link>
            , and{" "}
            <Link href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langgraph.html">
              LangGraph
            </Link>
            .
          </p>

          <h2 id="agent-tracing">What is Agent Tracing?</h2>

          <p>
            Agent tracing extends AI tracing to multi-step autonomous agents.
            While LLM tracing tracks individual model calls and AI tracing
            captures multi-component pipelines, agent tracing reveals the
            complete decision-making process of agents that reason, plan, and
            act across multiple turns.
          </p>

          <p>
            Agents built with frameworks like LangGraph, CrewAI, or AutoGen make
            dynamic decisions: which tools to call, when to retry, how to
            recover from errors, and when to ask for help. Agent tracing
            captures this execution graph:
          </p>

          <ul>
            <li>
              <strong>Reasoning Steps:</strong> Internal thoughts, planning, and
              reflection that guide the agent's actions.
            </li>
            <li>
              <strong>Tool Calls:</strong> Which tools were invoked, with what
              arguments, and what they returned.
            </li>
            <li>
              <strong>Conditional Branches:</strong> How the agent chose between
              different paths based on intermediate results.
            </li>
            <li>
              <strong>Iterative Loops:</strong> Retry logic, error handling, and
              multi-turn interactions with tools or users.
            </li>
            <li>
              <strong>Parallel Execution:</strong> Simultaneous tool calls and
              how results are merged.
            </li>
          </ul>

          <p>
            This visibility is critical for debugging agent failures. When an
            agent gets stuck in a loop, makes incorrect tool choices, or
            produces unexpected outputs, agent tracing shows exactly where the
            reasoning went wrong.
          </p>

          <p>
            <Link href={MLFLOW_GENAI_DOCS_URL + "agent-evaluation/"}>
              MLflow automatically traces agent workflows
            </Link>
            , capturing the full directed acyclic graph (DAG) of execution. You
            can see every reasoning step, tool call, and decision point, making
            it easy to identify and fix problematic agent behaviors.
          </p>

          <h2>Common Use Cases for LLM Tracing</h2>

          <p>LLM tracing solves real-world problems across the AI lifecycle:</p>

          <ul>
            <li>
              <strong>Debugging Hallucinations:</strong> When your LLM produces
              incorrect outputs, tracing shows exactly what prompt was sent,
              what context was included, and what parameters were used. This
              makes it easy to identify whether the problem is in prompt
              construction, retrieval quality, or model behavior.
            </li>
            <li>
              <strong>Optimizing Token Costs:</strong>{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/token-usage-cost/"}>
                Track token usage and costs
              </Link>{" "}
              per request to identify expensive queries, inefficient prompts, or
              opportunities to switch to smaller models. Teams use tracing to
              reduce LLM costs by 30-50% without sacrificing quality.
            </li>
            <li>
              <strong>Monitoring Production Quality:</strong> Continuous tracing
              combined with{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "llm-evaluate/"}>
                automated evaluation
              </Link>{" "}
              helps detect when model behavior degrades from API updates, prompt
              drift, or data changes—before users notice.
            </li>
            <li>
              <strong>A/B Testing Prompts:</strong> Before deploying prompt
              changes to production, use traced data to run{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "llm-evaluate/"}>
                side-by-side evaluations
              </Link>
              . Compare quality metrics like relevance, factuality, and safety
              to ensure changes improve output quality.
            </li>
            <li>
              <strong>Understanding Agent Behavior:</strong>{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "agent-evaluation/"}>
                Agents
              </Link>{" "}
              can behave unpredictably—getting stuck in loops, making incorrect
              tool choices, or producing inconsistent outputs. Agent tracing
              shows every reasoning step, tool call, and decision point so you
              can identify and fix problematic patterns.
            </li>
            <li>
              <strong>Compliance & Auditing:</strong> Capture complete audit
              trails showing what prompts were sent, what responses were
              received, and what data was accessed.{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "guides/responsible-ai/"}>
                Enforce PII redaction policies and content guardrails
              </Link>{" "}
              to meet regulatory requirements.
            </li>
          </ul>

          <h2 id="how-to-implement">How to Implement LLM Tracing</h2>

          <p>
            Modern open-source AI platforms like{" "}
            <Link href="/genai">MLflow</Link> make it easy to add
            production-grade LLM tracing with minimal code changes.
          </p>

          <p>
            With just a single line of code, you can automatically capture
            traces for every LLM call, including prompts, responses, token
            usage, latency, and model parameters. These traces are stored
            locally or sent to your MLflow tracking server, where you can
            search, filter, and analyze them in the MLflow UI.
          </p>

          <p>
            Here are quick examples of enabling automatic tracing. Check out the{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/integrations/"}>
              MLflow tracing integrations documentation
            </Link>{" "}
            to see how to use tracing with LangChain, LangGraph, LlamaIndex,
            Vercel AI SDK, and other frameworks.
          </p>

          <p style={{ marginTop: "32px", marginBottom: "0px" }}>
            <strong>OpenAI</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton
                code={`import mlflow

# Enable automatic tracing for OpenAI
mlflow.openai.autolog()

# That's it - every LLM call is now traced
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[{"role": "user", "content": "Hello!"}],
)`}
              />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={`import mlflow

# Enable automatic tracing for OpenAI
mlflow.openai.autolog()

# That's it - every LLM call is now traced
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[{"role": "user", "content": "Hello!"}],
)`}
                language="python"
              >
                {({ style, tokens, getLineProps, getTokenProps }) => (
                  <pre
                    className="text-xs font-mono !m-0 !p-0 text-left"
                    style={{
                      ...style,
                      backgroundColor: "transparent",
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

          <p style={{ marginTop: "32px", marginBottom: "0px" }}>
            <strong>LangGraph</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton
                code={`import mlflow

# Enable automatic tracing for LangChain
mlflow.langchain.autolog()

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

llm = ChatOpenAI(model="gpt-5.2")
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
agent.run("What is the weather in San Francisco?")`}
              />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={`import mlflow

# Enable automatic tracing for LangChain
mlflow.langchain.autolog()

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

llm = ChatOpenAI(model="gpt-5.2")
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
agent.run("What is the weather in San Francisco?")`}
                language="python"
              >
                {({ style, tokens, getLineProps, getTokenProps }) => (
                  <pre
                    className="text-xs font-mono !m-0 !p-0 text-left"
                    style={{
                      ...style,
                      backgroundColor: "transparent",
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

          <p style={{ marginTop: "32px", marginBottom: "0px" }}>
            <strong>Vercel AI SDK</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">
                typescript
              </span>
              <CopyButton
                code={`import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

// Configure OpenTelemetry to send traces to MLflow
// (see MLflow docs for setup details)

// Enable tracing for each AI SDK call
const result = await generateText({
  model: openai('gpt-5.2'),
  prompt: 'What is MLflow?',
  experimental_telemetry: { isEnabled: true }
});`}
              />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={`import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

// Configure OpenTelemetry to send traces to MLflow
// (see MLflow docs for setup details)

// Enable tracing for each AI SDK call
const result = await generateText({
  model: openai('gpt-5.2'),
  prompt: 'What is MLflow?',
  experimental_telemetry: { isEnabled: true }
});`}
                language="typescript"
              >
                {({ style, tokens, getLineProps, getTokenProps }) => (
                  <pre
                    className="text-xs font-mono !m-0 !p-0 text-left"
                    style={{
                      ...style,
                      backgroundColor: "transparent",
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

          <div style={{ margin: "40px 0", textAlign: "center" }}>
            <img
              src={TracingHero}
              alt="MLflow Trace UI showing captured LLM calls with prompts, responses, and metadata"
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
              The MLflow UI automatically captures and displays traces for every
              LLM call
            </p>
          </div>

          <div className="info-box">
            <p>
              <Link href="/genai" style={{ color: "#007bff" }}>
                <strong>MLflow</strong>
              </Link>{" "}
              is the largest open-source AI platform, backed by the Linux
              Foundation and licensed under Apache 2.0. With 20,000+ GitHub
              stars and 900+ contributors, it provides complete LLM tracing with
              no vendor lock-in.{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL}>Get started →</Link>
            </p>
          </div>

          <h2>Open Source vs. Proprietary LLM Tracing</h2>

          <p>
            When choosing an LLM tracing platform, the decision between open
            source and proprietary SaaS tools has significant long-term
            implications for your team, infrastructure, and data ownership.
          </p>

          <p>
            <strong>
              Open Source (<Link href="/genai">MLflow</Link>):
            </strong>{" "}
            With MLflow, you maintain complete control over your tracing
            infrastructure and data. Deploy on your own infrastructure or use
            managed versions on Databricks, AWS, or other platforms. There are
            no per-trace fees, no usage limits, and no vendor lock-in. Your
            trace data stays under your control, and you can customize the
            platform to your exact needs. MLflow integrates with any LLM
            provider and agent framework through OpenTelemetry-compatible
            tracing.
          </p>

          <p>
            <strong>Proprietary SaaS Tools:</strong> Commercial tracing
            platforms offer convenience but at the cost of flexibility and
            control. They typically charge per trace volume or per seat, which
            can become expensive at scale. Your data is sent to their servers,
            raising privacy and compliance concerns. You're locked into their
            ecosystem, making it difficult to switch providers or customize
            functionality. Most proprietary tools only support a subset of LLM
            providers and frameworks.
          </p>

          <p>
            <strong>Why Teams Choose Open Source:</strong> Organizations
            building production LLM applications increasingly choose MLflow
            because it offers enterprise-grade tracing without compromising on
            data sovereignty, cost predictability, or flexibility. The Apache
            2.0 license and Linux Foundation backing ensure MLflow remains truly
            open and community-driven, not controlled by a single vendor.
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
              <Link href="/faq/ai-observability">AI Observability Guide</Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                Agent Evaluation Guide
              </Link>
            </li>
            <li>
              <Link href="/llmops">LLMOps Guide</Link>
            </li>
            <li>
              <Link href="/aiops">AIOps Guide</Link>
            </li>
            <li>
              <Link href="/genai">MLflow for Agents and LLMs Overview</Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL}>
                MLflow for Agents and LLMs Documentation
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </>
  );
}
