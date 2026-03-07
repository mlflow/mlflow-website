import React, { useState } from "react";
import Layout from "@theme/Layout";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";
import { CopyButton } from "../components/CodeSnippet/CopyButton";

const SEO_TITLE = "AI Platform for LLM Apps & Agents | MLflow Agent Platform";
const SEO_DESCRIPTION =
  "Build, evaluate, and deploy LLM apps and AI agents with MLflow—the comprehensive, open-source agent engineering and ops platform with tracing, evaluation, and monitoring.";

const faqs: {
  question: string;
  answer: React.ReactNode;
  answerText?: string;
}[] = [
  {
    question: "What is an AI platform?",
    answer:
      "An AI platform is an integrated environment that provides the tools, infrastructure, and workflows needed to build, test, deploy, and monitor AI applications. Modern AI platforms focus on the full lifecycle of LLM-powered apps and agents, from development through production, covering tracing, evaluation, prompt management, and observability.",
  },
  {
    question: "What is an agent platform?",
    answer: (
      <>
        An agent platform is a specialized AI platform designed for building and
        operating autonomous AI agents. Agents use LLMs to reason, plan, and
        execute multi-step tasks by calling tools and APIs. An agent platform
        provides the infrastructure to{" "}
        <Link
          href="https://mlflow.org/docs/latest/genai/tracing/"
          style={{ color: "#0194e2", fontWeight: "600" }}
        >
          trace agent execution
        </Link>
        ,{" "}
        <Link
          href="https://mlflow.org/docs/latest/genai/eval-monitor/"
          style={{ color: "#0194e2", fontWeight: "600" }}
        >
          evaluate agent quality
        </Link>
        , and monitor agents in production.
      </>
    ),
    answerText:
      "An agent platform is a specialized AI platform designed for building and operating autonomous AI agents. Agents use LLMs to reason, plan, and execute multi-step tasks by calling tools and APIs. An agent platform provides the infrastructure to trace agent execution, evaluate agent quality, and monitor agents in production.",
  },
  {
    question:
      "What is the difference between an AI platform and an LLM platform?",
    answer:
      "An LLM platform focuses specifically on applications built around large language models, such as chatbots and text generation tools. An AI platform is broader and includes LLM apps, autonomous agents, RAG systems, and classical ML models. MLflow supports the full spectrum—from single LLM calls to complex multi-agent workflows—making it both an LLM platform and a comprehensive AI platform.",
  },
  {
    question: "How is MLflow different from other AI platforms?",
    answer: (
      <>
        MLflow is open-source and vendor-neutral. Unlike proprietary platforms,
        MLflow gives you full ownership of your data and avoids lock-in to any
        single cloud or LLM provider. It integrates with 20+ frameworks
        including{" "}
        <Link
          href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langchain.html"
          style={{ color: "#0194e2", fontWeight: "600" }}
        >
          LangChain
        </Link>
        ,{" "}
        <Link
          href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langgraph.html"
          style={{ color: "#0194e2", fontWeight: "600" }}
        >
          LangGraph
        </Link>
        ,{" "}
        <Link
          href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/llamaindex.html"
          style={{ color: "#0194e2", fontWeight: "600" }}
        >
          LlamaIndex
        </Link>
        , and{" "}
        <Link
          href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/crewai.html"
          style={{ color: "#0194e2", fontWeight: "600" }}
        >
          CrewAI
        </Link>
        , and its{" "}
        <Link
          href="https://mlflow.org/docs/latest/genai/tracing/"
          style={{ color: "#0194e2", fontWeight: "600" }}
        >
          OpenTelemetry-compatible tracing
        </Link>{" "}
        ensures portability.
      </>
    ),
    answerText:
      "MLflow is open-source and vendor-neutral. Unlike proprietary platforms, MLflow gives you full ownership of your data and avoids lock-in to any single cloud or LLM provider. It integrates with 20+ frameworks including LangChain, LangGraph, LlamaIndex, and CrewAI, and its OpenTelemetry-compatible tracing ensures portability.",
  },
  {
    question: "What features should I look for in an AI agent platform?",
    answer:
      "A strong AI agent platform should provide end-to-end tracing for debugging multi-step agent workflows, automated evaluation with LLM judges for measuring quality at scale, prompt management for versioning and iterating on instructions, production monitoring for catching regressions, and support for multiple LLM providers and agent frameworks. Open-source options like MLflow offer all of these without vendor lock-in.",
  },
  {
    question: "How do I evaluate AI agents?",
    answer: (
      <>
        Evaluating agents requires measuring quality across multi-step
        workflows, not just individual LLM calls. MLflow's{" "}
        <Link
          href="https://mlflow.org/docs/latest/genai/eval-monitor/"
          style={{ color: "#0194e2", fontWeight: "600" }}
        >
          evaluation framework
        </Link>{" "}
        provides LLM-as-a-judge scorers for metrics like correctness,
        hallucination, and relevance. You can run evaluations on datasets, apply
        scorers to production traces, and compare results across agent versions
        in the MLflow UI.
      </>
    ),
    answerText:
      "Evaluating agents requires measuring quality across multi-step workflows, not just individual LLM calls. MLflow's evaluation framework provides LLM-as-a-judge scorers for metrics like correctness, hallucination, and relevance. You can run evaluations on datasets, apply scorers to production traces, and compare results across agent versions in the MLflow UI.",
  },
  {
    question: "Can I use MLflow with any LLM provider?",
    answer: (
      <>
        Yes. MLflow's{" "}
        <Link
          href="/genai/ai-gateway"
          style={{ color: "#0194e2", fontWeight: "600" }}
        >
          AI Gateway
        </Link>{" "}
        provides a unified interface for routing requests to OpenAI, Anthropic,
        Google, Amazon Bedrock, Azure OpenAI, and other providers. You can
        switch between providers without changing application code, manage API
        keys centrally, and enforce rate limits and access policies.
      </>
    ),
    answerText:
      "Yes. MLflow's AI Gateway provides a unified interface for routing requests to OpenAI, Anthropic, Google, Amazon Bedrock, Azure OpenAI, and other providers. You can switch between providers without changing application code, manage API keys centrally, and enforce rate limits and access policies.",
  },
  {
    question:
      "What is the difference between an AI platform and an agent framework?",
    answer:
      "An agent framework (like LangGraph, CrewAI, or AutoGen) provides the building blocks for constructing agents—tools, memory, planning loops, and orchestration patterns. An AI platform sits above frameworks and provides the operational layer: tracing to see what your agent did, evaluation to measure whether it did well, and monitoring to catch issues in production. MLflow integrates with all major agent frameworks to provide this operational layer.",
  },
  {
    question: "How do I monitor AI agents in production?",
    answer: (
      <>
        Production monitoring for AI agents requires continuous evaluation of
        trace data. MLflow lets you{" "}
        <Link
          href="https://mlflow.org/docs/latest/genai/tracing/"
          style={{ color: "#0194e2", fontWeight: "600" }}
        >
          instrument your agents with tracing
        </Link>{" "}
        to capture every LLM call, tool invocation, and decision step. You can
        then apply{" "}
        <Link
          href="https://mlflow.org/docs/latest/genai/eval-monitor/"
          style={{ color: "#0194e2", fontWeight: "600" }}
        >
          automated scorers
        </Link>{" "}
        to production traces to detect quality regressions, track cost and
        latency trends, and surface issues before users report them.
      </>
    ),
    answerText:
      "Production monitoring for AI agents requires continuous evaluation of trace data. MLflow lets you instrument your agents with tracing to capture every LLM call, tool invocation, and decision step. You can then apply automated scorers to production traces to detect quality regressions, track cost and latency trends, and surface issues before users report them.",
  },
  {
    question: "Is MLflow free to use as an AI platform?",
    answer: (
      <>
        MLflow is completely free and open-source under the Apache 2.0 license.
        You can self-host it or use{" "}
        <Link
          href="https://databricks.com/product/managed-mlflow"
          style={{ color: "#0194e2", fontWeight: "600" }}
        >
          managed MLflow on Databricks
        </Link>{" "}
        for a hosted experience. The open-source version includes all core
        capabilities: tracing, evaluation, prompt registry, AI gateway, and
        experiment tracking.
      </>
    ),
    answerText:
      "MLflow is completely free and open-source under the Apache 2.0 license. You can self-host it or use managed MLflow on Databricks for a hosted experience. The open-source version includes all core capabilities: tracing, evaluation, prompt registry, AI gateway, and experiment tracking.",
  },
  {
    question: "How do I get started with MLflow as an AI platform?",
    answer: (
      <>
        Getting started takes three steps: install MLflow with{" "}
        <code>pip install mlflow</code>, add one line of tracing to your app,
        and open the MLflow UI to see your traces. See the{" "}
        <Link
          href="https://mlflow.org/docs/latest/genai/tracing/quickstart/"
          style={{ color: "#0194e2", fontWeight: "600" }}
        >
          quickstart guide
        </Link>{" "}
        for a complete walkthrough.
      </>
    ),
    answerText:
      "Getting started takes three steps: install MLflow with pip install mlflow, add one line of tracing to your app, and open the MLflow UI to see your traces. See the quickstart guide for a complete walkthrough.",
  },
  {
    question: "What GenAI frameworks does MLflow integrate with?",
    answer: (
      <>
        MLflow provides automatic tracing integrations for over 20 frameworks
        and providers, including{" "}
        <Link
          href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langchain.html"
          style={{ color: "#0194e2", fontWeight: "600" }}
        >
          LangChain
        </Link>
        ,{" "}
        <Link
          href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langgraph.html"
          style={{ color: "#0194e2", fontWeight: "600" }}
        >
          LangGraph
        </Link>
        ,{" "}
        <Link
          href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/llamaindex.html"
          style={{ color: "#0194e2", fontWeight: "600" }}
        >
          LlamaIndex
        </Link>
        ,{" "}
        <Link
          href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/crewai.html"
          style={{ color: "#0194e2", fontWeight: "600" }}
        >
          CrewAI
        </Link>
        , OpenAI, Anthropic, Google Gemini, Amazon Bedrock, and more. Each
        integration captures traces with a single line of code, giving you full
        visibility into your application&apos;s execution.
      </>
    ),
    answerText:
      "MLflow provides automatic tracing integrations for over 20 frameworks and providers, including LangChain, LangGraph, LlamaIndex, CrewAI, OpenAI, Anthropic, Google Gemini, Amazon Bedrock, and more. Each integration captures traces with a single line of code, giving you full visibility into your application's execution.",
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
      text: faq.answerText || faq.answer,
    },
  })),
};

const productSchema = {
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
};

const tracingExample = `import mlflow
from openai import OpenAI

# Enable tracing with one line
mlflow.openai.autolog()

client = OpenAI()

# Every call is now traced automatically
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize this document."}],
)`;

const agentTracingExample = `import mlflow
from langgraph.graph import StateGraph

# Trace your entire agent workflow
mlflow.langgraph.autolog()

# Build your agent as usual
graph = StateGraph(AgentState)
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("reviewer", reviewer_node)

# Run the agent — every step is captured
app = graph.compile()
result = app.invoke({"task": "Research competitor pricing"})`;

const evalExample = `import mlflow
from mlflow.genai.scorers import Correctness, Hallucination

# Evaluate your agent across a dataset
results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_agent,
    scorers=[Correctness(), Hallucination()],
)

# View results in the MLflow UI
print(results.tables["eval_results"])`;

function CodeBlock({
  code,
  language = "python",
  label,
}: {
  code: string;
  language?: string;
  label?: string;
}) {
  return (
    <>
      {label && (
        <p style={{ marginTop: "32px", marginBottom: "0px" }}>
          <strong>{label}</strong>
        </p>
      )}
      <div
        className="rounded-lg border border-white/10 overflow-hidden"
        style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
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
    </>
  );
}

export default function AIAgentPlatform() {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(null);

  return (
    <Layout>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="article" />
        <link rel="canonical" href="https://mlflow.org/ai-agent-platform" />
        <script type="application/ld+json">{JSON.stringify(faqSchema)}</script>
        <script type="application/ld+json">
          {JSON.stringify(productSchema)}
        </script>
      </Head>

      <div
        style={{
          backgroundColor: "#FFFFFF",
          minHeight: "100vh",
          fontFamily:
            "'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        }}
      >
        <style>{`
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
          }
          .article-container h2 {
            font-size: 32px;
            font-weight: 700;
            margin-top: 48px;
            margin-bottom: 20px;
            color: #1a1a1a;
          }
          .article-container h3 {
            font-size: 24px;
            font-weight: 600;
            margin-top: 36px;
            margin-bottom: 16px;
            color: #1a1a1a;
          }
          .article-container p {
            font-size: 16px;
            line-height: 1.7;
            color: #3d3d3d;
            margin-bottom: 20px;
          }
          .article-container a {
            color: #0194e2;
            font-weight: 600;
            text-decoration: none;
          }
          .article-container a:hover {
            text-decoration: underline;
          }
          .faq-list {
            margin: 40px 0;
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
            font-family: 'DM Sans', sans-serif;
          }
          .faq-chevron {
            transition: transform 0.3s;
            color: #6b7280;
            flex-shrink: 0;
            margin-left: 16px;
          }
          .faq-chevron.open {
            transform: rotate(180deg);
          }
          .faq-answer {
            padding-bottom: 20px;
            color: #3d3d3d;
            line-height: 1.7;
            font-size: 16px;
          }
          .faq-answer a {
            font-weight: 600;
            color: #0194e2 !important;
            text-decoration: none;
          }
          .faq-answer a:hover {
            text-decoration: underline;
          }
          .quick-nav {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 20px 24px;
            margin: 32px 0;
          }
          .quick-nav ul {
            margin: 8px 0 0 0 !important;
            padding-left: 0 !important;
            list-style: none !important;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
          }
          .quick-nav li {
            margin: 0 !important;
            padding: 0 !important;
          }
          .quick-nav a {
            display: inline-block;
            padding: 6px 14px;
            background: #e0f2fe;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
            color: #0369a1 !important;
            text-decoration: none;
          }
          .quick-nav a:hover {
            background: #bae6fd;
          }
        `}</style>

        {/* Hero Section */}
        <div
          style={{
            background: "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
            padding: "80px 20px 60px",
            textAlign: "center",
          }}
        >
          <div style={{ maxWidth: "800px", margin: "0 auto" }}>
            <h1
              style={{
                fontSize: "48px",
                fontWeight: 800,
                color: "#FFFFFF",
                lineHeight: 1.2,
                marginBottom: "20px",
              }}
            >
              AI Platform for LLM Apps and Agents
            </h1>
            <p
              style={{
                fontSize: "20px",
                color: "#94a3b8",
                lineHeight: 1.6,
                maxWidth: "650px",
                margin: "0 auto 32px",
              }}
            >
              The open-source agent engineering and ops platform to build,
              evaluate, and deploy production-quality AI applications.
            </p>
            <Link
              href="https://mlflow.org/docs/latest/genai/"
              style={{
                display: "inline-block",
                padding: "14px 32px",
                backgroundColor: "#0194e2",
                color: "#FFFFFF",
                borderRadius: "8px",
                fontWeight: 600,
                fontSize: "16px",
                textDecoration: "none",
              }}
            >
              Get Started with MLflow
            </Link>
          </div>
        </div>

        {/* Article Content */}
        <div
          style={{ maxWidth: "1200px", margin: "0 auto", padding: "40px 20px" }}
        >
          <div
            className="article-container"
            style={{ maxWidth: "900px", margin: "0 auto" }}
          >
            {/* What is an AI Platform */}
            <p>
              An AI platform is an integrated environment for building, testing,
              deploying, and monitoring AI applications throughout their entire
              lifecycle. Whether you&apos;re building LLM applications with
              OpenAI or Claude, deploying autonomous agents with LangGraph and
              CrewAI, or running RAG pipelines with LlamaIndex, an AI platform
              provides the operational layer you need to ship with confidence.
              For LLM-powered applications, this operational layer is commonly
              called an <a href="#llm-platform">LLM platform</a>. For autonomous
              agents that reason and take actions, it is called an{" "}
              <a href="#agent-platform">agent platform</a>.
            </p>
            <p>
              MLflow is the leading open-source{" "}
              <a href="#agent-platform">agent platform</a> used by thousands of
              teams to move from prototype to production. It provides{" "}
              <Link
                href="https://mlflow.org/docs/latest/genai/tracing/"
                style={{ color: "#0194e2", fontWeight: "600" }}
              >
                end-to-end tracing
              </Link>{" "}
              for debugging,{" "}
              <Link
                href="https://mlflow.org/docs/latest/genai/eval-monitor/"
                style={{ color: "#0194e2", fontWeight: "600" }}
              >
                automated evaluation
              </Link>{" "}
              for measuring quality, a{" "}
              <Link
                href="/genai/prompt-registry"
                style={{ color: "#0194e2", fontWeight: "600" }}
              >
                prompt registry
              </Link>{" "}
              for managing prompt templates, and an{" "}
              <Link
                href="/genai/ai-gateway"
                style={{ color: "#0194e2", fontWeight: "600" }}
              >
                AI gateway
              </Link>{" "}
              for unified access to LLM providers—all open-source and
              vendor-neutral.
            </p>

            {/* Quick Navigation */}
            <div className="quick-nav">
              <p style={{ margin: 0, fontWeight: 600, color: "#1a1a1a" }}>
                Quick Navigation
              </p>
              <ul>
                <li>
                  <a href="#why-ai-platform">Why You Need an AI Platform</a>
                </li>
                <li>
                  <a href="#llm-platform">LLM Platform</a>
                </li>
                <li>
                  <a href="#agent-platform">Agent Platform</a>
                </li>
                <li>
                  <a href="#key-components">Key Components</a>
                </li>
                <li>
                  <a href="#use-cases">Use Cases</a>
                </li>
                <li>
                  <a href="#how-to-implement">How to Implement</a>
                </li>
                <li>
                  <a href="#open-source-vs-proprietary">
                    Open Source vs. Proprietary
                  </a>
                </li>
                <li>
                  <a href="#faq">FAQ</a>
                </li>
              </ul>
            </div>

            {/* Why You Need an AI Platform */}
            <h2 id="why-ai-platform">Why You Need an AI Platform</h2>
            <p>
              Building an LLM-powered application or agent is straightforward.
              Making it production-ready is not. Teams encounter the same
              challenges once they move past the prototype stage:
            </p>
            <ul>
              <li>
                <strong>Debugging is opaque:</strong> LLM applications involve
                multiple steps—retrieval, reasoning, tool calls, prompt
                construction—and failures can happen at any point. Without
                tracing, you cannot see what went wrong or why.
              </li>
              <li>
                <strong>Quality is hard to measure:</strong> Free-form language
                output cannot be validated with unit tests. You need specialized
                evaluation methods like LLM-as-a-judge to assess correctness,
                hallucination, relevance, and safety at scale.
              </li>
              <li>
                <strong>Prompts drift silently:</strong> A small change to a
                system prompt can alter behavior across thousands of
                interactions. Without version control for prompts, regressions
                go unnoticed until users complain.
              </li>
              <li>
                <strong>Provider management grows complex:</strong> Routing
                requests across OpenAI, Anthropic, Gemini, and Bedrock while
                managing API keys, rate limits, and fallback logic creates
                operational overhead that compounds over time.
              </li>
              <li>
                <strong>Production monitoring lacks coverage:</strong> Classical
                APM tools were not designed for AI workloads. You need
                AI-specific monitoring that evaluates trace quality, tracks
                token costs, and surfaces regressions continuously.
              </li>
            </ul>

            {/* LLM Platform */}
            <h2 id="llm-platform">LLM Platform</h2>
            <p>
              An LLM platform provides the infrastructure to build and operate
              applications powered by large language models. This includes
              chatbots, text summarizers, question-answering systems, and any
              application that relies on LLM inference as a core component.
            </p>
            <p>
              The defining capabilities of an LLM platform are tracing LLM calls
              (capturing prompts, completions, token counts, and latency),
              evaluating response quality with automated scorers, managing
              prompt templates with version control, and providing a gateway
              layer for routing requests to multiple LLM providers.
            </p>
            <p>
              MLflow serves as an LLM platform with{" "}
              <Link
                href="https://mlflow.org/docs/latest/genai/tracing/"
                style={{ color: "#0194e2", fontWeight: "600" }}
              >
                one-line tracing integrations
              </Link>{" "}
              for OpenAI, Anthropic, Google Gemini, Amazon Bedrock, and 20+
              other providers. Every LLM call is captured automatically with
              full input/output data, enabling you to debug issues without
              reproducing them.
            </p>

            {/* Agent Platform */}
            <h2 id="agent-platform">Agent Platform</h2>
            <p>
              An agent platform extends the LLM platform with capabilities
              designed for autonomous AI agents. Agents differ from simple LLM
              applications because they reason across multiple steps, make
              decisions, call tools, and maintain state over extended
              interactions.
            </p>
            <p>
              Debugging an agent requires visibility into the full execution
              graph—not just individual LLM calls, but the planning steps, tool
              invocations, memory retrievals, and decision branches that led to
              the final output. An agent platform captures this entire chain and
              provides tools to evaluate whether the agent&apos;s reasoning and
              actions were correct.
            </p>
            <p>
              MLflow provides native tracing integrations for leading agent
              frameworks, including{" "}
              <Link
                href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langgraph.html"
                style={{ color: "#0194e2", fontWeight: "600" }}
              >
                LangGraph
              </Link>
              ,{" "}
              <Link
                href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/crewai.html"
                style={{ color: "#0194e2", fontWeight: "600" }}
              >
                CrewAI
              </Link>
              , AutoGen, and Anthropic&apos;s Claude Agent SDK. Each integration
              captures the complete execution trace—planner calls, tool
              invocations, intermediate reasoning, and final outputs—with a
              single line of code.
            </p>

            {/* Key Components */}
            <h2 id="key-components">Key Components of an AI Platform</h2>
            <ul>
              <li>
                <Link
                  href="https://mlflow.org/docs/latest/genai/tracing/"
                  style={{ color: "#0194e2", fontWeight: "600" }}
                >
                  Tracing
                </Link>
                : Capture complete execution traces including prompts, tool
                calls, retrievals, and responses. MLflow&apos;s tracing is
                OpenTelemetry-compatible and supports 20+ frameworks with
                one-line auto-instrumentation.
              </li>
              <li>
                <Link
                  href="https://mlflow.org/docs/latest/genai/eval-monitor/"
                  style={{ color: "#0194e2", fontWeight: "600" }}
                >
                  Evaluation
                </Link>
                : Measure quality at scale using LLM-as-a-judge scorers for
                correctness, hallucination, relevance, toxicity, and custom
                metrics. Run evaluations on datasets or apply them continuously
                to production traces.
              </li>
              <li>
                <Link
                  href="/genai/prompt-registry"
                  style={{ color: "#0194e2", fontWeight: "600" }}
                >
                  Prompt Registry
                </Link>
                : Version, compare, and iterate on prompt templates. Track which
                prompt versions are used by which application versions and
                measure the impact of prompt changes on quality.
              </li>
              <li>
                <Link
                  href="/genai/ai-gateway"
                  style={{ color: "#0194e2", fontWeight: "600" }}
                >
                  AI Gateway
                </Link>
                : Route requests to any LLM provider through a unified
                interface. Manage API keys centrally, enforce rate limits, set
                fallback routes, and track usage across providers.
              </li>
              <li>
                <Link
                  href="https://mlflow.org/docs/latest/genai/tracing/production-tracing/"
                  style={{ color: "#0194e2", fontWeight: "600" }}
                >
                  Production Monitoring
                </Link>
                : Apply automated scorers to production traces continuously.
                Detect quality regressions, track cost and latency trends, and
                surface issues before users report them.
              </li>
              <li>
                <Link
                  href="/genai/human-feedback"
                  style={{ color: "#0194e2", fontWeight: "600" }}
                >
                  Human Feedback
                </Link>
                : Collect structured feedback from users and reviewers. Annotate
                traces with quality assessments, build evaluation datasets from
                real interactions, and close the feedback loop between
                production and development.
              </li>
            </ul>

            {/* Use Cases */}
            <h2 id="use-cases">Common Use Cases</h2>
            <ul>
              <li>
                <strong>Debugging Agent Failures:</strong> When an autonomous
                agent takes the wrong action, MLflow&apos;s trace UI shows the
                complete reasoning chain—which tools were called, what context
                was retrieved, and where the decision went wrong—enabling rapid
                root cause analysis.
              </li>
              <li>
                <strong>Evaluating RAG Quality:</strong> For{" "}
                <Link
                  href="https://mlflow.org/docs/latest/genai/eval-monitor/"
                  style={{ color: "#0194e2", fontWeight: "600" }}
                >
                  RAG applications
                </Link>
                , automated scorers measure whether retrieved documents are
                relevant, whether the generated answer is faithful to the source
                material, and whether the response actually addresses the
                question.
              </li>
              <li>
                <strong>Iterating on Prompts:</strong> The{" "}
                <Link
                  href="/genai/prompt-registry"
                  style={{ color: "#0194e2", fontWeight: "600" }}
                >
                  prompt registry
                </Link>{" "}
                lets you version prompt templates, compare quality metrics
                across versions, and roll back to previous prompts if a change
                causes regressions.
              </li>
              <li>
                <strong>Multi-Provider Cost Optimization:</strong> The{" "}
                <Link
                  href="/genai/ai-gateway"
                  style={{ color: "#0194e2", fontWeight: "600" }}
                >
                  AI gateway
                </Link>{" "}
                combined with{" "}
                <Link
                  href="https://mlflow.org/docs/latest/genai/tracing/token-usage-cost/"
                  style={{ color: "#0194e2", fontWeight: "600" }}
                >
                  cost tracking
                </Link>{" "}
                helps you route requests to the most cost-effective provider for
                each use case while monitoring quality to ensure cheaper models
                meet your standards.
              </li>
              <li>
                <strong>Continuous Quality Monitoring:</strong> Apply scorers to
                every production trace to detect quality drift, hallucination
                spikes, or latency increases. Set up alerts based on score
                thresholds to catch regressions early.
              </li>
              <li>
                <strong>Safety and Governance:</strong> Use{" "}
                <Link
                  href="https://mlflow.org/docs/latest/genai/guides/responsible-ai/"
                  style={{ color: "#0194e2", fontWeight: "600" }}
                >
                  responsible AI guardrails
                </Link>{" "}
                to detect and filter toxic, biased, or personally identifiable
                content in both inputs and outputs across your entire AI stack.
              </li>
            </ul>

            {/* How to Implement */}
            <h2 id="how-to-implement">
              How to Implement an AI Platform with MLflow
            </h2>
            <p>
              MLflow makes it straightforward to add platform-level capabilities
              to any LLM application or agent. Here are examples for common
              scenarios.
            </p>

            <CodeBlock
              label="Trace an LLM application (OpenAI)"
              code={tracingExample}
            />

            <CodeBlock
              label="Trace a multi-step agent (LangGraph)"
              code={agentTracingExample}
            />

            <CodeBlock label="Evaluate agent quality" code={evalExample} />

            <p style={{ marginTop: "32px" }}>
              Check out the{" "}
              <Link
                href="https://mlflow.org/docs/latest/genai/tracing/integrations/"
                style={{ color: "#0194e2", fontWeight: "600" }}
              >
                MLflow tracing integrations documentation
              </Link>{" "}
              for examples with LangChain, LangGraph, LlamaIndex, CrewAI,
              Anthropic, Vercel AI SDK, and other frameworks.
            </p>

            {/* Open Source vs. Proprietary */}
            <h2 id="open-source-vs-proprietary">
              Open Source vs. Proprietary AI Platforms
            </h2>
            <p>
              Choosing between open-source and proprietary AI platforms involves
              trade-offs across cost, flexibility, and operational burden. Here
              is how they compare:
            </p>
            <div
              style={{
                overflowX: "auto",
                margin: "24px 0",
                border: "1px solid #e5e7eb",
                borderRadius: "8px",
              }}
            >
              <table
                style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  fontSize: "15px",
                }}
              >
                <thead>
                  <tr style={{ backgroundColor: "#f8fafc" }}>
                    <th
                      style={{
                        padding: "14px 16px",
                        textAlign: "left",
                        fontWeight: 600,
                        borderBottom: "2px solid #e5e7eb",
                      }}
                    >
                      Criteria
                    </th>
                    <th
                      style={{
                        padding: "14px 16px",
                        textAlign: "left",
                        fontWeight: 600,
                        borderBottom: "2px solid #e5e7eb",
                      }}
                    >
                      Open Source (MLflow)
                    </th>
                    <th
                      style={{
                        padding: "14px 16px",
                        textAlign: "left",
                        fontWeight: 600,
                        borderBottom: "2px solid #e5e7eb",
                      }}
                    >
                      Proprietary Platforms
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    [
                      "Cost",
                      "Free, Apache 2.0 license",
                      "Subscription or usage-based pricing",
                    ],
                    [
                      "Data Ownership",
                      "Full ownership, self-hosted",
                      "Data stored on vendor infrastructure",
                    ],
                    [
                      "Vendor Lock-in",
                      "None—OpenTelemetry-compatible",
                      "Tied to vendor format and tooling",
                    ],
                    [
                      "Customization",
                      "Full access to source code",
                      "Limited to vendor-provided options",
                    ],
                    [
                      "Framework Support",
                      "20+ integrations, extensible",
                      "Varies by vendor",
                    ],
                    [
                      "Managed Option",
                      "Available via Databricks",
                      "Built-in hosted offering",
                    ],
                    [
                      "Community",
                      "20M+ downloads/month, active contributors",
                      "Vendor-controlled development",
                    ],
                  ].map((row, i) => (
                    <tr
                      key={i}
                      style={{
                        borderBottom: "1px solid #e5e7eb",
                      }}
                    >
                      <td
                        style={{
                          padding: "12px 16px",
                          fontWeight: 600,
                          color: "#1a1a1a",
                        }}
                      >
                        {row[0]}
                      </td>
                      <td style={{ padding: "12px 16px", color: "#3d3d3d" }}>
                        {row[1]}
                      </td>
                      <td style={{ padding: "12px 16px", color: "#3d3d3d" }}>
                        {row[2]}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Related Resources */}
            <h2 id="resources">Related Resources</h2>
            <ul>
              <li>
                <Link
                  href="https://mlflow.org/docs/latest/genai/"
                  style={{ color: "#0194e2", fontWeight: "600" }}
                >
                  MLflow GenAI Documentation
                </Link>
                : Complete reference for all GenAI capabilities
              </li>
              <li>
                <Link
                  href="https://mlflow.org/docs/latest/genai/tracing/quickstart/"
                  style={{ color: "#0194e2", fontWeight: "600" }}
                >
                  Tracing Quickstart
                </Link>
                : Get tracing running in under five minutes
              </li>
              <li>
                <Link
                  href="https://mlflow.org/docs/latest/genai/eval-monitor/"
                  style={{ color: "#0194e2", fontWeight: "600" }}
                >
                  Evaluation and Monitoring Guide
                </Link>
                : Set up automated quality measurement
              </li>
              <li>
                <Link
                  href="/genai/observability"
                  style={{ color: "#0194e2", fontWeight: "600" }}
                >
                  Observability Overview
                </Link>
                : Deep dive into MLflow&apos;s observability capabilities
              </li>
              <li>
                <Link
                  href="/genai/evaluations"
                  style={{ color: "#0194e2", fontWeight: "600" }}
                >
                  Evaluations Overview
                </Link>
                : Explore MLflow&apos;s evaluation framework
              </li>
              <li>
                <Link
                  href="https://mlflow.org/docs/latest/genai/tracing/integrations/"
                  style={{ color: "#0194e2", fontWeight: "600" }}
                >
                  Framework Integrations
                </Link>
                : LangChain, LangGraph, LlamaIndex, CrewAI, and more
              </li>
            </ul>

            {/* FAQ */}
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
                      &#9660;
                    </span>
                  </button>
                  {openFaqIndex === index && (
                    <div className="faq-answer">{faq.answer}</div>
                  )}
                </div>
              ))}
            </div>

            {/* Footer CTA */}
            <div
              style={{
                textAlign: "center",
                padding: "48px 0",
                marginTop: "40px",
                borderTop: "1px solid #e5e7eb",
              }}
            >
              <h2
                style={{
                  fontSize: "28px",
                  fontWeight: 700,
                  color: "#1a1a1a",
                  marginBottom: "16px",
                }}
              >
                Start Building with MLflow
              </h2>
              <p
                style={{
                  fontSize: "16px",
                  color: "#6b7280",
                  marginBottom: "24px",
                }}
              >
                Add tracing to your AI application in under five minutes.
              </p>
              <Link
                href="https://mlflow.org/docs/latest/genai/tracing/quickstart/"
                style={{
                  display: "inline-block",
                  padding: "14px 32px",
                  backgroundColor: "#0194e2",
                  color: "#FFFFFF",
                  borderRadius: "8px",
                  fontWeight: 600,
                  fontSize: "16px",
                  textDecoration: "none",
                }}
              >
                Get Started
              </Link>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}
