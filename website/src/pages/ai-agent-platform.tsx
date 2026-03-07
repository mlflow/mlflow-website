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
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
          trace agent execution
        </Link>
        ,{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
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
          href={
            MLFLOW_GENAI_DOCS_URL +
            "tracing/integrations/listing/langchain.html"
          }
        >
          LangChain
        </Link>
        ,{" "}
        <Link
          href={
            MLFLOW_GENAI_DOCS_URL +
            "tracing/integrations/listing/langgraph.html"
          }
        >
          LangGraph
        </Link>
        ,{" "}
        <Link
          href={
            MLFLOW_GENAI_DOCS_URL +
            "tracing/integrations/listing/llamaindex.html"
          }
        >
          LlamaIndex
        </Link>
        , and{" "}
        <Link
          href={
            MLFLOW_GENAI_DOCS_URL + "tracing/integrations/listing/crewai.html"
          }
        >
          CrewAI
        </Link>
        , and its{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
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
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
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
        Yes. MLflow's <Link href="/genai/ai-gateway">AI Gateway</Link> provides
        a unified interface for routing requests to OpenAI, Anthropic, Google,
        Amazon Bedrock, Azure OpenAI, and other providers. You can switch
        between providers without changing application code, manage API keys
        centrally, and enforce rate limits and access policies.
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
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
          instrument your agents with tracing
        </Link>{" "}
        to capture every LLM call, tool invocation, and decision step. You can
        then apply{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
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
        <Link href="https://databricks.com/product/managed-mlflow">
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
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/quickstart/"}>
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
          href={
            MLFLOW_GENAI_DOCS_URL +
            "tracing/integrations/listing/langchain.html"
          }
        >
          LangChain
        </Link>
        ,{" "}
        <Link
          href={
            MLFLOW_GENAI_DOCS_URL +
            "tracing/integrations/listing/langgraph.html"
          }
        >
          LangGraph
        </Link>
        ,{" "}
        <Link
          href={
            MLFLOW_GENAI_DOCS_URL +
            "tracing/integrations/listing/llamaindex.html"
          }
        >
          LlamaIndex
        </Link>
        ,{" "}
        <Link
          href={
            MLFLOW_GENAI_DOCS_URL + "tracing/integrations/listing/crewai.html"
          }
        >
          CrewAI
        </Link>
        , OpenAI, Anthropic, Google Gemini, Amazon Bedrock, and more. Each
        integration captures traces with a single line of code, giving you full
        visibility into your application's execution.
      </>
    ),
    answerText:
      "MLflow provides automatic tracing integrations for over 20 frameworks and providers, including LangChain, LangGraph, LlamaIndex, CrewAI, OpenAI, Anthropic, Google Gemini, Amazon Bedrock, and more. Each integration captures traces with a single line of code, giving you full visibility into your application's execution.",
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
    "Open-source platform for building, evaluating, and deploying LLM apps and AI agents.",
  url: "https://mlflow.org",
  license: "https://www.apache.org/licenses/LICENSE-2.0",
};

export default function AIAgentPlatform() {
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
          content="https://mlflow.org/ai-agent-platform"
        />
        <link rel="canonical" href="https://mlflow.org/ai-agent-platform" />
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
          <h1>AI Platform for LLM Apps and Agents</h1>

          <p>
            An AI platform is an integrated environment for building, testing,
            deploying, and monitoring AI applications throughout their entire
            lifecycle. Whether you're building LLM applications with OpenAI or
            Claude, deploying autonomous agents with LangGraph and CrewAI, or
            running RAG pipelines with LlamaIndex, an AI platform provides the
            operational layer you need to ship with confidence. For LLM-powered
            applications, this operational layer is commonly called an{" "}
            <a href="#llm-platform">LLM platform</a>. For autonomous agents that
            reason and take actions, it is called an{" "}
            <a href="#agent-platform">agent platform</a>.
          </p>

          <p>
            MLflow is the leading open-source{" "}
            <a href="#agent-platform">agent platform</a> used by thousands of
            teams to move from prototype to production. It provides{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
              end-to-end tracing
            </Link>{" "}
            for debugging,{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
              automated evaluation
            </Link>{" "}
            for measuring quality, a{" "}
            <Link href="/genai/prompt-registry">prompt registry</Link> for
            managing prompt templates, and an{" "}
            <Link href="/genai/ai-gateway">AI gateway</Link> for unified access
            to LLM providers—all open-source and vendor-neutral.
          </p>

          <p>
            Unlike traditional software, AI applications are{" "}
            <strong>non-deterministic</strong>: the same input can produce
            different outputs depending on model state, retrieved context, and
            multi-step agent reasoning. This makes traditional logging and
            monitoring insufficient. An AI platform captures the full execution
            context (
            <Link href={MLFLOW_GENAI_DOCS_URL + "prompts/"}>prompts</Link>,
            model responses,{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>tool calls</Link>,
            retrieval results, and{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "llm-evaluate/"}>
              evaluation scores
            </Link>
            ) so teams can understand the "why" behind every output.
          </p>

          <h2 id="why-ai-platform">Why You Need an AI Platform</h2>

          <p>
            Building an LLM-powered application or agent is straightforward.
            Making it production-ready is not. Teams encounter the same
            challenges once they move past the prototype stage:
          </p>

          <div className="grid-2">
            <div className="card">
              <h3>Debugging is Opaque</h3>
              <p>
                <strong>Problem:</strong> LLM applications involve multiple
                steps—retrieval, reasoning, tool calls, prompt construction—and
                failures can happen at any point. Without tracing, you cannot
                see what went wrong or why.
              </p>
              <p>
                <strong>Solution:</strong> End-to-end tracing makes every step
                visible and debuggable, from initial request to final response.
              </p>
            </div>

            <div className="card">
              <h3>Quality is Hard to Measure</h3>
              <p>
                <strong>Problem:</strong> Free-form language output cannot be
                validated with unit tests. You need specialized evaluation
                methods to assess correctness, hallucination, and relevance at
                scale.
              </p>
              <p>
                <strong>Solution:</strong> LLM-as-a-judge evaluation
                automatically scores every response against quality benchmarks.
              </p>
            </div>

            <div className="card">
              <h3>Prompts Drift Silently</h3>
              <p>
                <strong>Problem:</strong> A small change to a system prompt can
                alter behavior across thousands of interactions. Without version
                control, regressions go unnoticed.
              </p>
              <p>
                <strong>Solution:</strong> A prompt registry versions, compares,
                and tracks the impact of prompt changes on quality metrics.
              </p>
            </div>

            <div className="card">
              <h3>Provider Management Grows Complex</h3>
              <p>
                <strong>Problem:</strong> Routing requests across OpenAI,
                Anthropic, Gemini, and Bedrock while managing API keys, rate
                limits, and fallback logic creates compounding overhead.
              </p>
              <p>
                <strong>Solution:</strong> An AI gateway provides a unified
                interface for all providers with central key management and
                fallback routing.
              </p>
            </div>
          </div>

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
            evaluating response quality with automated scorers, managing prompt
            templates with version control, and providing a gateway layer for
            routing requests to multiple LLM providers.
          </p>

          <p>
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
              MLflow serves as an LLM platform
            </Link>{" "}
            with one-line tracing integrations for OpenAI, Anthropic, Google
            Gemini, Amazon Bedrock, and 20+ other providers. Every LLM call is
            captured automatically with full input/output data, enabling you to
            debug issues without reproducing them.
          </p>

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
            provides tools to evaluate whether the agent's reasoning and actions
            were correct.
          </p>

          <p>
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
              MLflow provides native tracing integrations
            </Link>{" "}
            for leading agent frameworks, including{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "tracing/integrations/listing/langgraph.html"
              }
            >
              LangGraph
            </Link>
            ,{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "tracing/integrations/listing/crewai.html"
              }
            >
              CrewAI
            </Link>
            , AutoGen, and Anthropic's Claude Agent SDK. Each integration
            captures the complete execution trace—planner calls, tool
            invocations, intermediate reasoning, and final outputs—with a single
            line of code.
          </p>

          <h2 id="key-components">Key Components of an AI Platform</h2>

          <p>A comprehensive AI platform combines six capabilities:</p>

          <ul>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "tracing/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Tracing
              </Link>
              : Capture complete execution traces including prompts, tool calls,
              retrievals, and responses. MLflow's tracing is
              OpenTelemetry-compatible and supports 20+ frameworks with one-line
              auto-instrumentation.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Evaluation
              </Link>
              : Measure quality at scale using{" "}
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "llm-evaluate/llm-as-judge/"}
                style={{ color: "#007bff" }}
              >
                LLM-as-a-judge scorers
              </Link>{" "}
              for correctness, hallucination, relevance, toxicity, and custom
              metrics. Run evaluations on datasets or apply them continuously to
              production traces.
            </li>
            <li>
              <Link
                href="/genai/prompt-registry"
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Prompt Registry
              </Link>
              : Version, compare, and iterate on prompt templates. Track which
              prompt versions are used by which application versions and measure
              the impact of prompt changes on quality.
            </li>
            <li>
              <Link
                href="/genai/ai-gateway"
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                AI Gateway
              </Link>
              : Route requests to any LLM provider through a unified interface.
              Manage API keys centrally, enforce rate limits, set fallback
              routes, and track usage across providers.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "tracing/production-tracing/"}
                style={{ color: "#007bff", fontWeight: "600" }}
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
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Human Feedback
              </Link>
              : Collect structured feedback from users and reviewers. Annotate
              traces with quality assessments, build evaluation datasets from
              real interactions, and close the feedback loop between production
              and development.
            </li>
          </ul>

          <h2 id="use-cases">Common Use Cases</h2>

          <p>
            AI platforms solve real-world problems across the AI development
            lifecycle:
          </p>

          <ul>
            <li>
              <strong>Debugging Agent Failures:</strong> When an autonomous
              agent takes the wrong action, MLflow's trace UI shows the complete
              reasoning chain—which tools were called, what context was
              retrieved, and where the decision went wrong—enabling rapid root
              cause analysis.
            </li>
            <li>
              <strong>Evaluating RAG Quality:</strong> For{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                RAG applications
              </Link>
              , automated scorers measure whether retrieved documents are
              relevant, whether the generated answer is faithful to the source
              material, and whether the response actually addresses the
              question.
            </li>
            <li>
              <strong>Iterating on Prompts:</strong> The{" "}
              <Link href="/genai/prompt-registry">prompt registry</Link> lets
              you version prompt templates, compare quality metrics across
              versions, and roll back to previous prompts if a change causes
              regressions.
            </li>
            <li>
              <strong>Multi-Provider Cost Optimization:</strong> The{" "}
              <Link href="/genai/ai-gateway">AI gateway</Link> combined with{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/token-usage-cost/"}>
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
              <Link href={MLFLOW_GENAI_DOCS_URL + "guides/responsible-ai/"}>
                responsible AI guardrails
              </Link>{" "}
              to detect and filter toxic, biased, or personally identifiable
              content in both inputs and outputs across your entire AI stack.
            </li>
          </ul>

          <h2 id="how-to-implement">
            How to Implement an AI Platform with MLflow
          </h2>

          <p>
            MLflow makes it straightforward to add platform-level capabilities
            to any LLM application or agent. Here are examples for common
            scenarios. Check out the{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/integrations/"}>
              MLflow tracing integrations documentation
            </Link>{" "}
            for examples with LangChain, LangGraph, LlamaIndex, CrewAI,
            Anthropic, Vercel AI SDK, and other frameworks.
          </p>

          <p style={{ marginTop: "32px", marginBottom: "0px" }}>
            <strong>Trace an LLM application (OpenAI)</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton
                code={`import mlflow
from openai import OpenAI

# Enable tracing with one line
mlflow.openai.autolog()

client = OpenAI()

# Every call is now traced automatically
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize this document."}],
)`}
              />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={`import mlflow
from openai import OpenAI

# Enable tracing with one line
mlflow.openai.autolog()

client = OpenAI()

# Every call is now traced automatically
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize this document."}],
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
            <strong>Trace a multi-step agent (LangGraph)</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton
                code={`import mlflow
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
result = app.invoke({"task": "Research competitor pricing"})`}
              />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={`import mlflow
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
result = app.invoke({"task": "Research competitor pricing"})`}
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
            <strong>Evaluate agent quality</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton
                code={`import mlflow
from mlflow.genai.scorers import Correctness, Hallucination

# Evaluate your agent across a dataset
results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_agent,
    scorers=[Correctness(), Hallucination()],
)

# View results in the MLflow UI
print(results.tables["eval_results"])`}
              />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={`import mlflow
from mlflow.genai.scorers import Correctness, Hallucination

# Evaluate your agent across a dataset
results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_agent,
    scorers=[Correctness(), Hallucination()],
)

# View results in the MLflow UI
print(results.tables["eval_results"])`}
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

          <div className="info-box">
            <p>
              <Link href="/genai" style={{ color: "#007bff" }}>
                <strong>MLflow</strong>
              </Link>{" "}
              is the largest open-source{" "}
              <strong>AI engineering platform</strong>, with over 30 million
              monthly downloads. Thousands of organizations use MLflow to build,
              evaluate, deploy, and monitor production-quality AI agents and LLM
              applications while controlling costs and managing access to models
              and data. Backed by the Linux Foundation and licensed under Apache
              2.0, MLflow provides a complete AI platform with no vendor
              lock-in.{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL}>Get started &#8594;</Link>
            </p>
          </div>

          <h2 id="open-source-vs-proprietary">
            Open Source vs. Proprietary AI Platforms
          </h2>

          <p>
            When choosing an AI platform, the decision between open source and
            proprietary SaaS tools has significant long-term implications for
            your team, infrastructure, and data ownership.
          </p>

          <p>
            <strong>
              Open Source (<Link href="/genai">MLflow</Link>):
            </strong>{" "}
            With MLflow, you maintain complete control over your platform
            infrastructure and data. Deploy on your own infrastructure or use
            managed versions on Databricks, AWS, or other platforms. There are
            no per-seat fees, no usage limits, and no vendor lock-in. Your
            telemetry data stays under your control, and you can customize the
            platform to your exact needs. MLflow integrates with any LLM
            provider and agent framework through OpenTelemetry-compatible
            tracing, supports 20+ framework integrations out of the box, and has
            an active community with over 20 million monthly downloads.
          </p>

          <p>
            <strong>Proprietary SaaS Platforms:</strong> Commercial AI platforms
            offer convenience but at the cost of flexibility and control. They
            typically charge per seat or per trace volume, which can become
            expensive at scale. Your data is sent to their servers, raising
            privacy and compliance concerns. You're locked into their ecosystem,
            making it difficult to switch providers or customize functionality.
            Most proprietary platforms only support a subset of LLM providers
            and frameworks, and their development roadmap is controlled by the
            vendor rather than the community.
          </p>

          <p>
            <strong>Why Teams Choose Open Source:</strong> Organizations
            building production AI applications increasingly choose MLflow
            because it offers enterprise-grade platform capabilities without
            compromising on data sovereignty, cost predictability, or
            flexibility. The Apache 2.0 license and Linux Foundation backing
            ensure MLflow remains truly open and community-driven, not
            controlled by a single vendor.
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
                    &#9660;
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
              <Link href={MLFLOW_GENAI_DOCS_URL}>
                MLflow GenAI Documentation
              </Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/quickstart/"}>
                Tracing Quickstart
              </Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                Evaluation and Monitoring Guide
              </Link>
            </li>
            <li>
              <Link href="/ai-observability">AI Observability Overview</Link>
            </li>
            <li>
              <Link href="/llm-evaluation">Agent Evaluation FAQ</Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/integrations/"}>
                Framework Integrations
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
