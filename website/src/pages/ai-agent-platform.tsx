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

const SEO_TITLE = "AI Agent Platform: What It Is & What You Need | MLflow";
const SEO_DESCRIPTION =
  "Learn what an AI agent platform is, what components it requires—authoring, orchestration, observability, evaluation, and governance—and how MLflow provides the open-source operational layer every agent platform needs.";

const faqs: {
  question: string;
  answer: React.ReactNode;
  answerText?: string;
}[] = [
  {
    question: "What is an AI agent platform?",
    answer:
      "An AI agent platform is an integrated environment for building, deploying, and operating autonomous AI agents in production. It typically spans agent authoring and orchestration (defining how agents reason, plan, and use tools), a runtime for executing long-running stateful agents, observability for tracing multi-step execution, evaluation for measuring agent quality, and governance for enforcing safety and compliance policies. No single tool covers every layer—most teams combine an agent framework for authoring with an operational platform for observability and evaluation.",
  },
  {
    question:
      "What is the difference between an agent framework and an agent platform?",
    answer: (
      <>
        An agent framework (like{" "}
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
            MLFLOW_GENAI_DOCS_URL + "tracing/integrations/listing/crewai.html"
          }
        >
          CrewAI
        </Link>
        , or AutoGen) provides the building blocks for constructing
        agents—tools, memory, planning loops, and orchestration patterns. An
        agent platform is broader: it includes the framework layer plus the
        operational layer for tracing, evaluation, monitoring, and governance.
        MLflow provides that operational layer, integrating with any agent
        framework so you can see what your agent did, measure whether it did
        well, and catch issues in production.
      </>
    ),
    answerText:
      "An agent framework (like LangGraph, CrewAI, or AutoGen) provides the building blocks for constructing agents—tools, memory, planning loops, and orchestration patterns. An agent platform is broader: it includes the framework layer plus the operational layer for tracing, evaluation, monitoring, and governance. MLflow provides that operational layer, integrating with any agent framework so you can see what your agent did, measure whether it did well, and catch issues in production.",
  },
  {
    question: "Where does MLflow fit in an AI agent platform?",
    answer: (
      <>
        MLflow is the open-source operational layer for your agent platform. It
        does not author or orchestrate agents—that's the job of frameworks like
        LangGraph or CrewAI. Instead, MLflow provides the capabilities you need
        once agents are running:{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
          end-to-end tracing
        </Link>{" "}
        to debug multi-step execution,{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          automated evaluation
        </Link>{" "}
        to measure quality, a{" "}
        <Link href="/genai/prompt-registry">prompt registry</Link> for managing
        instructions, and an <Link href="/genai/ai-gateway">AI gateway</Link>{" "}
        for unified access to LLM providers.
      </>
    ),
    answerText:
      "MLflow is the open-source operational layer for your agent platform. It does not author or orchestrate agents—that's the job of frameworks like LangGraph or CrewAI. Instead, MLflow provides the capabilities you need once agents are running: end-to-end tracing to debug multi-step execution, automated evaluation to measure quality, a prompt registry for managing instructions, and an AI gateway for unified access to LLM providers.",
  },
  {
    question: "What features should I look for in an AI agent platform?",
    answer:
      "A complete agent platform needs five layers: (1) an authoring framework for defining agent logic, tools, and memory; (2) a runtime for executing agents reliably; (3) observability with end-to-end tracing across multi-step workflows; (4) evaluation with LLM-as-a-judge scorers and dataset-based testing; and (5) governance for safety guardrails, access control, and compliance. MLflow covers layers 3–4 and parts of layer 5, complementing whatever authoring framework you choose.",
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
    question: "Can I use MLflow with any LLM provider or agent framework?",
    answer: (
      <>
        Yes. MLflow integrates with 20+ frameworks and providers. For agent
        frameworks, it supports{" "}
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
            MLFLOW_GENAI_DOCS_URL + "tracing/integrations/listing/crewai.html"
          }
        >
          CrewAI
        </Link>
        , AutoGen, and others. For LLM providers, its{" "}
        <Link href="/genai/ai-gateway">AI Gateway</Link> provides a unified
        interface for OpenAI, Anthropic, Google, Amazon Bedrock, and Azure
        OpenAI. You can switch providers without changing application code and
        manage API keys centrally.
      </>
    ),
    answerText:
      "Yes. MLflow integrates with 20+ frameworks and providers. For agent frameworks, it supports LangGraph, CrewAI, AutoGen, and others. For LLM providers, its AI Gateway provides a unified interface for OpenAI, Anthropic, Google, Amazon Bedrock, and Azure OpenAI. You can switch providers without changing application code and manage API keys centrally.",
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
    question: "Is MLflow free to use?",
    answer: (
      <>
        MLflow is completely free and open-source under the Apache 2.0 license.
        You can self-host it or use{" "}
        <Link href="https://databricks.com/product/managed-mlflow">
          managed MLflow on Databricks
        </Link>{" "}
        for a hosted experience. The open-source version includes all
        operational capabilities: tracing, evaluation, prompt registry, AI
        gateway, and experiment tracking.
      </>
    ),
    answerText:
      "MLflow is completely free and open-source under the Apache 2.0 license. You can self-host it or use managed MLflow on Databricks for a hosted experience. The open-source version includes all operational capabilities: tracing, evaluation, prompt registry, AI gateway, and experiment tracking.",
  },
  {
    question: "How do I get started with MLflow for my agent platform?",
    answer: (
      <>
        Getting started takes three steps: install MLflow with{" "}
        <code>pip install mlflow</code>, enable tracing for your agent framework
        (e.g., <code>mlflow.langgraph.autolog()</code>), and open the MLflow UI
        to see your traces. See the{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/quickstart/"}>
          quickstart guide
        </Link>{" "}
        for a complete walkthrough.
      </>
    ),
    answerText:
      "Getting started takes three steps: install MLflow with pip install mlflow, enable tracing for your agent framework (e.g., mlflow.langgraph.autolog()), and open the MLflow UI to see your traces. See the quickstart guide for a complete walkthrough.",
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
    "Open-source operational platform for tracing, evaluating, and monitoring AI agents and LLM applications.",
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
          <h1>AI Agent Platform</h1>

          <p>
            An AI agent platform is the integrated stack for building,
            deploying, and operating autonomous AI agents in production. Agents
            go beyond single LLM calls — they reason across multiple steps, call
            tools and APIs, maintain state, and make decisions autonomously. A
            complete agent platform spans several layers: an{" "}
            <strong>authoring framework</strong> for defining agent logic (like{" "}
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
            , or AutoGen), a <strong>runtime</strong> for executing agents
            reliably, and an <strong>operational layer</strong> for
            observability, evaluation, and governance.
          </p>

          <p>
            No single tool covers every layer. Most production teams combine an
            agent framework for authoring with a separate operational platform
            for tracing, evaluation, and monitoring.{" "}
            <Link href="/genai">MLflow</Link> is the leading open-source
            operational layer for agent platforms — it provides{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
              end-to-end tracing
            </Link>{" "}
            to debug multi-step agent execution,{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
              automated evaluation
            </Link>{" "}
            to measure agent quality, a{" "}
            <Link href="/genai/prompt-registry">prompt registry</Link> for
            managing instructions, and an{" "}
            <Link href="/genai/ai-gateway">AI gateway</Link> for unified access
            to LLM providers. MLflow integrates with whatever agent framework
            you choose, giving you full visibility without locking you into a
            specific authoring tool.
          </p>

          <h2 id="what-makes-up-agent-platform">
            What Makes Up an AI Agent Platform
          </h2>

          <p>
            An AI agent platform is not a single product — it is a stack of
            complementary capabilities. Understanding each layer helps you
            choose the right tools for your team:
          </p>

          <div className="grid-2">
            <div className="card">
              <h3>Agent Authoring &amp; Orchestration</h3>
              <p>
                <strong>What it does:</strong> Defines how agents reason, plan,
                use tools, and coordinate with other agents. This layer handles
                memory, state management, and multi-step workflow logic.
              </p>
              <p>
                <strong>Examples:</strong> LangGraph, CrewAI, AutoGen, Amazon
                Bedrock Agents, Microsoft Copilot Studio.
              </p>
            </div>

            <div className="card">
              <h3>Observability &amp; Tracing</h3>
              <p>
                <strong>What it does:</strong> Captures the full execution graph
                — every LLM call, tool invocation, retrieval step, and decision
                branch — so you can debug failures and understand agent
                behavior.
              </p>
              <p>
                <strong>MLflow provides this:</strong>{" "}
                <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
                  OpenTelemetry-compatible tracing
                </Link>{" "}
                with one-line integrations for 20+ frameworks.
              </p>
            </div>

            <div className="card">
              <h3>Evaluation &amp; Quality</h3>
              <p>
                <strong>What it does:</strong> Measures agent quality using
                automated scorers — correctness, hallucination, relevance, and
                custom metrics — across datasets and production traces.
              </p>
              <p>
                <strong>MLflow provides this:</strong>{" "}
                <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                  LLM-as-a-judge evaluation
                </Link>{" "}
                that runs on datasets or continuously on live traffic.
              </p>
            </div>

            <div className="card">
              <h3>Governance &amp; Safety</h3>
              <p>
                <strong>What it does:</strong> Enforces guardrails, access
                policies, and compliance rules. Ensures agents operate within
                organizational boundaries and don't produce harmful outputs.
              </p>
              <p>
                <strong>MLflow contributes:</strong>{" "}
                <Link href={MLFLOW_GENAI_DOCS_URL + "guides/responsible-ai/"}>
                  Responsible AI scorers
                </Link>
                , the <Link href="/genai/ai-gateway">AI Gateway</Link> for
                centralized key management and rate limiting, and full trace
                auditability.
              </p>
            </div>
          </div>

          <h2 id="why-operational-layer">Why You Need an Operational Layer</h2>

          <p>
            Authoring an agent is straightforward. Operating it in production is
            not. Unlike traditional software, agents are{" "}
            <strong>non-deterministic</strong> — the same input can produce
            different outputs depending on model state, retrieved context, and
            multi-step reasoning. This creates challenges that agent authoring
            frameworks alone don't solve:
          </p>

          <ul>
            <li>
              <strong>Debugging is opaque:</strong> Agent failures can happen at
              any step — retrieval, reasoning, tool execution, prompt
              construction. Without{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>tracing</Link>,
              you can't see what went wrong or why.
            </li>
            <li>
              <strong>Quality is hard to measure:</strong> Free-form language
              output can't be validated with unit tests. You need{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                LLM-as-a-judge evaluation
              </Link>{" "}
              to assess correctness, hallucination, and relevance at scale.
            </li>
            <li>
              <strong>Prompts drift silently:</strong> A small change to a
              system prompt can alter agent behavior across thousands of
              interactions. A{" "}
              <Link href="/genai/prompt-registry">prompt registry</Link>{" "}
              versions and tracks the impact of changes on quality.
            </li>
            <li>
              <strong>Provider management grows complex:</strong> Routing
              requests across OpenAI, Anthropic, Google, and Bedrock while
              managing API keys, rate limits, and fallback logic creates
              compounding overhead. An{" "}
              <Link href="/genai/ai-gateway">AI gateway</Link> provides a
              unified interface.
            </li>
          </ul>

          <h2 id="what-mlflow-provides">What MLflow Provides</h2>

          <p>
            MLflow is not an agent authoring tool — it does not define how your
            agent reasons, plans, or calls tools. Instead, it provides the
            operational infrastructure that every agent platform needs,
            regardless of which framework you use to build your agents:
          </p>

          <ul>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "tracing/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Tracing
              </Link>
              : Capture complete execution traces including LLM calls, tool
              invocations, retrievals, and agent decisions.
              OpenTelemetry-compatible with one-line auto-instrumentation for
              LangGraph, CrewAI, LangChain, OpenAI, Anthropic, and 20+ other
              frameworks.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Evaluation
              </Link>
              : Measure agent quality at scale using{" "}
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
              prompt versions are used by which agent versions and measure the
              impact of prompt changes on quality.
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
              real interactions, and close the feedback loop.
            </li>
          </ul>

          <h2 id="how-to-add-mlflow">
            How to Add MLflow to Your Agent Platform
          </h2>

          <p>
            MLflow integrates with your existing agent framework in minutes —
            you don't need to change how you build agents. Here are examples
            showing how to add tracing and evaluation to common setups. See the{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/integrations/"}>
              integrations documentation
            </Link>{" "}
            for LangChain, LangGraph, CrewAI, LlamaIndex, Anthropic, Vercel AI
            SDK, and more.
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
              is the largest open-source operational platform for AI, with over
              30 million monthly downloads. Thousands of organizations use
              MLflow to trace, evaluate, and monitor their AI agents and LLM
              applications — regardless of which authoring framework or LLM
              provider they use. Backed by the Linux Foundation and licensed
              under Apache 2.0, MLflow complements your agent platform with no
              vendor lock-in.{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL}>Get started &#8594;</Link>
            </p>
          </div>

          <h2 id="open-source-vs-proprietary">
            Open Source vs. Proprietary Operational Platforms
          </h2>

          <p>
            When choosing the operational layer for your agent platform, the
            decision between open source and proprietary SaaS tools has
            long-term implications for data ownership, cost, and flexibility.
          </p>

          <p>
            <strong>
              Open Source (<Link href="/genai">MLflow</Link>):
            </strong>{" "}
            You maintain complete control over your telemetry data and platform
            infrastructure. Deploy on your own infrastructure or use managed
            versions on Databricks or other clouds. No per-seat fees, no usage
            limits, no vendor lock-in. MLflow integrates with any agent
            framework and LLM provider through OpenTelemetry-compatible tracing,
            supports 20+ integrations out of the box, and has an active
            community with over 20 million monthly downloads.
          </p>

          <p>
            <strong>Proprietary SaaS Platforms:</strong> Commercial
            observability and evaluation platforms offer convenience but at the
            cost of flexibility and control. They typically charge per seat or
            per trace volume, which grows expensive at scale. Your trace data is
            sent to their servers, raising privacy and compliance concerns.
            You're locked into their ecosystem, and their development roadmap is
            controlled by the vendor rather than the community.
          </p>

          <p>
            <strong>Why Teams Choose Open Source:</strong> Organizations
            building production agents increasingly choose MLflow because it
            provides enterprise-grade observability and evaluation without
            compromising on data sovereignty, cost predictability, or
            flexibility. The Apache 2.0 license and Linux Foundation backing
            ensure MLflow remains truly open and community-driven.
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
