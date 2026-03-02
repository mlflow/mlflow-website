import { useState } from "react";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../components/Header/Header";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import ObservabilityHero from "@site/static/img/GenAI_observability/GenAI_observability_hero.png";
import { CopyButton } from "../components/CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";

const SEO_TITLE =
  "What is LLMOps? LLM Operations Guide | MLflow Agent Platform";
const SEO_DESCRIPTION =
  "Learn LLMOps (LLM Operations) and AgentOps with MLflow, the open-source platform for tracing, evaluating, and deploying large language model applications and agents.";

const faqs: {
  question: string;
  answer: React.ReactNode;
  answerText?: string;
}[] = [
  {
    question: "What is LLMOps?",
    answer:
      "LLMOps (Large Language Model Operations) is the set of practices, tools, and workflows for building, deploying, monitoring, and maintaining LLM-powered applications in production. It covers the full lifecycle from prompt engineering and evaluation through deployment, tracing, and continuous improvement.",
  },
  {
    question: "How is LLMOps different from MLOps?",
    answer:
      "MLOps focuses on training, versioning, and deploying traditional machine learning models. LLMOps deals with challenges unique to LLMs: prompt management, non-deterministic outputs, token cost optimization, multi-step agent orchestration, retrieval-augmented generation, and evaluation with LLM judges rather than static metrics.",
  },
  {
    question: "What are the key components of an LLMOps platform?",
    answer: (
      <>
        An LLMOps platform typically includes:{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>tracing</Link>{" "}
        (execution capture for debugging),{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "llm-evaluate/"}>evaluation</Link>{" "}
        (automated quality assessment with LLM judges),{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "prompts/"}>prompt management</Link>{" "}
        (versioning and registry), deployment infrastructure, and{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          production monitoring
        </Link>
        .
      </>
    ),
    answerText:
      "An LLMOps platform typically includes: tracing (execution capture for debugging), evaluation (automated quality assessment with LLM judges), prompt management (versioning and registry), deployment infrastructure, and production monitoring.",
  },
  {
    question: "Do I need LLMOps for my LLM application?",
    answer:
      "Yes, if you're moving beyond prototypes. LLMOps practices help you ship reliable LLM applications by providing visibility into model behavior, systematic evaluation before deployments, prompt version control, cost tracking, and production monitoring. Without LLMOps, teams struggle with debugging non-deterministic outputs and managing quality at scale.",
  },
  {
    question: "What is the LLMOps lifecycle?",
    answer:
      "The LLMOps lifecycle covers: (1) Development, including prompt engineering, retrieval pipeline design, and agent authoring; (2) Evaluation with LLM judges and human review; (3) Deployment of models and agents with versioned prompts; (4) Monitoring production requests, tracking quality scores, and detecting regressions; (5) Iteration using production insights to improve prompts, retrieval, and agent logic.",
  },
  {
    question: "What is LLM Ops vs LLMOps?",
    answer:
      "LLM Ops and LLMOps refer to the same discipline: operations for large language model applications. 'LLMOps' is the more common spelling, following the convention of MLOps and DevOps. Both terms describe the practices and tools needed to build, deploy, and maintain LLM-powered applications in production.",
  },
  {
    question: "What is AgentOps?",
    answer:
      "AgentOps extends LLMOps to multi-step agentic systems. While LLMOps covers single LLM calls and simple applications, AgentOps addresses the unique challenges of autonomous agents: tracing multi-step reasoning, debugging tool call sequences, evaluating agent decision-making, and monitoring complex workflows. AgentOps includes all LLMOps capabilities plus agent-specific observability, evaluation, and optimization.",
  },
  {
    question: "What is the best LLMOps platform?",
    answer:
      "The best LLMOps platform depends on your needs. MLflow is the leading open-source option, offering complete tracing, evaluation, prompt management, and monitoring without vendor lock-in. MLflow supports any LLM provider (OpenAI, Anthropic, Bedrock, etc.) and agent framework (LangChain, LangGraph, LlamaIndex, CrewAI, etc.), and is backed by the Linux Foundation with over 30 million monthly downloads.",
  },
  {
    question: "How does MLflow support LLMOps?",
    answer: (
      <>
        MLflow provides a complete LLMOps stack:{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>automatic tracing</Link>{" "}
        for debugging and monitoring,{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "llm-evaluate/"}>
          evaluation with LLM judges
        </Link>{" "}
        for quality assurance,{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "prompts/"}>a prompt registry</Link>{" "}
        for version control, and{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          production monitoring
        </Link>{" "}
        for ongoing quality tracking. All features work with any LLM provider
        and agent framework.
      </>
    ),
    answerText:
      "MLflow provides a complete LLMOps stack: automatic tracing for debugging and monitoring, evaluation with LLM judges for quality assurance, a prompt registry for version control, and production monitoring for ongoing quality tracking. All features work with any LLM provider and agent framework.",
  },
  {
    question: "How does LLMOps handle prompt management?",
    answer:
      "LLMOps platforms like MLflow provide prompt registries that version-control prompt templates, track which prompts are used in production, enable A/B testing of prompt variants, and allow rollbacks when quality degrades. This brings the same rigor to prompt engineering that Git brings to source code.",
  },
  {
    question: "Is MLflow free for LLMOps?",
    answer:
      "Yes. MLflow is 100% open source under the Apache 2.0 license, backed by the Linux Foundation. You can use all LLMOps features (tracing, evaluation, prompt management, monitoring) for free, including in commercial applications. There are no per-seat fees, no usage limits, and no vendor lock-in.",
  },
  {
    question: "How do I get started with LLMOps?",
    answer: (
      <>
        Getting started with LLMOps using MLflow takes minutes. Install MLflow,
        enable{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/quickstart/"}>
          automatic tracing
        </Link>{" "}
        with a single line of code, and every LLM call is captured with full
        context. From there, add{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "llm-evaluate/"}>evaluations</Link>{" "}
        to assess quality and{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "prompts/"}>
          register your prompts
        </Link>{" "}
        for version control.
      </>
    ),
    answerText:
      "Getting started with LLMOps using MLflow takes minutes. Install MLflow, enable automatic tracing with a single line of code, and every LLM call is captured with full context. From there, add evaluations to assess quality and register your prompts for version control.",
  },
  {
    question: "What's the difference between LLMOps and AI observability?",
    answer: (
      <>
        <Link href="/faq/ai-observability">AI observability</Link> is a subset
        of LLMOps focused on monitoring and understanding AI system behavior
        (tracing, metrics, evaluation). LLMOps is broader, also encompassing
        prompt management, deployment workflows, CI/CD for LLM applications, and
        the full operational lifecycle from development through production.
      </>
    ),
    answerText:
      "AI observability is a subset of LLMOps focused on monitoring and understanding AI system behavior (tracing, metrics, evaluation). LLMOps is broader, also encompassing prompt management, deployment workflows, CI/CD for LLM applications, and the full operational lifecycle from development through production.",
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
    "Open-source LLMOps platform for tracing, evaluating, and deploying LLM applications and agents.",
  url: "https://mlflow.org",
  license: "https://www.apache.org/licenses/LICENSE-2.0",
};

const TRACING_EVAL_CODE = `import mlflow
from openai import OpenAI

# Enable automatic tracing for OpenAI
mlflow.openai.autolog()

# Every LLM call is now traced with full context
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Summarize MLflow"}],
)

# Evaluate traced outputs with LLM judges
import mlflow.genai
results = mlflow.genai.evaluate(
    data=mlflow.search_traces(experiment_ids=["1"]),
    scorers=[mlflow.genai.scorers.Relevance()],
)`;

const PROMPT_MGMT_CODE = `import mlflow
from mlflow.genai.prompts import register_prompt

# Register a versioned prompt template
prompt = register_prompt(
    name="summarizer",
    template="Summarize the following in {{style}} style:\\n{{text}}",
)

# Load the latest version in your application
prompt = mlflow.genai.prompts.get_prompt("summarizer")
formatted = prompt.format(style="concise", text="MLflow is...")`;

export default function LLMOps() {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://mlflow.org/llmops" />
        <link rel="canonical" href="https://mlflow.org/llmops" />
        <script type="application/ld+json">{JSON.stringify(faqJsonLd)}</script>
        <script type="application/ld+json">
          {JSON.stringify(softwareJsonLd)}
        </script>
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
          <h1>What is LLMOps?</h1>

          <p>
            LLMOps (LLM Operations) is the discipline of building, deploying,
            monitoring, and maintaining large language model applications in
            production. It encompasses the tools, practices, and workflows that
            teams need to move LLM-powered applications from prototype to
            production, including{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>tracing</Link>,{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "llm-evaluate/"}>
              evaluation
            </Link>
            ,{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "prompts/"}>
              prompt management
            </Link>
            , <Link href="/genai/ai-gateway">AI Gateways</Link> for governed
            model access, and{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
              production monitoring
            </Link>
            . For multi-step agentic systems, this is known as{" "}
            <a href="#agentops">AgentOps</a>.
          </p>

          <p>
            As LLM applications evolve from single-turn chatbots to multi-step{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "agent-evaluation/"}>
              agents
            </Link>{" "}
            and RAG systems, the operational challenges grow significantly. LLMs
            are non-deterministic, expensive, and difficult to evaluate with
            traditional software testing. LLMOps gives teams the tools to manage
            these challenges, bringing the same structure to LLM applications
            that DevOps and MLOps brought to software and machine learning.
          </p>

          <p>
            LLMOps platforms provide the tooling to address these challenges:
            tracing for debugging, evaluation with LLM judges for quality
            assurance, prompt registries for version control, AI gateways for
            governed model access, and production monitoring for catching
            regressions.
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
                <a href="#why-llmops-matters">Why LLMOps Matters</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#from-mlops-to-llmops">From MLOps to LLMOps</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#agentops">AgentOps</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#key-components">Key Components</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#llmops-with-mlflow">LLMOps with MLflow</a>
              </li>
              <li style={{ marginBottom: "0" }}>
                <a href="#faq">FAQ</a>
              </li>
            </ul>
          </div>

          <h2 id="why-llmops-matters">Why LLMOps Matters</h2>

          <p>
            LLM applications introduce unique operational challenges that
            traditional DevOps and MLOps can't address:
          </p>

          <div className="grid-2">
            <div className="card">
              <h3>Non-Deterministic Outputs</h3>
              <p>
                <strong>Problem:</strong> The same prompt can produce different
                outputs across runs, making it impossible to test LLM
                applications with traditional assertions.
              </p>
              <p>
                <strong>Solution:</strong> LLMOps uses automated evaluation with
                LLM judges to assess quality at scale, replacing brittle
                exact-match tests with semantic quality scoring.
              </p>
            </div>

            <div className="card">
              <h3>Prompt Fragility</h3>
              <p>
                <strong>Problem:</strong> Small changes to prompts can
                dramatically alter output quality, and there's no built-in
                version control for prompt templates.
              </p>
              <p>
                <strong>Solution:</strong> Prompt registries provide version
                control, A/B testing, and rollback capabilities for prompt
                templates, bringing Git-like rigor to prompt engineering.
              </p>
            </div>

            <div className="card">
              <h3>Governance and Cost Controls</h3>
              <p>
                <strong>Problem:</strong> Teams lack centralized control over
                which models are used, how they're accessed, and what rate
                limits apply. Token costs can also spiral with multi-step agents
                making many LLM calls per request.
              </p>
              <p>
                <strong>Solution:</strong>{" "}
                <Link href="/genai/ai-gateway">AI Gateways</Link> provide a
                single control plane for model access with rate limiting,
                authentication, fallback routing, and cost tracking.{" "}
                <Link href="/faq/llm-tracing">Tracing</Link> captures token
                usage and latency per span, making it easy to find expensive
                operations and debug unexpected behavior.
              </p>
            </div>

            <div className="card">
              <h3>Complex Debugging</h3>
              <p>
                <strong>Problem:</strong> When agents fail, it's nearly
                impossible to understand why without visibility into every
                reasoning step, tool call, and retrieval.
              </p>
              <p>
                <strong>Solution:</strong>{" "}
                <Link href="/faq/llm-tracing">End-to-end tracing</Link> makes
                every step visible and debuggable, from initial request through
                tool calls to final response.
              </p>
            </div>
          </div>

          <h2 id="from-mlops-to-llmops">From MLOps to LLMOps</h2>

          <p>
            Traditional MLOps focuses on training, validating, and deploying
            machine learning models. LLMOps addresses a different set of
            problems. LLM applications are driven by prompts rather than
            training data, their outputs are non-deterministic, and quality
            can't be measured with simple accuracy metrics. Agents add even more
            complexity: multi-step reasoning, tool calls, and autonomous
            decision-making all need to be traced, evaluated, and governed.
          </p>

          <p>
            LLMOps is closely related to <Link href="/aiops">AIOps</Link> (the
            broader discipline of running all AI applications in production) and{" "}
            <Link href="/faq/ai-observability">AI observability</Link> (the
            monitoring and debugging subset). LLMOps specifically targets
            LLM-powered applications, while AIOps also covers traditional ML
            experiment tracking and model management.
          </p>

          <h2 id="agentops">AgentOps</h2>

          <p>
            AgentOps extends LLMOps to multi-step agentic systems. While LLMOps
            covers single LLM calls and simple applications, AgentOps addresses
            the unique challenges of autonomous agents: tracing multi-step
            reasoning chains, debugging complex tool call sequences, evaluating
            agent decision-making, and monitoring workflows where agents make
            dozens of LLM calls per request.
          </p>

          <p>
            AgentOps includes all LLMOps capabilities (tracing, evaluation,
            prompt management) plus agent-specific tooling:{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
              execution graph visualization
            </Link>{" "}
            to debug reasoning loops,{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "agent-evaluation/"}>
              agent evaluation
            </Link>{" "}
            with multi-turn testing, tool call correctness scoring, and
            optimization of agent workflows to reduce token costs and latency.
            MLflow provides complete AgentOps support for all agent frameworks,
            including LangGraph, CrewAI, AutoGen, and custom agent
            implementations.
          </p>

          <h2 id="key-components">Key Components of LLMOps</h2>

          <p>A production LLMOps workflow combines several capabilities:</p>

          <ul>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "tracing/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Tracing
              </Link>
              : Record every step of LLM and agent execution (prompts,
              completions, tool calls, retrieval results, token usage, and
              latency) for debugging and production monitoring.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "llm-evaluate/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Evaluation
              </Link>
              : Assess output quality using{" "}
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "llm-evaluate/llm-as-judge/"}
                style={{ color: "#007bff" }}
              >
                LLM judges
              </Link>
              , custom scorers, and human feedback before and after deployment.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "prompts/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Prompt Management
              </Link>
              : Version-control prompt templates, track which versions are in
              production, and enable safe rollbacks when quality degrades.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Production Monitoring
              </Link>
              : Track quality scores, error rates, costs, and latency over time
              with LLM judges to catch regressions early.
            </li>
            <li>
              <Link
                href="/genai/ai-gateway"
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                AI Gateway
              </Link>
              : Route requests across LLM providers (OpenAI, Anthropic, Bedrock,
              etc.) through a single endpoint with unified authentication, rate
              limiting, and fallback routing.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "guides/responsible-ai/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Governance & Safety
              </Link>
              : Maintain audit trails, enforce PII policies, and apply content
              guardrails across your LLM applications.
            </li>
          </ul>

          <h2 id="llmops-with-mlflow">LLMOps with MLflow</h2>

          <p>
            <Link href="/genai">MLflow</Link> is the only open-source,
            production-grade, end-to-end LLMOps platform. It supports any LLM,
            framework, and programming language, and is backed by the Linux
            Foundation. MLflow provides solutions for every layer of the LLMOps
            stack:
          </p>

          <ul>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "tracing/"}
                style={{ color: "#0194e2", fontWeight: 700 }}
              >
                MLflow Tracing
              </Link>{" "}
              — Auto-instrument any LLM framework in one line of code. Captures
              prompts, completions, tool calls, token usage, and latency for
              every request.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "llm-evaluate/"}
                style={{ color: "#0194e2", fontWeight: 700 }}
              >
                MLflow LLM Evaluation
              </Link>{" "}
              — Score outputs with{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "llm-evaluate/llm-as-judge/"}>
                LLM judges
              </Link>
              , custom scorers, and{" "}
              <Link href="/genai/human-feedback">human feedback</Link>. Built-in
              judges for correctness, safety, groundedness, and RAG quality.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "prompts/"}
                style={{ color: "#0194e2", fontWeight: 700 }}
              >
                MLflow Prompt Registry
              </Link>{" "}
              — Version control, diff tracking, and aliases (dev, staging,
              production) for prompt templates. Edit in the UI without code
              changes.
            </li>
            <li>
              <Link
                href="/genai/ai-gateway"
                style={{ color: "#0194e2", fontWeight: 700 }}
              >
                MLflow AI Gateway
              </Link>{" "}
              — Single control plane for model access across providers with
              governance, rate limiting, authentication, fallback routing, and
              cost tracking.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}
                style={{ color: "#0194e2", fontWeight: 700 }}
              >
                MLflow Production Monitoring
              </Link>{" "}
              — Run LLM judges continuously against production traces to catch
              quality regressions before users report them.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "agent-server/"}
                style={{ color: "#0194e2", fontWeight: 700 }}
              >
                MLflow Agent Server
              </Link>{" "}
              — Deploy agents to production with built-in tracing, streaming,
              and request validation.
            </li>
          </ul>

          <div style={{ margin: "32px 0", textAlign: "center" }}>
            <img
              src={ObservabilityHero}
              alt="MLflow UI showing traced LLM calls with prompts, responses, and metadata"
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
              MLflow captures traces for every LLM call with full execution
              context
            </p>
          </div>

          <div
            style={{
              margin: "32px 0",
              borderRadius: "8px",
              overflow: "hidden",
              boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
            }}
          >
            <video
              autoPlay
              muted
              loop
              playsInline
              style={{ width: "100%", display: "block" }}
            >
              <source
                src={
                  require("@site/static/img/releases/3.10.0/gateway-usage.mp4")
                    .default
                }
                type="video/mp4"
              />
            </video>
            <p
              style={{
                marginTop: "12px",
                fontSize: "14px",
                color: "#6b7280",
                fontStyle: "italic",
                textAlign: "center",
              }}
            >
              AI Gateway: governed model access with usage tracking across
              providers
            </p>
          </div>

          <div className="info-box">
            <p>
              <Link href="/genai" style={{ color: "#007bff" }}>
                <strong>MLflow</strong>
              </Link>{" "}
              is the largest open-source AI platform, with over 30 million
              monthly downloads. Backed by the Linux Foundation and licensed
              under Apache 2.0, it provides a complete LLMOps stack with no
              vendor lock-in.{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL}>Get started →</Link>
            </p>
          </div>

          <h2>Open Source vs. Proprietary LLMOps</h2>

          <p>
            When choosing an LLMOps platform, the decision between open source
            and proprietary SaaS tools has significant long-term implications
            for your team, infrastructure, and data ownership.
          </p>

          <p>
            <strong>
              Open Source (<Link href="/genai">MLflow</Link>):
            </strong>{" "}
            With MLflow, you maintain complete control over your LLMOps
            infrastructure and data. Deploy on your own infrastructure or use
            managed versions on Databricks, AWS, or other platforms. There are
            no per-seat fees, no usage limits, and no vendor lock-in. MLflow
            integrates with any LLM provider and agent framework through
            OpenTelemetry-compatible tracing.
          </p>

          <p>
            <strong>Proprietary SaaS Tools:</strong> Commercial LLMOps platforms
            offer convenience but at the cost of flexibility and control. They
            typically charge per seat or per trace volume, which can become
            expensive at scale. Your data is sent to their servers, raising
            privacy and compliance concerns. You're locked into their ecosystem,
            making it difficult to switch providers or customize functionality.
          </p>

          <p>
            <strong>Why Teams Choose Open Source:</strong> Organizations
            building production LLM applications increasingly choose MLflow
            because it offers production-ready LLMOps without giving up control
            of their data, cost predictability, or flexibility. The Apache 2.0
            license and Linux Foundation backing ensure MLflow remains truly
            open and community-driven.
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
                LLM Tracing Documentation
              </Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "llm-evaluate/"}>
                LLM Evaluation Guide
              </Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "prompts/"}>
                Prompt Registry Documentation
              </Link>
            </li>
            <li>
              <Link href="/faq/ai-observability">AI Observability Guide</Link>
            </li>
            <li>
              <Link href="/faq/llm-tracing">LLM Tracing Guide</Link>
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
