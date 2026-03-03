import { useState } from "react";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../components/Header/Header";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import ObservabilityHero from "@site/static/img/GenAI_observability/GenAI_observability_hero.png";
import { CopyButton } from "../components/CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";

const SEO_TITLE = "AI Observability for LLMs & Agents | MLflow AI Platform";
const SEO_DESCRIPTION =
  "Learn AI observability, LLM observability, and agent observability with MLflow—the comprehensive, open-source agent engineering and ops platform.";

const faqs: {
  question: string;
  answer: React.ReactNode;
  answerText?: string;
}[] = [
  {
    question: "What is AI observability?",
    answer:
      "AI observability is the practice of collecting, analyzing, and correlating telemetry data (traces, metrics, evaluations, and logs) across AI systems to understand how they behave in development and production. It goes beyond traditional software monitoring by providing deep visibility into the internal state of non-deterministic AI applications like agents, LLMs, and RAG pipelines.",
  },
  {
    question:
      "How is AI observability different from traditional software monitoring?",
    answer:
      "Traditional monitoring tracks deterministic metrics like uptime, CPU, and error rates. AI observability must also capture the quality and correctness of free-form language outputs, multi-step agent reasoning, tool call chains, retrieval accuracy, and token costs (none of which exist in traditional software systems).",
  },
  {
    question: "What are the key components of an AI observability platform?",
    answer: (
      <>
        A comprehensive AI observability platform includes:{" "}
        <Link href="/llm-tracing">tracing</Link> (end-to-end execution capture),
        evaluation (automated quality assessment with LLM judges), monitoring
        (production metrics and drift detection), cost and latency tracking,
        human feedback collection, and governance (audit trails and policy
        enforcement).
      </>
    ),
    answerText:
      "A comprehensive AI observability platform includes: tracing (end-to-end execution capture), evaluation (automated quality assessment with LLM judges), monitoring (production metrics and drift detection), cost and latency tracking, human feedback collection, and governance (audit trails and policy enforcement).",
  },
  {
    question: "Do I need AI observability for my agent or LLM application?",
    answer:
      "Yes, if you're building production AI applications. AI observability helps you detect hallucinations, track costs, debug complex agent behaviors, monitor quality over time, and maintain compliance. Without observability, you're flying blind—unable to understand why your AI system produces certain outputs or how to improve it.",
  },
  {
    question:
      "What's the difference between LLM observability and AI observability?",
    answer:
      "LLM observability focuses specifically on large language model calls (prompts, completions, tokens, latency). AI observability is broader, encompassing LLMs plus agents (multi-step reasoning, tool calls), RAG systems (retrieval, chunking, embeddings), and other AI components. MLflow provides comprehensive AI observability that covers all these use cases.",
  },
  {
    question: "What is agent observability?",
    answer:
      "Agent observability extends LLM observability to multi-step agentic systems. It traces how agents reason, which tools they call, how they handle errors, and how they chain multiple LLM calls together. MLflow automatically captures agent execution graphs, making it easy to debug when agents get stuck in loops, make incorrect tool choices, or produce unexpected results.",
  },
  {
    question: "What is the best AI observability tool?",
    answer:
      "The best AI observability tool depends on your needs. MLflow is the leading open-source option, with over 30 million monthly downloads. Thousands of organizations, developers, and research teams use MLflow each day to build and deploy production-grade agents and LLM applications. It offers complete tracing, evaluation, and monitoring without vendor lock-in. MLflow supports any agent framework (LangChain, LangGraph, LlamaIndex, CrewAI, AutoGen, etc.), any LLM provider (OpenAI, Anthropic, Bedrock, etc.), is fully OpenTelemetry compatible, and gives you full control over your data.",
  },
  {
    question: "What agent frameworks and LLMs does MLflow support?",
    answer:
      "MLflow supports any LLM, agent authoring framework, and programming language. This includes popular LLM providers like OpenAI, Anthropic (Claude), AWS Bedrock, Google Gemini, Azure OpenAI, Mistral, Cohere, AI21, Together AI, Anyscale, vLLM, and Ollama. For agent frameworks, MLflow integrates with LangChain, LangGraph, LlamaIndex, CrewAI, AutoGen, DSPy, Haystack, Semantic Kernel, and many more. MLflow SDKs are available for Python, JavaScript, and TypeScript.",
  },
  {
    question: "How does MLflow compare to other AI observability tools?",
    answer:
      "Unlike proprietary tools that lock you into a vendor's ecosystem, MLflow provides a complete, open-source observability stack with no vendor lock-in. It supports any LLM or agent authoring framework, is OpenTelemetry compatible, and is trusted by thousands of organizations worldwide. MLflow is also available on Databricks, AWS, and other platforms.",
  },
  {
    question: "Does MLflow support OpenTelemetry?",
    answer:
      "Yes. MLflow's tracing is fully compatible with OpenTelemetry, so you can export traces to any OpenTelemetry-compatible backend. This gives you total ownership and portability of your AI telemetry data without vendor lock-in.",
  },
  {
    question: "Is MLflow free for AI observability?",
    answer:
      "Yes. MLflow is 100% open source under the Apache 2.0 license, backed by the Linux Foundation. You can use all of its observability features (tracing, evaluation, monitoring, and more) for free, including in commercial applications.",
  },
  {
    question: "How do I get started with AI observability?",
    answer: (
      <>
        Getting started with MLflow AI observability takes just one line of
        code. Install MLflow, call{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/quickstart/"}>
          mlflow.openai.autolog()
        </Link>{" "}
        (or the equivalent for your framework), and every LLM call is
        automatically traced. You can then view traces in the MLflow UI, run
        evaluations with LLM judges, and monitor production metrics. See the{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
          MLflow tracing documentation
        </Link>{" "}
        for framework-specific examples.
      </>
    ),
    answerText:
      "Getting started with MLflow AI observability takes just one line of code. Install MLflow, call mlflow.openai.autolog() (or the equivalent for your framework), and every LLM call is automatically traced. You can then view traces in the MLflow UI, run evaluations with LLM judges, and monitor production metrics. See the MLflow tracing documentation for framework-specific examples.",
  },
  {
    question:
      "Is it easy to integrate MLflow with my existing agent or LLM application?",
    answer:
      "Yes. MLflow integrates seamlessly with your existing stack. It supports OpenTelemetry for exporting traces to any compatible backend, works with any LLM provider (OpenAI, Anthropic, Bedrock, etc.), and integrates with popular frameworks like LangChain, LangGraph, and LlamaIndex. You can also self-host MLflow or use managed versions on Databricks, AWS, and other platforms.",
  },
  {
    question:
      "How does MLflow AI Observability help with compliance, governance, and policy enforcement?",
    answer: (
      <>
        MLflow provides multiple layers for governance and compliance.{" "}
        <Link href="/llm-tracing">LLM tracing</Link> creates comprehensive audit
        trails of all inputs, outputs, and model interactions - essential for
        regulatory compliance and incident investigation. The AI Gateway adds
        real-time policy enforcement through guardrails that filter inputs for
        prompt injection attempts and outputs for PII, toxicity, or policy
        violations. Combined with LLM judges that continuously assess safety and
        responsible AI metrics, you get end-to-end visibility and control over
        your AI systems' behavior.
      </>
    ),
    answerText:
      "MLflow provides multiple layers for governance and compliance. LLM tracing creates comprehensive audit trails of all inputs, outputs, and model interactions - essential for regulatory compliance and incident investigation. The AI Gateway adds real-time policy enforcement through guardrails that filter inputs for prompt injection attempts and outputs for PII, toxicity, or policy violations. Combined with LLM judges that continuously assess safety and responsible AI metrics, you get end-to-end visibility and control over your AI systems' behavior.",
  },
  {
    question: "How does MLflow AI Observability help prevent runaway costs?",
    answer: (
      <>
        MLflow tracks LLM costs at multiple levels.{" "}
        <Link href="https://mlflow.org/docs/latest/genai/tracing/token-usage-cost/">
          Trace-level cost tracking
        </Link>{" "}
        automatically calculates spending per request based on token usage and
        model pricing, with aggregated dashboards showing cost trends and
        expensive queries. The AI Gateway adds proactive controls through rate
        limiting and cost budgets per endpoint. Together, these give you both
        real-time visibility into spending and guardrails to prevent cost
        overruns before they happen.
      </>
    ),
    answerText:
      "MLflow tracks LLM costs at multiple levels. Trace-level cost tracking automatically calculates spending per request based on token usage and model pricing, with aggregated dashboards showing cost trends and expensive queries. The AI Gateway adds proactive controls through rate limiting and cost budgets per endpoint. Together, these give you both real-time visibility into spending and guardrails to prevent cost overruns before they happen.",
  },
  {
    question:
      "How can I monitor the operational health of my agent or LLM application in production?",
    answer: (
      <>
        MLflow's Observability dashboards provide real-time metrics on latency,
        throughput, error rates, and quality scores across all your agent
        deployments. <Link href="/ai-observability">AI observability</Link>{" "}
        combines distributed tracing (to understand execution flows), automated
        evaluation (to measure quality continuously), and custom judges (to
        monitor application-specific KPIs). You can set up alerts on any metric
        and drill down from high-level trends to individual trace details when
        investigating issues.
      </>
    ),
    answerText:
      "MLflow's Observability dashboards provide real-time metrics on latency, throughput, error rates, and quality scores across all your agent deployments. AI observability combines distributed tracing (to understand execution flows), automated evaluation (to measure quality continuously), and custom judges (to monitor application-specific KPIs). You can set up alerts on any metric and drill down from high-level trends to individual trace details when investigating issues.",
  },
  {
    question:
      "How do I ensure my agent or LLM application is delivering value and meeting user needs?",
    answer:
      "MLflow offers 70+ pre-built LLM judges covering conversation quality (completeness, coherence), relevance (context relevance, groundedness), safety (toxicity, bias), and user experience (frustration detection, helpfulness). You can run these as batch evaluations during development or enable continuous monitoring in production to score every interaction. Combine automated judges with human feedback collection to get a complete picture of whether your agent meets user expectations. The evaluation framework is fully customizable - create domain-specific judges tailored to your use case.",
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
    "Open-source platform for AI observability, experiment tracking, evaluation, and deployment.",
  url: "https://mlflow.org",
  license: "https://www.apache.org/licenses/LICENSE-2.0",
};

export default function AIObservability() {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://mlflow.org/ai-observability" />
        <link rel="canonical" href="https://mlflow.org/ai-observability" />
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
          <h1>AI Observability for LLMs and Agents</h1>

          <p>
            AI observability is the practice of collecting, analyzing, and
            correlating telemetry data across AI systems to understand how they
            behave in development and production. For LLM applications, this is
            known as <a href="#llm-observability">LLM observability</a>. For
            autonomous{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
              agents
            </Link>
            , this is known as{" "}
            <a href="#agent-observability">agent observability</a>. LLM
            observability helps you track prompt quality, token usage, and
            response accuracy. Agent observability helps you debug multi-step
            workflows, tool calls, and reasoning chains.
          </p>

          <p>
            AI observability gives engineering teams deep visibility into how
            their AI applications actually behave: not just whether they are
            running, but whether they are producing correct, safe, and useful
            results. As AI systems move from prototypes to production-critical
            applications, observability becomes essential for maintaining
            quality and trust.
          </p>

          <p>
            Unlike traditional software, AI applications are{" "}
            <strong>non-deterministic</strong>: the same input can produce
            different outputs depending on model state, retrieved context, and
            multi-step agent reasoning. This makes traditional logging and
            monitoring insufficient. AI observability captures the full
            execution context (
            <Link href={MLFLOW_GENAI_DOCS_URL + "prompts/"}>prompts</Link>,
            model responses,{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>tool calls</Link>,
            retrieval results, and{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "llm-evaluate/"}>
              evaluation scores
            </Link>
            ) so teams can understand the "why" behind every output.
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
                src="https://mlflow.org/assets/medias/overview_demo-22b5fa3cb0408e33cd92eea39813ab73.mp4"
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
                <a href="#llm-observability">LLM Observability</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#agent-observability">Agent Observability</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#key-components">Key Components</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#how-to-implement">How to Implement</a>
              </li>
              <li style={{ marginBottom: "0" }}>
                <a href="#faq">FAQ</a>
              </li>
            </ul>
          </div>

          <h2>Why AI Observability Matters</h2>

          <p>
            AI systems, such as agents, LLM applications, and RAG systems,
            introduce unique challenges that traditional software monitoring
            can't address:
          </p>

          <div className="grid-2">
            <div className="card">
              <h3>Debugging Complexity</h3>
              <p>
                <strong>Problem:</strong> Multi-step agents, tool calls, and
                retrieval chains create complex execution paths that are
                difficult to debug.
              </p>
              <p>
                <strong>Solution:</strong> Tracing makes every step visible and
                debuggable, from initial request to final response.
              </p>
            </div>

            <div className="card">
              <h3>Cost Control</h3>
              <p>
                <strong>Problem:</strong> Token costs can spiral out of control
                without visibility into usage patterns and inefficiencies.
              </p>
              <p>
                <strong>Solution:</strong> Track token usage, model selection
                efficiency, and per-request costs to identify optimization
                opportunities.
              </p>
            </div>

            <div className="card">
              <h3>Quality & Reliability</h3>
              <p>
                <strong>Problem:</strong> AI systems can produce hallucinations,
                regressions, and degraded outputs that undermine user trust.
              </p>
              <p>
                <strong>Solution:</strong> Detect issues before they reach
                users. Evaluate every response against quality benchmarks
                automatically.
              </p>
            </div>

            <div className="card">
              <h3>Compliance & Governance</h3>
              <p>
                <strong>Problem:</strong> AI systems make decisions that need
                auditing, and can inadvertently expose PII or violate content
                policies.
              </p>
              <p>
                <strong>Solution:</strong> Maintain complete audit trails and
                enforce PII policies, content guardrails, and access controls
                across your AI stack.
              </p>
            </div>
          </div>

          <h2 id="llm-observability">LLM Observability</h2>

          <p>
            LLM observability focuses on monitoring individual large language
            model calls and LLM-powered applications. This includes tracking
            prompts sent to models like GPT, Claude, or Gemini, capturing the
            completions they return, measuring token consumption and costs, and
            monitoring response latency and quality.
          </p>

          <p>
            For LLM applications (chatbots, content generators, summarization
            tools), observability helps you understand which prompts produce the
            best results, identify expensive or slow queries, and detect quality
            regressions when models are updated. By tracing every LLM call with
            full context (system prompts, user messages, temperature settings,
            token counts), you can debug hallucinations, optimize prompt
            templates, and track costs across different models and use cases.
          </p>

          <p>
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
              MLflow's automatic tracing
            </Link>{" "}
            captures all of this telemetry with a single line of code, storing
            traces locally or sending them to your tracking server for analysis,
            evaluation, and monitoring.
          </p>

          <h2 id="agent-observability">Agent Observability</h2>

          <p>
            Agent observability extends LLM observability to multi-step agentic
            systems. While LLM observability tracks individual model calls,
            agent observability captures the complete execution graph of
            autonomous agents: how they reason about tasks, which tools they
            call and in what order, how they handle errors and retries, and how
            they chain multiple LLM calls together to accomplish complex goals.
          </p>

          <p>
            Agents built with frameworks like LangGraph, CrewAI, or AutoGen can
            behave unpredictably—getting stuck in loops, making incorrect tool
            choices, or producing inconsistent outputs across runs. Agent
            observability makes every reasoning step visible: you can see
            exactly which tools were called with what arguments, what the agent
            learned from each step, and how it decided what to do next.
          </p>

          <p>
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
              MLflow automatically traces agent workflows
            </Link>
            , capturing the full directed acyclic graph (DAG) of execution,
            including parallel tool calls, conditional branches, and iterative
            reasoning loops. This makes it easy to debug agent failures,
            optimize agent prompts and tool selection logic, and monitor agent
            behavior in production.
          </p>

          <h2>Common Use Cases for AI Observability</h2>

          <p>
            AI observability solves real-world problems across the AI
            development lifecycle:
          </p>

          <ul>
            <li>
              <strong>Debugging Hallucinations:</strong> When your agents, LLM
              applications, or RAG systems produce incorrect outputs, tracing
              shows exactly what happened—which documents were retrieved, what
              tool calls were made, which prompts were sent, and what context
              was used. This makes it easy to identify whether the problem is in
              retrieval, reasoning, tool selection, or generation.
            </li>
            <li>
              <strong>Monitoring Agent Behavior in Production:</strong>{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                Agents
              </Link>{" "}
              can behave unpredictably—getting stuck in loops, making incorrect
              tool choices, or producing inconsistent outputs. AI observability
              platforms automatically capture agent execution graphs, showing
              every reasoning step, tool call, and decision point so you can
              identify and fix problematic patterns.
            </li>
            <li>
              <strong>Optimizing LLM Costs:</strong> Track token usage and costs
              across all LLM calls to identify expensive queries, inefficient
              prompts, or opportunities to switch to smaller models for specific
              tasks. AI observability platforms help teams reduce spend by
              30-50% without sacrificing quality.
            </li>
            <li>
              <strong>A/B Testing Prompt Changes:</strong> Before deploying
              prompt modifications to production, AI observability platforms let
              you run{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "llm-evaluate/"}>
                side-by-side evaluations
              </Link>{" "}
              with LLM judges. Compare quality metrics like relevance,
              factuality, and safety to ensure changes improve—not
              degrade—output quality.
            </li>
            <li>
              <strong>Catching Production Regressions:</strong> Monitor quality
              scores, error rates, and latency over time to detect when model
              behavior degrades from API updates, prompt changes, or data
              drift—before users notice.
            </li>
            <li>
              <strong>Maintaining Compliance:</strong> Capture complete audit
              trails showing what prompts were sent, what responses were
              received, and what data was accessed. Enforce PII redaction
              policies and content guardrails to meet regulatory requirements.
            </li>
          </ul>

          <h2 id="key-components">Key Components of AI Observability</h2>

          <p>
            A comprehensive AI observability platform combines six capabilities:
          </p>

          <ul>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "tracing/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Tracing
              </Link>
              : Record every step of request execution with inputs, outputs, and
              latency for each LLM call, retrieval, and tool use.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Evaluation
              </Link>
              : Compare agents and LLM applications side-by-side using automated{" "}
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "llm-evaluate/llm-as-judge/"}
                style={{ color: "#007bff" }}
              >
                LLM judges
              </Link>{" "}
              or custom scoring logic to measure quality improvements.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Monitoring
              </Link>
              : Track quality scores, error rates, and drift with LLM judges to
              catch regressions early with online monitoring.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "tracing/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Cost & Latency Tracking
              </Link>
              : Monitor token consumption and costs per request to optimize
              spending and performance across models.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Human Feedback
              </Link>
              : Gather expert reviews and end-user ratings to identify
              production failures and turn them into test cases for preventing
              regressions.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "governance/ai-gateway/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Governance
              </Link>
              : Maintain complete audit logs of prompts, responses, and data
              access for compliance and debugging.
            </li>
          </ul>

          <h2 id="how-to-implement">How to Implement AI Observability</h2>

          <p>
            Modern open-source AI platforms like{" "}
            <Link href="/genai">MLflow</Link> make it easy to add comprehensive,
            production-grade observability to your agents, LLM applications, and
            RAG systems with minimal code changes.
          </p>

          <p>
            With just a single line of code, you can automatically capture
            traces for every LLM call, including prompts, responses, token
            usage, latency, and model parameters. These traces are stored
            locally or sent to your MLflow tracking server, where you can
            search, filter, and analyze them in the MLflow UI. You can then
            evaluate traces with LLM judges to find quality issues like
            hallucinations and relevance problems, monitor production metrics to
            catch regressions, and debug failures.
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

# Enable automatic tracing for your LLM framework
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

# Enable automatic tracing for your LLM framework
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
              src={ObservabilityHero}
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
              is the largest open-source AI platform, with over 30 million
              monthly downloads. Thousands of organizations, developers, and
              research teams use MLflow each day to build and deploy
              production-grade agents and LLM applications. Backed by the Linux
              Foundation and licensed under Apache 2.0, MLflow provides a
              complete observability stack with no vendor lock-in.{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL}>Get started →</Link>
            </p>
          </div>

          <h2>Open Source vs. Proprietary AI Observability</h2>

          <p>
            When choosing an AI observability platform, the decision between
            open source and proprietary SaaS tools has significant long-term
            implications for your team, infrastructure, and data ownership.
          </p>

          <p>
            <strong>
              Open Source (<Link href="/genai">MLflow</Link>):
            </strong>{" "}
            With MLflow, you maintain complete control over your observability
            infrastructure and data. Deploy on your own infrastructure or use
            managed versions on Databricks, AWS, or other platforms. There are
            no per-seat fees, no usage limits, and no vendor lock-in. Your
            telemetry data stays under your control, and you can customize the
            platform to your exact needs. MLflow integrates with any LLM
            provider and agent framework through OpenTelemetry-compatible
            tracing.
          </p>

          <p>
            <strong>Proprietary SaaS Tools:</strong> Commercial observability
            platforms offer convenience but at the cost of flexibility and
            control. They typically charge per seat or per trace volume, which
            can become expensive at scale. Your data is sent to their servers,
            raising privacy and compliance concerns. You're locked into their
            ecosystem, making it difficult to switch providers or customize
            functionality. Most proprietary tools only support a subset of LLM
            providers and frameworks.
          </p>

          <p>
            <strong>Why Teams Choose Open Source:</strong> Organizations
            building production AI applications increasingly choose MLflow
            because it offers enterprise-grade observability without
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
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                Agent Evaluation Guide
              </Link>
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
