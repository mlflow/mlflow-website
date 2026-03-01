import { useState } from "react";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../../components/Header/Header";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import { CopyButton } from "../../components/CodeSnippet/CopyButton";
import {
  customNightOwl,
  CODE_BG,
} from "../../components/CodeSnippet/codeTheme";

const SEO_TITLE = "AI Gateway for LLMs & Agents | MLflow Agent Platform";
const SEO_DESCRIPTION =
  "Learn AI Gateway, LLM Gateway, and Agent Gateway with MLflow - the comprehensive, open-source agent engineering and ops platform.";

const faqs: {
  question: string;
  answer: React.ReactNode;
  answerText?: string;
}[] = [
  {
    question: "What is an AI Gateway?",
    answer:
      "An AI Gateway is a centralized proxy layer that sits between your applications and LLM providers (OpenAI, Anthropic, Bedrock, etc.). It provides a single, unified API endpoint for all your LLM calls, centralizes API key management, tracks usage and costs, and enforces governance policies. AI Gateways eliminate the need to scatter API keys across your infrastructure and give you complete visibility into how your organization uses LLMs.",
  },
  {
    question: "How is an AI Gateway different from calling LLM APIs directly?",
    answer:
      "Calling LLM APIs directly means managing separate credentials for each provider, tracking usage manually, and having no centralized control or visibility. An AI Gateway centralizes all of this: one authentication method, automatic usage tracking, cost monitoring, traffic splitting for A/B testing, automatic fallbacks for reliability, and complete audit trails. It also makes it easy to switch providers without changing application code.",
  },
  {
    question: "What are the key components of an AI Gateway?",
    answer:
      "A comprehensive AI Gateway includes: unified API endpoint (OpenAI-compatible), centralized credential management, usage tracking and cost monitoring, traffic splitting and A/B testing, automatic fallback and retry logic, governance policies (PII redaction, content filtering), and integration with observability platforms for tracing and evaluation.",
  },
  {
    question: "Do I need an AI Gateway for my LLM application?",
    answer:
      "Yes, if you're building production LLM applications. An AI Gateway helps you manage API keys securely, track token costs across teams and projects, enforce governance policies, switch between providers without code changes, and maintain complete audit trails for compliance. Without a gateway, you're managing credentials manually, tracking costs in spreadsheets, and risking PII leaks to third-party APIs.",
  },
  {
    question: "What's the difference between an LLM Gateway and an AI Gateway?",
    answer:
      "LLM Gateway and AI Gateway are often used interchangeably. Historically, LLM Gateway referred specifically to routing calls to large language models like GPT or Claude. AI Gateway is broader, encompassing LLMs plus other AI services. Most modern AI Gateways focus on LLM routing and governance, making the terms functionally equivalent.",
  },
  {
    question: "What is an Agent Gateway?",
    answer:
      "Agent Gateway extends LLM Gateway capabilities to multi-step agentic systems. While an LLM Gateway routes individual model calls, an Agent Gateway handles complex agent workflows involving multiple LLM calls, tool invocations, and retrieval steps. It provides end-to-end tracing across agent execution graphs, aggregates costs across all calls in an agent session, and enforces governance policies at every step of agent reasoning.",
  },
  {
    question: "What is the best AI Gateway?",
    answer:
      "The best AI Gateway depends on your needs. MLflow AI Gateway is the leading open-source option, offering complete routing, usage tracking, and observability without vendor lock-in. MLflow supports any LLM provider (OpenAI, Anthropic, Bedrock, Gemini, etc.), any agent framework (LangChain, LangGraph, CrewAI, AutoGen, etc.), is fully OpenTelemetry compatible, and gives you full control over your data. Unlike proprietary SaaS gateways, MLflow is backed by a community of 20,000+ GitHub stars and 900+ contributors.",
  },
  {
    question: "What LLM providers does the MLflow AI Gateway support?",
    answer:
      "MLflow AI Gateway supports all major LLM providers: OpenAI (GPT models), Anthropic (Claude), AWS Bedrock, Google Gemini, Azure OpenAI, Mistral, Cohere, Groq, Together AI, Fireworks AI, DeepSeek, Qwen, and more. The gateway exposes an OpenAI-compatible API, so switching providers requires only a configuration change - no code changes needed.",
  },
  {
    question: "How does the MLflow AI Gateway compare to other AI Gateways?",
    answer:
      "Unlike proprietary gateways that lock you into a vendor's ecosystem, MLflow AI Gateway is fully open source and runs as part of your existing MLflow Tracking Server. You get AI Gateway capabilities without deploying separate infrastructure, and your usage data automatically feeds into MLflow's tracing and evaluation workflows. Proprietary gateways charge per request or per seat, while MLflow is 100% free under the Apache 2.0 license.",
  },
  {
    question: "Does the MLflow AI Gateway support OpenTelemetry?",
    answer:
      "Yes. MLflow AI Gateway's usage tracking is built on MLflow Tracing, which is fully compatible with OpenTelemetry. You can export traces to any OpenTelemetry-compatible backend, giving you total ownership and portability of your telemetry data without vendor lock-in.",
  },
  {
    question: "Is the MLflow AI Gateway free?",
    answer:
      "Yes. MLflow is 100% open source under the Apache 2.0 license, backed by the Linux Foundation. You can use all of its AI Gateway features (routing, usage tracking, governance, and observability integration) for free, including in commercial applications.",
  },
  {
    question: "How do I get started with the MLflow AI Gateway?",
    answer: (
      <>
        Getting started with MLflow AI Gateway is simple. Install MLflow with{" "}
        <Link
          href={MLFLOW_GENAI_DOCS_URL + "governance/ai-gateway/quickstart/"}
        >
          pip install 'mlflow[genai]'
        </Link>
        , start the server with <code>mlflow server</code>, and configure your
        first endpoint in the MLflow UI. Then point your application to the
        gateway's base URL and start making requests. See the{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "governance/ai-gateway/"}>
          MLflow AI Gateway documentation
        </Link>{" "}
        for detailed setup instructions and examples.
      </>
    ),
    answerText:
      "Getting started with MLflow AI Gateway is simple. Install MLflow with pip install 'mlflow[genai]', start the server with mlflow server, and configure your first endpoint in the MLflow UI. Then point your application to the gateway's base URL and start making requests. See the MLflow AI Gateway documentation for detailed setup instructions and examples.",
  },
  {
    question:
      "Is it easy to integrate MLflow AI Gateway with my existing LLM application?",
    answer:
      "Yes. MLflow AI Gateway exposes an OpenAI-compatible API, so if your application already uses the OpenAI SDK, integration requires only changing the base_url parameter. The gateway works with any LLM provider, any agent framework, and supports OpenTelemetry for exporting traces. You can self-host MLflow or use managed versions on Databricks and AWS.",
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
    "Open-source platform for AI Gateway, AI observability, experiment tracking, evaluation, and deployment.",
  url: "https://mlflow.org",
  license: "https://www.apache.org/licenses/LICENSE-2.0",
};

export default function AIGateway() {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://mlflow.org/faq/ai-gateway" />
        <link rel="canonical" href="https://mlflow.org/faq/ai-gateway" />
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
          .faq-answer a {
            font-weight: 600;
            color: #0194e2 !important;
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
          <h1>AI Gateway for LLMs and Agents</h1>

          <p>
            An AI Gateway is a centralized proxy layer that routes requests to
            LLM providers through a single, unified API. It manages credentials,
            tracks usage, enforces governance policies, and provides complete
            observability across all LLM calls. For LLM applications, this is
            known as an <a href="#llm-gateway">LLM Gateway</a>. For autonomous{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "agent-evaluation/"}>
              agents
            </Link>
            , this is known as an <a href="#agent-gateway">Agent Gateway</a>.
            LLM Gateways centralize API key management and track usage across
            providers. Agent Gateways extend this to multi-step agentic
            workflows with end-to-end tracing and cost tracking.
          </p>

          <p>
            AI Gateways give engineering teams centralized control over how
            their applications access LLMs. They route requests, manage
            credentials securely, track token costs, enforce governance
            policies, and maintain complete audit trails. As AI systems move
            from prototypes to production, gateways become essential for
            security, compliance, and cost control.
          </p>

          <p>
            Unlike direct LLM API calls, which scatter credentials across your
            infrastructure and provide no visibility into usage patterns, an AI
            Gateway centralizes everything. It provides a single authentication
            point, automatic usage tracking, cost monitoring dashboards, traffic
            splitting for A/B testing, automatic fallback chains for
            reliability, and complete{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>tracing</Link>{" "}
            integration so you can analyze every request in context.
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
                  require("@site/static/img/releases/3.10.0/gateway-usage.mp4")
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
                <a href="#llm-gateway">LLM Gateway</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#agent-gateway">Agent Gateway</a>
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

          <h2>Why AI Gateway Matters</h2>

          <p>
            AI systems, such as agents, LLM applications, and RAG systems,
            introduce unique operational challenges that direct API calls can't
            address:
          </p>

          <div className="grid-2">
            <div className="card">
              <h3>Security & Credential Management</h3>
              <p>
                <strong>Problem:</strong> API keys scattered across notebooks,
                CI environments, and developer machines create security risks
                and compliance headaches.
              </p>
              <p>
                <strong>Solution:</strong> Centralize all credentials in the
                gateway. Applications authenticate to the gateway, never
                directly to LLM providers.
              </p>
            </div>

            <div className="card">
              <h3>Cost Visibility & Control</h3>
              <p>
                <strong>Problem:</strong> Token costs spiral out of control when
                teams have no visibility into who's using what models or how
                much they're spending.
              </p>
              <p>
                <strong>Solution:</strong> Track usage and costs per endpoint,
                team, or project. Identify expensive queries and optimize
                spending.
              </p>
            </div>

            <div className="card">
              <h3>Vendor Flexibility</h3>
              <p>
                <strong>Problem:</strong> Switching LLM providers requires code
                changes across every application that calls them.
              </p>
              <p>
                <strong>Solution:</strong> Change provider configurations in the
                gateway without touching application code. A/B test models or
                set up automatic fallbacks.
              </p>
            </div>

            <div className="card">
              <h3>Governance & Compliance</h3>
              <p>
                <strong>Problem:</strong> Sensitive data and PII can leak to
                third-party APIs without centralized controls or audit trails.
              </p>
              <p>
                <strong>Solution:</strong> Enforce PII redaction, content
                policies, and access controls at the gateway level. Maintain
                complete audit logs.
              </p>
            </div>
          </div>

          <h2 id="llm-gateway">LLM Gateway</h2>

          <p>
            An LLM Gateway routes requests to large language model providers
            like OpenAI, Anthropic, and Bedrock through a single, unified API.
            Instead of integrating with each provider's SDK separately, your
            application points to the gateway's OpenAI-compatible endpoint and
            specifies which model to use by name.
          </p>

          <p>
            For LLM applications (chatbots, content generators, summarization
            tools), an LLM Gateway centralizes credential management so API keys
            never touch application code, tracks token usage and costs across
            all providers in one dashboard, enables traffic splitting for A/B
            testing different models, and provides automatic fallback chains
            when providers have outages.
          </p>

          <p>
            <Link href={MLFLOW_GENAI_DOCS_URL + "governance/ai-gateway/"}>
              MLflow AI Gateway
            </Link>{" "}
            runs as part of your MLflow Tracking Server and exposes an
            OpenAI-compatible endpoint for any LLM provider. Configure endpoints
            in the MLflow UI, and your application code stays unchanged when
            switching providers or models.
          </p>

          <h2 id="agent-gateway">Agent Gateway</h2>

          <p>
            An Agent Gateway extends LLM Gateway capabilities to multi-step
            agentic systems. While an LLM Gateway routes individual model calls,
            an Agent Gateway handles complex agent workflows involving multiple
            LLM calls, tool invocations, retrieval steps, and reasoning loops.
          </p>

          <p>
            Agents built with frameworks like LangGraph, CrewAI, or AutoGen make
            dozens or hundreds of LLM calls per user request. An Agent Gateway
            aggregates costs across entire agent sessions, provides end-to-end{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>tracing</Link>{" "}
            across the agent's execution graph, enforces governance policies at
            every step, and automatically captures usage metrics for debugging
            and optimization.
          </p>

          <p>
            <Link href={MLFLOW_GENAI_DOCS_URL + "governance/ai-gateway/"}>
              MLflow AI Gateway
            </Link>{" "}
            integrates natively with MLflow Tracing, so every request through
            the gateway automatically becomes an MLflow trace. This gives you
            complete visibility into agent behavior, token costs, and quality
            metrics without additional instrumentation.
          </p>

          <h2>Common Use Cases for AI Gateway</h2>

          <p>
            AI Gateway solves real-world problems across production AI systems:
          </p>

          <ul>
            <li>
              <strong>Securing API Keys:</strong> Instead of distributing OpenAI
              or Anthropic API keys to every developer and service, store them
              encrypted in the gateway. Applications authenticate to the gateway
              using your existing auth system, and credentials never leave the
              server.
            </li>
            <li>
              <strong>Tracking Token Costs by Team:</strong> When multiple teams
              share the same LLM provider account, the gateway tracks usage per
              endpoint or per team, making it easy to allocate costs and
              identify optimization opportunities.
            </li>
            <li>
              <strong>A/B Testing Model Changes:</strong> Before switching from
              GPT to Claude (or from one model version to another), use{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "governance/ai-gateway/"}>
                traffic splitting
              </Link>{" "}
              to route 10% of requests to the new model. Compare quality metrics
              and costs before fully migrating.
            </li>
            <li>
              <strong>Automatic Failover:</strong> Configure the gateway with
              fallback chains: if OpenAI is unavailable, automatically route
              requests to Anthropic. This improves reliability without changing
              application code.
            </li>
            <li>
              <strong>Enforcing Content Guardrails:</strong> Apply content
              safety filters, PII redaction, and toxicity detection at the
              gateway level to ensure all LLM requests and responses meet
              compliance and safety requirements before reaching users.
            </li>
            <li>
              <strong>Compliance Audit Trails:</strong> Capture complete logs of
              every request and response passing through the gateway.
              Demonstrate compliance with data policies and regulatory
              requirements.
            </li>
          </ul>

          <h2 id="key-components">Key Components of AI Gateway</h2>

          <p>
            A comprehensive AI Gateway platform combines seven capabilities:
          </p>

          <ul>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "governance/ai-gateway/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Unified API Endpoint
              </Link>
              : Single OpenAI-compatible endpoint for all LLM providers. Switch
              models by changing configuration, not code.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "governance/ai-gateway/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Credential Management
              </Link>
              : Centralized, encrypted storage of API keys. Applications
              authenticate to the gateway, not to LLM providers directly.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "governance/ai-gateway/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Usage Tracking
              </Link>
              : Automatic tracking of token usage, costs, latency, and error
              rates per endpoint, model, or team.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "governance/ai-gateway/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Traffic Splitting
              </Link>
              : A/B test different models or providers by routing a percentage
              of requests to each. Gradual rollouts without code changes.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "governance/ai-gateway/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Fallback & Retry Logic
              </Link>
              : Automatic failover to backup providers when primary is
              unavailable. Configurable retry policies for transient errors.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "tracing/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Observability Integration
              </Link>
              : Native integration with tracing platforms to capture request
              context, evaluate responses, and monitor production metrics.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "governance/ai-gateway/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Guardrails & Policy Enforcement
              </Link>
              : Apply content filters, PII redaction, and safety policies at the
              gateway level to ensure all LLM requests meet compliance and
              security requirements.
            </li>
          </ul>

          <h2 id="how-to-implement">Getting Started with AI Gateways</h2>

          <p>
            Modern open-source AI platforms like{" "}
            <Link href="/genai">MLflow</Link> make it easy to deploy a
            production-grade AI Gateway with minimal setup. MLflow AI Gateway
            runs as part of the MLflow Tracking Server, so there's no separate
            infrastructure to deploy or maintain.
          </p>

          <h3 style={{ marginTop: "48px", marginBottom: "32px" }}>
            Setting Up the MLflow AI Gateway
          </h3>

          <p style={{ marginTop: "16px", marginBottom: "32px" }}>
            For a comprehensive setup guide, visit the{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL + "governance/ai-gateway/quickstart/"
              }
            >
              MLflow AI Gateway quickstart documentation
            </Link>
            . Here's a quick overview to get started:
          </p>

          <p style={{ marginTop: "32px", marginBottom: "12px" }}>
            <strong>1. Install MLflow with GenAI support:</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "16px 0 32px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">bash</span>
              <CopyButton code={`pip install 'mlflow[genai]'`} />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={`pip install 'mlflow[genai]'`}
                language="bash"
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

          <p style={{ marginTop: "24px", marginBottom: "12px" }}>
            <strong>2. Start the MLflow server:</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "16px 0 32px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">bash</span>
              <CopyButton code={`mlflow server`} />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={`mlflow server`}
                language="bash"
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

          <p style={{ marginTop: "24px", marginBottom: "12px" }}>
            <strong>
              3. Configure your first gateway endpoint in the MLflow UI:
            </strong>
          </p>

          <div
            style={{
              margin: "16px 0 40px 0",
              borderRadius: "8px",
              overflow: "hidden",
              border: "1px solid #e5e7eb",
            }}
          >
            <video width="100%" controls loop muted playsInline>
              <source
                src={
                  require("@site/static/img/blog/mlflow-ai-gateway/gateway-usage-tracking.mp4")
                    .default
                }
                type="video/mp4"
              />
              Your browser does not support the video tag.
            </video>
          </div>

          <p>
            Navigate to the AI Gateway tab in the MLflow UI, create a new
            endpoint, select your LLM provider (OpenAI, Anthropic, Bedrock,
            etc.), configure your API credentials, and save. The gateway is now
            ready to route requests.
          </p>

          <p>
            Check out the{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "governance/ai-gateway/"}>
              MLflow AI Gateway documentation
            </Link>{" "}
            for detailed configuration options and advanced features like
            traffic splitting and fallback chains.
          </p>

          <h3 style={{ marginTop: "48px", marginBottom: "32px" }}>
            Querying the Gateway
          </h3>

          <p style={{ marginTop: "16px" }}>
            Once your gateway is configured, point your application to the
            gateway's base URL using the OpenAI SDK (or any OpenAI-compatible
            client). The gateway handles authentication, routes requests to the
            correct provider, and automatically captures{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>traces</Link> for
            every request.
          </p>

          <p style={{ marginTop: "32px", marginBottom: "0px" }}>
            <strong>Example: Querying with OpenAI SDK</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton
                code={`from openai import OpenAI

client = OpenAI(
    base_url="https://your-mlflow-server/gateway/mlflow/v1",
    api_key="",  # authentication handled by gateway
)

response = client.chat.completions.create(
    model="prod-gpt5",  # name of your gateway endpoint
    messages=[{"role": "user", "content": "Summarize this support ticket..."}],
)`}
              />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={`from openai import OpenAI

client = OpenAI(
    base_url="https://your-mlflow-server/gateway/mlflow/v1",
    api_key="",  # authentication handled by gateway
)

response = client.chat.completions.create(
    model="prod-gpt5",  # name of your gateway endpoint
    messages=[{"role": "user", "content": "Summarize this support ticket..."}],
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
            <strong>Example: Querying with Anthropic Claude SDK</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton
                code={`import anthropic

client = anthropic.Anthropic(
    base_url="https://your-mlflow-server/gateway/anthropic",
    api_key="dummy",  # authentication handled by gateway
)

response = client.messages.create(
    model="my-claude-endpoint",  # name of your gateway endpoint
    max_tokens=1024,
    messages=[{"role": "user", "content": "Summarize this support ticket..."}],
)`}
              />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={`import anthropic

client = anthropic.Anthropic(
    base_url="https://your-mlflow-server/gateway/anthropic",
    api_key="dummy",  # authentication handled by gateway
)

response = client.messages.create(
    model="my-claude-endpoint",  # name of your gateway endpoint
    max_tokens=1024,
    messages=[{"role": "user", "content": "Summarize this support ticket..."}],
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

          <div className="info-box">
            <p>
              <Link href="/genai" style={{ color: "#007bff" }}>
                <strong>MLflow</strong>
              </Link>{" "}
              is the largest open-source AI platform, backed by the Linux
              Foundation and licensed under Apache 2.0. With 20,000+ GitHub
              stars and 900+ contributors, it provides a complete AI Gateway
              solution with no vendor lock-in.{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL}>Get started →</Link>
            </p>
          </div>

          <h2>End-to-End Platform vs. Standalone Gateway</h2>

          <p>
            When evaluating AI Gateway solutions, the most important decision is
            whether to use a standalone gateway or one integrated into an
            end-to-end AI platform. This choice has significant implications for
            your team's productivity, infrastructure complexity, and ability to
            debug and improve AI applications.
          </p>

          <p>
            <strong>Standalone Gateways (LiteLLM, etc.):</strong> A standalone
            AI gateway solves one piece of the puzzle: it proxies your LLM calls
            and centralizes credentials. But in practice, routing requests is
            just the beginning. You still need to trace what happened inside
            your application after the LLM responded, evaluate whether the
            output was actually good, and tie cost and latency data back to
            specific features, prompts, or model versions. With a standalone
            gateway, that means integrating a separate observability tool, a
            separate evaluation framework, and building the glue code to connect
            them all to the same data. Every new tool in the stack is another
            thing to deploy, monitor, and keep in sync.
          </p>

          <p>
            <strong>
              End-to-End Platform (<Link href="/genai">MLflow</Link>):
            </strong>{" "}
            MLflow eliminates the integration tax. Because the AI Gateway,
            tracing, and evaluation all live in the same platform, you get
            automatic benefits that standalone gateways can't provide:
          </p>

          <ul>
            <li>
              <strong>Traces are automatic:</strong> Every gateway request
              becomes an{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
                MLflow trace
              </Link>
              , no additional SDK or instrumentation required. Those traces
              include the full request/response payload alongside latency and
              token counts.
            </li>
            <li>
              <strong>Evaluation runs on real traffic:</strong> Traces captured
              through the gateway feed directly into{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "llm-evaluate/"}>
                MLflow's evaluation APIs
              </Link>
              , so you can run LLM judges over production data without exporting
              anything or wiring up a pipeline.
            </li>
            <li>
              <strong>Debugging is one click away:</strong> When the usage
              dashboard shows a latency spike or error rate increase, you can
              drill straight into the individual traces that caused it - no
              context-switching between tools.
            </li>
            <li>
              <strong>Cost data has context:</strong> Token costs link to
              application traces, showing you exactly why spending increased or
              decreased.
            </li>
          </ul>

          <p>
            The alternative - stitching together a gateway, an observability
            platform, and an evaluation framework - creates data silos,
            duplicated configuration, and a fragile integration surface.
            MLflow's approach is to make the gateway a natural extension of the
            platform teams are already using for GenAI development, so that
            governance and observability come for free rather than as an
            afterthought.
          </p>

          <h2>Open Source vs. Proprietary AI Gateway</h2>

          <p>
            When choosing an AI Gateway platform, the decision between open
            source and proprietary SaaS tools has significant long-term
            implications for your infrastructure, security posture, and costs.
          </p>

          <p>
            <strong>
              Open Source (<Link href="/genai">MLflow</Link>):
            </strong>{" "}
            With MLflow AI Gateway, you maintain complete control over your
            gateway infrastructure and routing policies. Deploy on your own
            infrastructure or use managed versions on Databricks or AWS. There
            are no per-request fees, no usage limits, and no vendor lock-in.
            Your API keys and request data stay under your control, and you can
            customize the gateway to your exact security and compliance
            requirements. MLflow integrates with any LLM provider through
            OpenTelemetry-compatible tracing.
          </p>

          <p>
            <strong>Proprietary SaaS Gateways:</strong> Commercial AI Gateway
            platforms offer convenience but at the cost of flexibility and
            control. They typically charge per request or per seat, which can
            become expensive at scale. Your API keys and request data are sent
            to their servers, raising privacy and compliance concerns. You're
            locked into their ecosystem, making it difficult to switch providers
            or add custom functionality. Most proprietary gateways only support
            a subset of LLM providers.
          </p>

          <p>
            <strong>Why Teams Choose Open Source:</strong> Organizations
            building production AI applications increasingly choose MLflow AI
            Gateway because it offers enterprise-grade routing and governance
            without compromising on data sovereignty, cost predictability, or
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
              <Link href={MLFLOW_GENAI_DOCS_URL + "governance/ai-gateway/"}>
                AI Gateway Documentation
              </Link>
            </li>
            <li>
              <Link
                href={
                  MLFLOW_GENAI_DOCS_URL + "governance/ai-gateway/quickstart/"
                }
              >
                AI Gateway Quickstart Guide
              </Link>
            </li>
            <li>
              <Link href="/blog/mlflow-ai-gateway">
                MLflow AI Gateway Announcement
              </Link>
            </li>
            <li>
              <Link href="/faq/ai-observability">AI Observability FAQ</Link>
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
