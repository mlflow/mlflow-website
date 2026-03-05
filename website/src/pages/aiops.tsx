import { useState } from "react";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../components/Header/Header";
import { SocialLinksFooter } from "../components/SocialLinksFooter/SocialLinksFooter";
import { ArticleSidebar } from "../components/ArticleSidebar/ArticleSidebar";
import { MLFLOW_GENAI_DOCS_URL, MLFLOW_DOCS_URL } from "@site/src/constants";
import ObservabilityHero from "@site/static/img/GenAI_observability/GenAI_observability_hero.png";
import { CopyButton } from "../components/CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";

const SEO_TITLE = "What is AIOps? AI Operations Guide | MLflow AI Platform";
const SEO_DESCRIPTION =
  "Learn AIOps with MLflow, the largest open source AI engineering platform. Trace, evaluate, and monitor LLM agents, RAG systems, and ML models in production.";

const faqs: {
  question: string;
  answer: React.ReactNode;
  answerText?: string;
}[] = [
  {
    question: "What is AIOps?",
    answer:
      "AIOps (AI Operations) refers to the practices, tools, and workflows for running AI applications in production across their full lifecycle, from development and evaluation through deployment and monitoring. This includes LLM applications, agents, RAG systems, and traditional machine learning models.",
  },
  {
    question:
      "How is modern AIOps different from traditional AIOps (AI for IT)?",
    answer:
      "Traditionally, 'AIOps' meant using AI to improve IT operations (log analysis, anomaly detection, incident management). Modern AIOps has evolved to also mean operations for AI: the practices and platforms needed to build, deploy, and maintain AI applications in production. MLflow focuses on this modern definition: running AI applications at scale.",
  },
  {
    question: "How is AIOps different from MLOps and LLMOps?",
    answer: (
      <>
        AIOps is the broadest term, encompassing operations for all AI
        applications. MLOps focuses specifically on traditional machine learning
        (training, versioning, deploying models).{" "}
        <Link href="/llmops">LLMOps</Link> focuses on LLM-specific challenges
        (prompt management, non-deterministic evaluation, token costs). AIOps
        unifies both under a single operational discipline, recognizing that
        modern AI teams work across ML and LLM workloads.
      </>
    ),
    answerText:
      "AIOps is the broadest term, encompassing operations for all AI applications. MLOps focuses specifically on traditional machine learning (training, versioning, deploying models). LLMOps focuses on LLM-specific challenges (prompt management, non-deterministic evaluation, token costs). AIOps unifies both under a single operational discipline, recognizing that modern AI teams work across ML and LLM workloads.",
  },
  {
    question: "What are the key capabilities of an AIOps platform?",
    answer: (
      <>
        An AIOps platform typically provides:{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>tracing</Link>{" "}
        (execution capture for LLM and agent debugging),{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>evaluation</Link>{" "}
        (automated quality assessment),{" "}
        <Link href={MLFLOW_DOCS_URL + "ml/tracking/"}>experiment tracking</Link>{" "}
        (for ML and LLM experiments),{" "}
        <Link href={MLFLOW_DOCS_URL + "ml/model-registry/"}>
          model registry
        </Link>{" "}
        (versioning and lifecycle management), and{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          production monitoring
        </Link>
        .
      </>
    ),
    answerText:
      "An AIOps platform typically provides: tracing (execution capture for LLM and agent debugging), evaluation (automated quality assessment), experiment tracking (for ML and LLM experiments), model registry (versioning and lifecycle management), and production monitoring.",
  },
  {
    question: "Do I need AIOps?",
    answer:
      "Yes, if you're building production AI applications of any kind. AIOps helps you manage the full lifecycle, whether you're training traditional ML models, building LLM-powered chatbots, or deploying multi-step agents. Without AIOps practices, teams struggle with reproducibility, debugging, quality assurance, and cost management at scale.",
  },
  {
    question: "What is AI Ops vs AIOps?",
    answer:
      "AI Ops and AIOps refer to the same concept: operations for AI applications. 'AIOps' is the more common compound form, following the convention of DevOps and MLOps. Both terms describe the tools and practices needed to operationalize AI applications across their full lifecycle.",
  },
  {
    question: "What is the best AIOps platform?",
    answer:
      "The best AIOps platform depends on your needs. MLflow is the leading open-source option, providing a unified platform for both traditional ML operations (experiment tracking, model registry) and modern LLM operations (tracing, evaluation, prompt management). MLflow supports any framework, any model, and any cloud provider, with over 30 million monthly downloads and Linux Foundation backing.",
  },
  {
    question: "How does MLflow support AIOps?",
    answer: (
      <>
        MLflow provides a unified AIOps platform covering both traditional ML
        and modern LLM workloads:{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>automatic tracing</Link>{" "}
        for LLM debugging,{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          evaluation with LLM judges
        </Link>
        ,{" "}
        <Link href={MLFLOW_DOCS_URL + "ml/tracking/"}>experiment tracking</Link>{" "}
        for ML workflows,{" "}
        <Link href={MLFLOW_DOCS_URL + "ml/model-registry/"}>
          model registry
        </Link>{" "}
        for versioning, and{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          production monitoring
        </Link>{" "}
        for ongoing quality tracking.
      </>
    ),
    answerText:
      "MLflow provides a unified AIOps platform covering both traditional ML and modern LLM workloads: automatic tracing for LLM debugging, evaluation with LLM judges, experiment tracking for ML workflows, model registry for versioning, and production monitoring for ongoing quality tracking.",
  },
  {
    question: "Can AIOps handle both ML models and LLM applications?",
    answer:
      "Yes. A modern AIOps platform like MLflow is designed to handle both traditional ML models (scikit-learn, PyTorch, TensorFlow) and LLM applications (OpenAI, Anthropic, open-source models) under a single operational framework. This unified approach prevents tool sprawl and gives teams a consistent workflow across all AI workloads.",
  },
  {
    question: "Is MLflow free for AIOps?",
    answer:
      "Yes. MLflow is 100% open source under the Apache 2.0 license, backed by the Linux Foundation. You can use all AIOps features (tracing, evaluation, experiment tracking, model registry, monitoring) for free, including in commercial applications. There are no per-seat fees, no usage limits, and no vendor lock-in.",
  },
  {
    question: "How do I get started with AIOps?",
    answer: (
      <>
        Getting started with AIOps using MLflow depends on your workload. For
        LLM applications, enable{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/quickstart/"}>
          automatic tracing
        </Link>{" "}
        with a single line of code. For traditional ML, start with{" "}
        <Link href={MLFLOW_DOCS_URL + "ml/getting-started/"}>
          experiment tracking
        </Link>{" "}
        to log parameters, metrics, and artifacts. MLflow provides a unified
        platform so you can adopt both incrementally.
      </>
    ),
    answerText:
      "Getting started with AIOps using MLflow depends on your workload. For LLM applications, enable automatic tracing with a single line of code. For traditional ML, start with experiment tracking to log parameters, metrics, and artifacts. MLflow provides a unified platform so you can adopt both incrementally.",
  },
  {
    question: "What's the relationship between AIOps and AI observability?",
    answer: (
      <>
        <Link href="/ai-observability">AI observability</Link> is a core
        component of AIOps, focused on monitoring and understanding AI system
        behavior through tracing, metrics, and evaluation. AIOps is broader,
        also encompassing experiment management, model versioning, deployment
        workflows, prompt management, and the full operational lifecycle from
        development through production.
      </>
    ),
    answerText:
      "AI observability is a core component of AIOps, focused on monitoring and understanding AI system behavior through tracing, metrics, and evaluation. AIOps is broader, also encompassing experiment management, model versioning, deployment workflows, prompt management, and the full operational lifecycle from development through production.",
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
    "Open-source AIOps platform for tracing, evaluating, tracking, and deploying AI applications and ML models.",
  url: "https://mlflow.org",
  license: "https://www.apache.org/licenses/LICENSE-2.0",
};

const TRACING_CODE = `import mlflow
from openai import OpenAI

# Enable automatic tracing for OpenAI
mlflow.openai.autolog()

# Every LLM call is now traced with full context
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Summarize MLflow"}],
)`;

const EVAL_CODE = `import mlflow.genai

# Evaluate traced outputs with LLM judges
results = mlflow.genai.evaluate(
    data=mlflow.search_traces(experiment_ids=["1"]),
    scorers=[
        mlflow.genai.scorers.Relevance(),
        mlflow.genai.scorers.Safety(),
    ],
)`;

const EXPERIMENT_CODE = `import mlflow

# Track traditional ML experiments
mlflow.set_experiment("my-classification-model")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 100)

    # Train your model...
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("f1_score", 0.93)

    # Log the model artifact
    mlflow.sklearn.log_model(model, "model")`;

export default function AIOps() {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://mlflow.org/aiops" />
        <link rel="canonical" href="https://mlflow.org/aiops" />
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
          <h1>What is AIOps?</h1>

          <p>
            AIOps (AI Operations) is the practice of running AI applications in
            production across their full lifecycle, from development and
            evaluation through deployment and monitoring. This encompasses{" "}
            <Link href="/llmops">LLM applications and agents</Link>, RAG
            systems, and{" "}
            <Link href={MLFLOW_DOCS_URL + "ml/tracking/"}>
              traditional machine learning models
            </Link>
            .
          </p>

          <p>
            Historically, "AIOps" referred to using AI for IT operations
            (automated log analysis, anomaly detection, incident management).
            Today, the term has evolved to also describe the{" "}
            <strong>operations for AI</strong>: the practices and platforms
            needed to build, deploy, and maintain AI applications in production.
            As organizations adopt LLMs, agents, and ML models at scale, AIOps
            provides a unified framework to manage all of these workloads.
          </p>

          <p>
            <Link href="/genai">MLflow</Link> is the most adopted open-source
            AIOps platform, providing a unified stack for both LLMOps (
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>tracing</Link>,{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
              evaluation
            </Link>
            ,{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/"}>
              prompt management
            </Link>
            , <Link href="/genai/ai-gateway">AI Gateway</Link>) and traditional
            ML operations (
            <Link href={MLFLOW_DOCS_URL + "ml/tracking/"}>
              experiment tracking
            </Link>
            ,{" "}
            <Link href={MLFLOW_DOCS_URL + "ml/model-registry/"}>
              model registry
            </Link>
            ).
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

          <h2 id="why-aiops-matters">Why AIOps Matters</h2>

          <p>
            AI applications, whether LLM-powered agents or traditional ML
            models, introduce operational challenges that standard DevOps can't
            address:
          </p>

          <div className="grid-2">
            <div className="card">
              <h3>Fragmented AI Tooling</h3>
              <p>
                <strong>Problem:</strong> Teams use separate tools for ML
                experiment tracking, LLM tracing, evaluation, and deployment,
                creating tool sprawl and fragmented workflows.
              </p>
              <p>
                <strong>Solution:</strong> A unified AIOps platform manages all
                AI workloads (ML models, LLM apps, and agents) under a single
                framework.
              </p>
            </div>

            <div className="card">
              <h3>Quality at Scale</h3>
              <p>
                <strong>Problem:</strong> AI outputs are non-deterministic and
                can degrade silently, making it hard to maintain quality across
                thousands of daily requests.
              </p>
              <p>
                <strong>Solution:</strong> Automated evaluation with LLM judges
                and continuous monitoring catch regressions before they reach
                users.
              </p>
            </div>

            <div className="card">
              <h3>Reproducibility</h3>
              <p>
                <strong>Problem:</strong> Without systematic tracking of
                parameters, data, models, and prompts, AI experiments and
                deployments become impossible to reproduce.
              </p>
              <p>
                <strong>Solution:</strong> Experiment tracking and model
                registries capture every artifact, enabling full reproducibility
                across ML and LLM workloads.
              </p>
            </div>

            <div className="card">
              <h3>Cost & Resource Management</h3>
              <p>
                <strong>Problem:</strong> AI workloads consume expensive compute
                (GPU training) and API costs (LLM tokens) that can spiral
                without visibility.
              </p>
              <p>
                <strong>Solution:</strong> AIOps platforms track resource usage
                across all AI workloads, helping teams optimize costs and
                allocate resources effectively.
              </p>
            </div>
          </div>

          <h2 id="what-is-aiops">What is AIOps?</h2>

          <p>
            Modern AIOps is the operational discipline for all AI applications.
            It unifies the practices previously split across MLOps (for
            traditional ML) and <Link href="/llmops">LLMOps</Link> (for LLM
            applications) into a single framework that covers:
          </p>

          <ul>
            <li>
              <strong>LLMOps / AgentOps:</strong>{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>Tracing</Link>,{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                evaluation
              </Link>
              ,{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/"}>
                prompt management
              </Link>
              , and{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                production monitoring
              </Link>{" "}
              for LLM applications, agents, and RAG systems.
            </li>
            <li>
              <strong>ML Operations:</strong>{" "}
              <Link href={MLFLOW_DOCS_URL + "ml/tracking/"}>
                Experiment tracking
              </Link>
              ,{" "}
              <Link href={MLFLOW_DOCS_URL + "ml/model-registry/"}>
                model registry
              </Link>
              , and model deployment for traditional ML workflows using
              frameworks like scikit-learn, PyTorch, and TensorFlow.
            </li>
            <li>
              <strong>Cross-Cutting Concerns:</strong> Governance, audit trails,
              access control, cost tracking, and compliance that apply to all AI
              workloads regardless of type.
            </li>
          </ul>

          <p>
            The key insight behind modern AIOps is that organizations rarely
            build with just one type of AI. Most teams have a mix of traditional
            ML models, LLM-powered features, and increasingly autonomous agents.
            AIOps provides a unified platform to operationalize all of these,
            preventing tool sprawl and ensuring consistent practices across all
            their AI work.
          </p>

          <p>
            AIOps is closely related to{" "}
            <Link href="/ai-observability">AI observability</Link> (the
            monitoring and understanding subset) and{" "}
            <Link href="/llmops">LLMOps</Link> (the LLM-specific subset). AIOps
            is the broadest term, encompassing both and adding experiment
            management, model versioning, and unified deployment.
          </p>

          <h2 id="key-capabilities">Key AIOps Capabilities</h2>

          <p>
            A comprehensive AIOps platform combines capabilities for both
            LLMOps/AgentOps and traditional ML workloads:
          </p>

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
              latency) for debugging and monitoring.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Evaluation
              </Link>
              : Assess AI output quality using{" "}
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}
                style={{ color: "#007bff" }}
              >
                LLM judges
              </Link>
              , custom scorers, and traditional metrics across all workload
              types.
            </li>
            <li>
              <Link
                href={MLFLOW_DOCS_URL + "ml/tracking/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Experiment Tracking
              </Link>
              : Log parameters, metrics, and artifacts for ML experiments and
              LLM development, enabling comparison and reproducibility.
            </li>
            <li>
              <Link
                href={MLFLOW_DOCS_URL + "ml/model-registry/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Model Registry
              </Link>
              : Version, stage, and manage the lifecycle of ML models and LLM
              configurations in a centralized registry.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Prompt Management
              </Link>
              : Version-control prompt templates, track production usage, and
              enable safe rollbacks for LLM applications.
            </li>
            <li>
              <Link
                href="/genai/ai-gateway"
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                AI Gateway
              </Link>
              : Route requests across LLM providers through a single endpoint
              with unified authentication, rate limiting, and fallback routing.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Production Monitoring
              </Link>
              : Track quality scores, error rates, costs, and drift over time to
              catch regressions across all AI workloads.
            </li>
          </ul>

          <h2 id="how-to-implement">How to Implement AIOps</h2>

          <p>
            <Link href="/genai">MLflow</Link> provides a complete, open-source
            AIOps platform. Here's how teams use MLflow across different AI
            workloads:
          </p>

          <p style={{ marginTop: "32px", marginBottom: "0px" }}>
            <strong>LLM Tracing</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton code={TRACING_CODE} />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={TRACING_CODE}
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
            <strong>Evaluation with LLM Judges</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton code={EVAL_CODE} />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={EVAL_CODE}
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
            <strong>ML Experiment Tracking</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton code={EXPERIMENT_CODE} />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={EXPERIMENT_CODE}
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

          <div style={{ margin: "40px 0", textAlign: "center" }}>
            <img
              src={ObservabilityHero}
              alt="MLflow UI showing traced AI operations with full execution context for AIOps workflows"
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
              MLflow provides unified visibility across all AI operations: LLM
              tracing, evaluation, and experiment tracking
            </p>
          </div>

          <div className="info-box">
            <p>
              <Link href="/genai" style={{ color: "#007bff" }}>
                <strong>MLflow</strong>
              </Link>{" "}
              is the largest open-source AI platform, with over 30 million
              monthly downloads. Backed by the Linux Foundation and licensed
              under Apache 2.0, it provides a complete AIOps stack with no
              vendor lock-in.{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL}>Get started →</Link>
            </p>
          </div>

          <h2>Open Source vs. Proprietary AIOps</h2>

          <p>
            When choosing an AIOps platform, the decision between open source
            and proprietary SaaS tools has significant long-term implications
            for your team, infrastructure, and data ownership.
          </p>

          <p>
            <strong>
              Open Source (<Link href="/genai">MLflow</Link>):
            </strong>{" "}
            With MLflow, you maintain complete control over your AIOps
            infrastructure and data. Deploy on your own infrastructure or use
            managed versions on Databricks, AWS, or other platforms. There are
            no per-seat fees, no usage limits, and no vendor lock-in. MLflow
            supports any AI framework, from scikit-learn and PyTorch to OpenAI
            and LangChain, under a single platform.
          </p>

          <p>
            <strong>Proprietary SaaS Tools:</strong> Commercial AIOps platforms
            offer convenience but at the cost of flexibility and control. They
            typically charge per seat or per usage volume, which can become
            expensive at scale. Your data is sent to their servers, raising
            privacy and compliance concerns. Most proprietary tools specialize
            in either ML or LLM workloads, not both, leading to tool sprawl.
          </p>

          <p>
            <strong>Why Teams Choose Open Source:</strong> Organizations
            building production AI applications increasingly choose MLflow
            because it offers production-ready AIOps for both ML and LLM
            workloads without giving up control of their data, cost
            predictability, or flexibility. The Apache 2.0 license and Linux
            Foundation backing ensure MLflow remains truly open and
            community-driven.
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
              <Link href="/llmops">LLMOps Guide</Link>
            </li>
            <li>
              <Link href="/ai-observability">AI Observability Guide</Link>
            </li>
            <li>
              <Link href="/llm-tracing">LLM Tracing Guide</Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
                LLM Tracing Documentation
              </Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                LLM Evaluation Guide
              </Link>
            </li>
            <li>
              <Link href={MLFLOW_DOCS_URL + "ml/tracking/"}>
                Experiment Tracking Documentation
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

        <ArticleSidebar />
        <SocialLinksFooter />
      </div>
    </>
  );
}
