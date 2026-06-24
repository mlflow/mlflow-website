import { useState } from "react";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../components/Header/Header";
import { SocialLinksFooter } from "../components/SocialLinksFooter/SocialLinksFooter";
import { ArticleSidebar } from "../components/ArticleSidebar/ArticleSidebar";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import ObservabilityHero from "@site/static/img/GenAI_observability/GenAI_observability_hero.png";
import { CopyButton } from "../components/CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";

const SEO_TITLE =
  "LLM Optimization: Reduce Costs & Improve Quality | MLflow AI Platform";
const SEO_DESCRIPTION =
  "Learn how to optimize LLM applications for cost, latency, and quality. Use MLflow's open-source tracing, evaluation, and prompt optimization to systematically improve LLM performance.";

const faqs: {
  question: string;
  answer: React.ReactNode;
  answerText?: string;
}[] = [
  {
    question: "What is LLM optimization?",
    answer:
      "LLM optimization is the practice of systematically improving LLM applications and agents across dimensions like quality, cost, and latency. It includes techniques like prompt optimization, model selection, token usage reduction, caching, evaluation-driven iteration, and production monitoring to ensure LLM applications and agents perform well and cost-effectively at scale.",
  },
  {
    question: "How do I reduce LLM API costs with MLflow?",
    answer: (
      <>
        The most effective way to reduce LLM API costs is to gain visibility
        into where tokens are being spent.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>MLflow Tracing</Link>{" "}
        captures token counts per span, so you can identify expensive
        operations, redundant LLM calls, and oversized prompts. From there, you
        can apply targeted optimizations: shorten prompts, cache repeated
        queries, use smaller models for simpler tasks, or route through an{" "}
        <Link href="/ai-gateway">AI Gateway</Link> with rate limiting and
        fallback routing.
      </>
    ),
    answerText:
      "The most effective way to reduce LLM API costs is to gain visibility into where tokens are being spent. MLflow Tracing captures token counts per span, so you can identify expensive operations, redundant LLM calls, and oversized prompts. From there, you can apply targeted optimizations: shorten prompts, cache repeated queries, use smaller models for simpler tasks, or route through an AI Gateway with rate limiting and fallback routing.",
  },
  {
    question: "How do I improve LLM response quality with MLflow?",
    answer: (
      <>
        Improving LLM response quality requires measurement and iteration.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          MLflow Evaluation
        </Link>{" "}
        lets you score outputs with LLM judges across dimensions like
        correctness, relevance, safety, and groundedness. Once you have a
        quality baseline, you can improve quality through{" "}
        <Link href="/prompt-optimization">prompt optimization</Link>, better
        retrieval pipelines for RAG, or model upgrades, and measure the impact
        of each change.
      </>
    ),
    answerText:
      "Improving LLM response quality requires measurement and iteration. MLflow Evaluation lets you score outputs with LLM judges across dimensions like correctness, relevance, safety, and groundedness. Once you have a quality baseline, you can improve quality through prompt optimization, better retrieval pipelines for RAG, or model upgrades, and measure the impact of each change.",
  },
  {
    question: "How do I reduce LLM latency with MLflow?",
    answer: (
      <>
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>MLflow Tracing</Link>{" "}
        captures latency per span in your LLM pipeline, making it easy to find
        bottlenecks. Common latency optimizations include using streaming
        responses, caching frequent queries, parallelizing independent LLM
        calls, using smaller or faster models for non-critical steps, and
        reducing prompt length to decrease time-to-first-token.
      </>
    ),
    answerText:
      "MLflow Tracing captures latency per span in your LLM pipeline, making it easy to find bottlenecks. Common latency optimizations include using streaming responses, caching frequent queries, parallelizing independent LLM calls, using smaller or faster models for non-critical steps, and reducing prompt length to decrease time-to-first-token.",
  },
  {
    question: "What is prompt optimization in MLflow and how does it work?",
    answer: (
      <>
        <Link href="/prompt-optimization">Prompt optimization</Link> automates
        the process of improving prompts using data-driven algorithms instead of
        manual trial-and-error. Optimizers like GEPA evaluate prompts across
        training examples, analyze failure patterns, generate improved variants,
        and repeat until quality converges. MLflow provides a unified{" "}
        <Link
          href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/optimize-prompts/"}
        >
          prompt optimization API
        </Link>{" "}
        that tracks every version and metric automatically.
      </>
    ),
    answerText:
      "Prompt optimization automates the process of improving prompts using data-driven algorithms instead of manual trial-and-error. Optimizers like GEPA evaluate prompts across training examples, analyze failure patterns, generate improved variants, and repeat until quality converges. MLflow provides a unified prompt optimization API that tracks every version and metric automatically.",
  },
  {
    question:
      "How do I optimize a RAG (Retrieval-Augmented Generation) pipeline?",
    answer:
      "Optimizing a RAG pipeline involves improving both the retrieval and generation stages. Use MLflow Tracing to see exactly what documents are retrieved and how they affect the LLM's response. Use MLflow Evaluation with groundedness and relevance judges to measure retrieval quality. Then iterate on your chunking strategy, embedding model, retrieval parameters, and generation prompts, measuring the impact of each change.",
  },
  {
    question: "How do I optimize an AI agent with MLflow?",
    answer: (
      <>
        Agent optimization requires visibility into every reasoning step, tool
        call, and LLM invocation.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>MLflow Tracing</Link>{" "}
        captures the full execution graph so you can identify unnecessary tool
        calls, redundant reasoning loops, and expensive LLM invocations.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          MLflow Evaluation
        </Link>{" "}
        lets you assess agent decision-making quality with LLM judges, and{" "}
        <Link href="/prompt-optimization">prompt optimization</Link> can improve
        the agent's system prompts algorithmically.
      </>
    ),
    answerText:
      "Agent optimization requires visibility into every reasoning step, tool call, and LLM invocation. MLflow Tracing captures the full execution graph so you can identify unnecessary tool calls, redundant reasoning loops, and expensive LLM invocations. MLflow Evaluation lets you assess agent decision-making quality with LLM judges, and prompt optimization can improve the agent's system prompts algorithmically.",
  },
  {
    question: "What is the best tool for LLM optimization?",
    answer:
      "The best tool for LLM optimization depends on your needs. MLflow is the leading open-source option, providing the complete toolkit: tracing for cost and latency visibility, evaluation for quality measurement, prompt optimization for algorithmic improvement, and an AI Gateway for cost management, compliance, and governance. Unlike proprietary tools, MLflow is 100% free, supports any LLM provider and agent framework, and is backed by the Linux Foundation with over 30 million monthly downloads.",
  },
  {
    question: "How do I measure LLM performance with MLflow?",
    answer: (
      <>
        LLM performance is measured across multiple dimensions: quality (using{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          LLM judge scorers
        </Link>{" "}
        for correctness, relevance, safety, etc.), cost (token usage per
        request), and latency (response time per span).{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>MLflow Tracing</Link>{" "}
        captures cost and latency automatically, while{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          MLflow Evaluation
        </Link>{" "}
        provides automated quality scoring with 70+ built-in judges.
      </>
    ),
    answerText:
      "LLM performance is measured across multiple dimensions: quality (using LLM judge scorers for correctness, relevance, safety, etc.), cost (token usage per request), and latency (response time per span). MLflow Tracing captures cost and latency automatically, while MLflow Evaluation provides automated quality scoring with 70+ built-in judges.",
  },
  {
    question: "Is MLflow free for LLM optimization?",
    answer:
      "Yes. MLflow is 100% open source under the Apache 2.0 license, backed by the Linux Foundation. You can use all optimization features (tracing, evaluation, prompt optimization, AI Gateway) for free, including in commercial applications. There are no per-seat fees, no usage limits, and no vendor lock-in.",
  },
  {
    question: "How do I get started with LLM optimization using MLflow?",
    answer: (
      <>
        Start by enabling{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/quickstart/"}>
          MLflow Tracing
        </Link>{" "}
        with a single line of code to capture token usage, latency, and
        execution details for every LLM call. This gives you a baseline. Then
        use{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          MLflow Evaluation
        </Link>{" "}
        to measure output quality. Once you can see and measure performance,
        apply targeted optimizations: shorten prompts, optimize retrieval,
        adjust model selection, or run{" "}
        <Link href="/prompt-optimization">automated prompt optimization</Link>.
      </>
    ),
    answerText:
      "Start by enabling MLflow Tracing with a single line of code to capture token usage, latency, and execution details for every LLM call. This gives you a baseline. Then use MLflow Evaluation to measure output quality. Once you can see and measure performance, apply targeted optimizations: shorten prompts, optimize retrieval, adjust model selection, or run automated prompt optimization.",
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
    "Open-source platform for optimizing LLM applications with tracing, evaluation, prompt optimization, and cost management, compliance, and governance.",
  url: "https://mlflow.org",
  license: "https://www.apache.org/licenses/LICENSE-2.0",
};

const TRACING_CODE = `import mlflow
from openai import OpenAI

# Enable automatic tracing for OpenAI
mlflow.openai.autolog()

# Every LLM call is now traced with token counts,
# latency, prompts, and responses
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Summarize MLflow"}],
)

# Search traces to analyze cost and latency patterns
traces = mlflow.search_traces(experiment_ids=["1"])
for trace in traces:
    print(f"Tokens: {trace.info.total_tokens}")
    print(f"Latency: {trace.info.execution_duration_ms}ms")`;

const EVAL_CODE = `import mlflow
from mlflow.genai.scorers import (
    Correctness,
    RelevanceToInput,
    Safety,
)

# Evaluate LLM outputs with built-in judges
results = mlflow.genai.evaluate(
    data=mlflow.search_traces(experiment_ids=["1"]),
    scorers=[
        Correctness(),    # Are responses factually correct?
        RelevanceToInput(),  # Are responses relevant to the query?
        Safety(),         # Are responses free of harmful content?
    ],
)

# View results in the MLflow UI or programmatically
print(results.tables["eval_results"])`;

export default function LLMOptimization() {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://mlflow.org/llm-optimization" />
        <link rel="canonical" href="https://mlflow.org/llm-optimization" />
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
          <h1>LLM Optimization</h1>

          <p>
            LLM optimization is the practice of systematically improving LLM
            applications and agents across dimensions like quality, cost, and
            latency. For single-turn LLM applications, optimization typically
            focuses on prompt quality, model selection, and token efficiency.
            Agent optimization goes further: because agents make multiple LLM
            calls, invoke tools, and follow multi-step reasoning chains, they
            introduce additional challenges around debugging complex execution
            paths, reducing compounding latency, and controlling costs that
            scale with each reasoning step. Both require specialized tooling
            because LLM behavior is non-deterministic, costs scale with token
            usage, and quality can only be measured with semantic evaluation
            rather than unit tests.
          </p>

          <p>
            Effective LLM optimization starts with visibility.{" "}
            <Link href="/llm-tracing">Tracing</Link> captures token counts,
            latency, and execution details for every LLM call, revealing where
            costs and bottlenecks are.{" "}
            <Link href="/llm-evaluation">Evaluation</Link> measures output
            quality with LLM judges, providing a baseline to track whether
            changes actually improve performance.{" "}
            <Link href="/prompt-optimization">Prompt optimization</Link>{" "}
            automates the process of improving prompts algorithmically,
            replacing manual trial-and-error with systematic, data-driven
            iteration.
          </p>

          <p>
            <Link href="/genai">MLflow</Link> provides the complete open-source
            toolkit for LLM optimization:{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>tracing</Link> for
            cost and latency visibility,{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
              evaluation
            </Link>{" "}
            for quality measurement,{" "}
            <Link
              href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/optimize-prompts/"}
            >
              prompt optimization
            </Link>{" "}
            for algorithmic prompt improvement, and an{" "}
            <Link href="/ai-gateway">AI Gateway</Link> for cost management,
            compliance, and governance across LLM providers.
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
                <a href="#why-llm-optimization-matters">
                  Why LLM Optimization Matters
                </a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#optimization-strategies">
                  LLM Optimization Techniques
                </a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#common-use-cases">Common Use Cases</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#how-to-implement">
                  How to Implement LLM Optimization
                </a>
              </li>
              <li style={{ marginBottom: "0" }}>
                <a href="#faq">FAQ</a>
              </li>
            </ul>
          </div>

          <h2 id="why-llm-optimization-matters">
            Why LLM Optimization Matters
          </h2>

          <p>
            LLM applications face unique optimization challenges that
            traditional software profiling and testing can't address:
          </p>

          <div className="grid-2">
            <div className="card">
              <h3>Runaway Costs</h3>
              <p>
                <strong>Problem:</strong> Token costs scale with usage, and
                multi-step agents can make dozens of LLM calls per request.
                Without visibility, API bills grow unpredictably.
              </p>
              <p>
                <strong>Solution:</strong>{" "}
                <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>Tracing</Link>{" "}
                captures token usage per span so you can find expensive
                operations, and an <Link href="/ai-gateway">AI Gateway</Link>{" "}
                enforces rate limits and budget controls across providers.
              </p>
            </div>

            <div className="card">
              <h3>Quality & Reliability</h3>
              <p>
                <strong>Problem:</strong> LLM applications can produce
                hallucinations, irrelevant responses, and degraded outputs that
                undermine user trust, and traditional testing can't catch these
                issues.
              </p>
              <p>
                <strong>Solution:</strong>{" "}
                <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                  Evaluation with LLM judges
                </Link>{" "}
                continuously assesses quality dimensions like correctness,
                relevance, and safety across every response, catching
                regressions before users report them.
              </p>
            </div>

            <div className="card">
              <h3>Slow Response Times</h3>
              <p>
                <strong>Problem:</strong> LLM calls add hundreds of milliseconds
                to seconds of latency per request. Agent workflows with multiple
                sequential calls compound this significantly.
              </p>
              <p>
                <strong>Solution:</strong>{" "}
                <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
                  Per-span latency tracing
                </Link>{" "}
                identifies bottlenecks so you can parallelize calls, cache
                repeated queries, or use faster models for non-critical steps.
              </p>
            </div>

            <div className="card">
              <h3>Inefficient Iteration</h3>
              <p>
                <strong>Problem:</strong> Manual prompt engineering is slow,
                inconsistent, and plateaus quickly. Engineers can't
                systematically identify which prompt changes improve quality.
              </p>
              <p>
                <strong>Solution:</strong>{" "}
                <Link href="/prompt-optimization">
                  Automated prompt optimization
                </Link>{" "}
                uses algorithms to systematically improve prompts across
                hundreds of examples, replacing guesswork with measured
                improvement.
              </p>
            </div>
          </div>

          <h2 id="optimization-strategies">LLM Optimization Techniques</h2>

          <p>
            LLM optimization spans several complementary strategies. The most
            effective approach combines visibility (tracing), measurement
            (evaluation), and systematic improvement (prompt optimization):
          </p>

          <ul>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "tracing/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Cost Optimization with Tracing
              </Link>
              : Capture token counts per span to identify the most expensive
              operations in your pipeline. Common wins include shortening
              verbose prompts, eliminating redundant LLM calls in agent loops,
              routing simple queries to smaller models, and caching repeated
              requests.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Quality Optimization with Evaluation
              </Link>
              : Use LLM judges to measure output quality across dimensions like
              correctness, relevance, safety, and groundedness. Establish a
              quality baseline, then measure the impact of each change.
              Automated evaluation replaces subjective spot-checking with
              systematic measurement.
            </li>
            <li>
              <Link
                href={
                  MLFLOW_GENAI_DOCS_URL + "prompt-registry/optimize-prompts/"
                }
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Prompt Optimization
              </Link>
              : Replace manual prompt engineering with algorithms that
              systematically improve prompts across training data. Optimizers
              like GEPA analyze failure patterns, generate improved variants,
              and select the best performer automatically.
            </li>
            <li>
              <Link
                href="/ai-gateway"
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Cost Governance with AI Gateway
              </Link>
              : Route LLM requests through a centralized gateway with rate
              limiting, budget controls, fallback routing, and unified
              credential management. Track usage and costs across all providers
              from a single dashboard.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Production Monitoring
              </Link>
              : Run LLM judges continuously against production traces to catch
              quality regressions, cost spikes, and latency degradation before
              users report them.
            </li>
          </ul>

          <h2 id="common-use-cases">Common Use Cases for LLM Optimization</h2>

          <p>
            LLM optimization applies across the full lifecycle of LLM
            applications:
          </p>

          <ul>
            <li>
              <strong>Reducing agent costs:</strong> Multi-step agents can make
              many LLM calls per request. Use{" "}
              <Link href="/llm-tracing">tracing</Link> to identify unnecessary
              reasoning loops and redundant tool calls, then optimize the
              agent's prompts and logic to reduce token usage.
            </li>
            <li>
              <strong>Improving RAG quality:</strong> Retrieval-augmented
              generation pipelines depend on both retrieval and generation
              quality. Use <Link href="/llm-evaluation">evaluation</Link> with
              groundedness and relevance judges to measure end-to-end quality,
              then iterate on chunking, embedding, and generation prompts.
            </li>
            <li>
              <strong>Optimizing prompts at scale:</strong> Instead of manually
              tweaking prompts for each feature, use{" "}
              <Link href="/prompt-optimization">prompt optimization</Link> to
              algorithmically improve prompts across hundreds of examples with
              full tracking and versioning.
            </li>
            <li>
              <strong>Model selection and routing:</strong> Not all queries need
              the most expensive model. Use tracing and evaluation to identify
              which queries can be routed to cheaper, faster models without
              sacrificing quality, and implement routing through the{" "}
              <Link href="/ai-gateway">AI Gateway</Link>.
            </li>
            <li>
              <strong>Latency optimization:</strong> Use{" "}
              <Link href="/llm-tracing">per-span latency tracing</Link> to find
              bottlenecks, then apply streaming, caching, parallelization, or
              model downsizing to reduce response times.
            </li>
            <li>
              <strong>Production quality monitoring:</strong> Run{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                LLM judges continuously
              </Link>{" "}
              against production traces to detect quality regressions,
              hallucination spikes, and cost anomalies before they impact users.
            </li>
          </ul>

          <h2 id="how-to-implement">How to Implement LLM Optimization</h2>

          <p>
            <Link href="/genai">MLflow</Link> provides the complete open-source
            toolkit for LLM optimization. Start with tracing to gain visibility,
            add evaluation to measure quality, then apply targeted optimizations
            based on what the data shows.
          </p>

          <p style={{ marginTop: "32px", marginBottom: "0px" }}>
            <strong>
              Step 1: Enable tracing for cost and latency visibility
            </strong>
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
            <strong>Step 2: Evaluate quality with LLM judges</strong>
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

          <p>
            With tracing and evaluation in place, you have the visibility needed
            to apply targeted optimizations: shorten prompts to reduce costs,
            run{" "}
            <Link href="/prompt-optimization">
              automated prompt optimization
            </Link>{" "}
            to improve quality, route through the{" "}
            <Link href="/ai-gateway">AI Gateway</Link> for cost management,
            compliance, and governance, and monitor production quality with
            continuous evaluation.
          </p>

          <div style={{ margin: "32px 0", textAlign: "center" }}>
            <img
              src={ObservabilityHero}
              alt="MLflow UI showing traced LLM calls with token counts, latency, and execution details for optimization"
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
              MLflow captures traces with token counts, latency, and full
              execution context for every LLM call
            </p>
          </div>

          <div className="info-box">
            <p>
              <Link href="/genai" style={{ color: "#007bff" }}>
                <strong>MLflow</strong>
              </Link>{" "}
              is the largest open-source{" "}
              <strong>AI engineering platform</strong>, with over 30 million
              monthly downloads. Thousands of organizations use MLflow to debug,
              evaluate, monitor, and optimize production-quality AI agents and
              LLM applications while controlling costs and managing access to
              models and data. Backed by the Linux Foundation and licensed under
              Apache 2.0, MLflow provides a complete LLM optimization toolkit
              with no vendor lock-in.{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL}>Get started &rarr;</Link>
            </p>
          </div>

          <h2>Open Source vs. Proprietary LLM Optimization</h2>

          <p>
            When choosing tools for LLM optimization, the decision between open
            source and proprietary platforms has significant implications for
            cost, flexibility, and data ownership.
          </p>

          <p>
            <strong>
              Open Source (<Link href="/genai">MLflow</Link>):
            </strong>{" "}
            With MLflow, you maintain complete control over your optimization
            data and infrastructure. Trace data, evaluation results, and prompt
            versions stay on your own systems. There are no per-seat fees, no
            usage limits, and no vendor lock-in. MLflow integrates with any LLM
            provider and agent framework through OpenTelemetry-compatible
            tracing.
          </p>

          <p>
            <strong>Proprietary SaaS Tools:</strong> Commercial optimization and
            observability platforms offer convenience but at the cost of
            flexibility and control. They typically charge per seat or per trace
            volume, which becomes expensive at scale. Your trace data and
            evaluation results are sent to their servers, raising privacy and
            compliance concerns. You're locked into their ecosystem, making it
            difficult to switch providers or customize workflows.
          </p>

          <p>
            <strong>Why Teams Choose Open Source:</strong> Organizations
            optimizing LLM applications at scale choose MLflow because it
            provides production-grade tracing, evaluation, prompt optimization,
            and cost management, compliance, and governance without giving up
            control of their data, cost predictability, or flexibility. The
            Apache 2.0 license and Linux Foundation backing ensure MLflow
            remains truly open and community-driven.
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
                LLM Evaluation Guide
              </Link>
            </li>
            <li>
              <Link
                href={
                  MLFLOW_GENAI_DOCS_URL + "prompt-registry/optimize-prompts/"
                }
              >
                Prompt Optimization Documentation
              </Link>
            </li>
            <li>
              <Link href="/ai-observability">AI Observability Guide</Link>
            </li>
            <li>
              <Link href="/llm-tracing">LLM Tracing Guide</Link>
            </li>
            <li>
              <Link href="/llm-evaluation">LLM Evaluation Guide</Link>
            </li>
            <li>
              <Link href="/ai-gateway">AI Gateway Guide</Link>
            </li>
            <li>
              <Link href="/prompt-optimization">Prompt Optimization Guide</Link>
            </li>
            <li>
              <Link href="/llmops">LLMOps Guide</Link>
            </li>
            <li>
              <Link href="/genai">MLflow for Agents and LLMs Overview</Link>
            </li>
          </ul>
        </div>

        <ArticleSidebar />
        <SocialLinksFooter />
      </div>
    </>
  );
}
