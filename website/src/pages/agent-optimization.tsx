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

const SEO_TITLE =
  "Agent Optimization: Debug, Evaluate & Improve AI Agents | MLflow";
const SEO_DESCRIPTION =
  "Learn how to optimize AI agents for quality, cost, and latency. Use MLflow's open-source tracing, evaluation, and prompt optimization to debug and improve agent performance.";

const faqs: {
  question: string;
  answer: React.ReactNode;
  answerText?: string;
}[] = [
  {
    question: "What is agent optimization?",
    answer:
      "Agent optimization is the practice of improving AI agents across quality, cost, and latency. Unlike optimizing a single LLM call, agent optimization addresses multi-step reasoning chains, tool selection, execution paths, and compounding costs. It includes techniques like tracing agent execution, evaluating decision-making quality with LLM judges, optimizing system prompts with algorithms, and monitoring agent reliability in production.",
  },
  {
    question: "How does prompt optimization help improve AI agents?",
    answer: (
      <>
        Agent behavior is heavily influenced by system prompts, tool
        descriptions, and few-shot examples.{" "}
        <Link href="/prompt-optimization">Prompt optimization</Link> automates
        the process of improving these prompts using data-driven algorithms
        instead of manual trial-and-error. Optimizers like GEPA evaluate agent
        prompts across training examples, analyze failure patterns, generate
        improved variants, and repeat until quality converges. MLflow provides a
        unified{" "}
        <Link
          href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/optimize-prompts/"}
        >
          prompt optimization API
        </Link>{" "}
        that tracks every version and metric automatically.
      </>
    ),
    answerText:
      "Agent behavior is heavily influenced by system prompts, tool descriptions, and few-shot examples. Prompt optimization automates the process of improving these prompts using data-driven algorithms instead of manual trial-and-error. Optimizers like GEPA evaluate agent prompts across training examples, analyze failure patterns, generate improved variants, and repeat until quality converges. MLflow provides a unified prompt optimization API that tracks every version and metric automatically.",
  },
  {
    question:
      "How do I optimize an agent with RAG (Retrieval-Augmented Generation)?",
    answer:
      "Optimizing a RAG agent involves improving both the retrieval and generation stages within the agent's execution. Use MLflow Tracing to see exactly what documents the agent retrieves and how they influence its reasoning and responses. Use MLflow Evaluation with groundedness and relevance judges to measure retrieval quality and answer accuracy. Then iterate on the agent's retrieval strategy, chunking approach, embedding model, and generation prompts, measuring the impact of each change across the full agent pipeline.",
  },
  {
    question: "What is the best tool for AI agent optimization?",
    answer:
      "The best tool for agent optimization depends on your needs. MLflow is the leading open-source option, providing the complete toolkit: tracing for full execution visibility across reasoning steps and tool calls, evaluation for quality measurement with LLM judges, prompt optimization for algorithmic improvement of system prompts, and an AI Gateway for cost management, compliance, and governance. Unlike proprietary tools, MLflow is 100% free, supports any LLM provider and agent framework (LangGraph, CrewAI, OpenAI Agents, and more), and is backed by the Linux Foundation with over 30 million monthly downloads.",
  },
  {
    question: "How do I reduce AI agent costs with MLflow?",
    answer: (
      <>
        Agents make multiple LLM calls per request, so costs compound quickly.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>MLflow Tracing</Link>{" "}
        captures token counts per span across every reasoning step and tool
        call, revealing where tokens are being spent. From there, you can
        eliminate redundant reasoning loops, use smaller models for simple
        sub-tasks, cache repeated tool calls, shorten system prompts, or route
        through an <Link href="/ai-gateway">AI Gateway</Link> with rate limiting
        and budget controls.
      </>
    ),
    answerText:
      "Agents make multiple LLM calls per request, so costs compound quickly. MLflow Tracing captures token counts per span across every reasoning step and tool call, revealing where tokens are being spent. From there, you can eliminate redundant reasoning loops, use smaller models for simple sub-tasks, cache repeated tool calls, shorten system prompts, or route through an AI Gateway with rate limiting and budget controls.",
  },
  {
    question: "How do I improve agent response quality with MLflow?",
    answer: (
      <>
        Agent quality depends on correct reasoning, appropriate tool selection,
        and accurate final responses.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          MLflow Evaluation
        </Link>{" "}
        lets you score agent outputs with LLM judges across dimensions like
        correctness, relevance, safety, and tool use quality. Once you have a
        quality baseline, improve through{" "}
        <Link href="/prompt-optimization">prompt optimization</Link> of system
        prompts, better tool descriptions, retrieval improvements, or model
        upgrades, and measure the impact of each change.
      </>
    ),
    answerText:
      "Agent quality depends on correct reasoning, appropriate tool selection, and accurate final responses. MLflow Evaluation lets you score agent outputs with LLM judges across dimensions like correctness, relevance, safety, and tool use quality. Once you have a quality baseline, improve through prompt optimization of system prompts, better tool descriptions, retrieval improvements, or model upgrades, and measure the impact of each change.",
  },
  {
    question: "How do I reduce agent latency with MLflow?",
    answer: (
      <>
        Agents suffer from compounding latency because each reasoning step and
        tool call adds time.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>MLflow Tracing</Link>{" "}
        captures per-span latency across the full execution graph, making it
        easy to find bottlenecks. Common optimizations include parallelizing
        independent tool calls, caching repeated operations, using faster models
        for lower-complexity reasoning steps, reducing the number of reasoning
        loops, and streaming intermediate results.
      </>
    ),
    answerText:
      "Agents suffer from compounding latency because each reasoning step and tool call adds time. MLflow Tracing captures per-span latency across the full execution graph, making it easy to find bottlenecks. Common optimizations include parallelizing independent tool calls, caching repeated operations, using faster models for lower-complexity reasoning steps, reducing the number of reasoning loops, and streaming intermediate results.",
  },
  {
    question: "How do I debug a multi-step AI agent with MLflow?",
    answer: (
      <>
        Multi-step agents are difficult to debug because failures can occur at
        any reasoning step, tool call, or decision point.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>MLflow Tracing</Link>{" "}
        captures the full execution graph, including every LLM invocation, tool
        call, input, output, and intermediate reasoning step. This lets you
        pinpoint exactly where an agent went wrong: a bad tool selection, a
        hallucinated reasoning step, an unnecessary loop, or an incorrect final
        synthesis.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          MLflow Evaluation
        </Link>{" "}
        can then assess agent decision-making quality across many examples.
      </>
    ),
    answerText:
      "Multi-step agents are difficult to debug because failures can occur at any reasoning step, tool call, or decision point. MLflow Tracing captures the full execution graph, including every LLM invocation, tool call, input, output, and intermediate reasoning step. This lets you pinpoint exactly where an agent went wrong: a bad tool selection, a hallucinated reasoning step, an unnecessary loop, or an incorrect final synthesis. MLflow Evaluation can then assess agent decision-making quality across many examples.",
  },
  {
    question: "How do I measure AI agent performance with MLflow?",
    answer: (
      <>
        Agent performance is measured across multiple dimensions: quality (using{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          LLM judge scorers
        </Link>{" "}
        for correctness, relevance, tool use quality, and safety), cost (total
        token usage across all reasoning steps), and latency (end-to-end
        response time including tool calls).{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>MLflow Tracing</Link>{" "}
        captures cost and latency automatically across the full agent execution
        graph, while{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          MLflow Evaluation
        </Link>{" "}
        provides automated quality scoring with 70+ built-in judges.
      </>
    ),
    answerText:
      "Agent performance is measured across multiple dimensions: quality (using LLM judge scorers for correctness, relevance, tool use quality, and safety), cost (total token usage across all reasoning steps), and latency (end-to-end response time including tool calls). MLflow Tracing captures cost and latency automatically across the full agent execution graph, while MLflow Evaluation provides automated quality scoring with 70+ built-in judges.",
  },
  {
    question: "Is MLflow free for agent optimization?",
    answer:
      "Yes. MLflow is 100% open source under the Apache 2.0 license, backed by the Linux Foundation. You can use all agent optimization features (tracing, evaluation, prompt optimization, AI Gateway) for free, including in commercial applications. There are no per-seat fees, no usage limits, and no vendor lock-in.",
  },
  {
    question: "How do I get started with agent optimization using MLflow?",
    answer: (
      <>
        Start by enabling{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/quickstart/"}>
          MLflow Tracing
        </Link>{" "}
        with a single line of code to capture the full execution graph for every
        agent invocation: reasoning steps, tool calls, token usage, and latency.
        This gives you a baseline. Then use{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          MLflow Evaluation
        </Link>{" "}
        to measure agent output quality with LLM judges. Once you can see and
        measure performance, apply targeted optimizations: improve system
        prompts with{" "}
        <Link href="/prompt-optimization">automated prompt optimization</Link>,
        reduce unnecessary tool calls, route sub-tasks to faster models, or
        cache repeated operations.
      </>
    ),
    answerText:
      "Start by enabling MLflow Tracing with a single line of code to capture the full execution graph for every agent invocation: reasoning steps, tool calls, token usage, and latency. This gives you a baseline. Then use MLflow Evaluation to measure agent output quality with LLM judges. Once you can see and measure performance, apply targeted optimizations: improve system prompts with automated prompt optimization, reduce unnecessary tool calls, route sub-tasks to faster models, or cache repeated operations.",
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
    "Open-source platform for optimizing AI agents with tracing, evaluation, prompt optimization, and cost management, compliance, and governance.",
  url: "https://mlflow.org",
  license: "https://www.apache.org/licenses/LICENSE-2.0",
};

const OPTIMIZE_CODE = `import mlflow
from mlflow.genai.optimize import GepaPromptOptimizer
from mlflow.genai.scorers import Correctness

base_prompt = mlflow.genai.register_prompt(
    name="agent-system-prompt",
    template="Answer the question based on the context.\\n\\n"
             "Context: {{ context }}\\n"
             "Question: {{ question }}\\n\\nAnswer:",
)

result = mlflow.genai.optimize_prompts(
    predict_fn=my_predict_fn,
    train_data=train_data,  # labeled examples
    prompt_uris=[base_prompt.uri],
    optimizer=GepaPromptOptimizer(
        reflection_model="openai:/gpt-5.2",
    ),
    scorers=[Correctness()],
)

optimized = mlflow.genai.load_prompt(result.optimized_prompts[0].uri)
print(optimized.template)`;

export default function AgentOptimization() {
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
          content="https://mlflow.org/agent-optimization"
        />
        <link rel="canonical" href="https://mlflow.org/agent-optimization" />
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
          <h1>Agent Optimization</h1>

          <p>
            Agent optimization is the practice of improving AI agents across
            quality, cost, and latency. Unlike optimizing a single LLM call,
            agents introduce unique challenges: they make multiple LLM calls per
            request, invoke tools, follow multi-step reasoning chains, and
            produce compounding costs and latency at every step. Debugging why
            an agent chose the wrong tool, entered a redundant reasoning loop,
            or hallucinated an intermediate step requires specialized tooling
            that captures the full execution graph and evaluates decision-making
            quality at every level.
          </p>

          <p>
            Effective agent optimization starts with visibility.{" "}
            <Link href="/llm-tracing">Tracing</Link> captures the full execution
            graph for every agent invocation: reasoning steps, tool calls, token
            counts, and latency per span, revealing exactly where costs,
            bottlenecks, and failures occur.{" "}
            <Link href="/llm-evaluation">Evaluation</Link> measures agent output
            quality with LLM judges, providing a baseline to track whether
            changes actually improve performance.{" "}
            <Link href="/prompt-optimization">Prompt optimization</Link>{" "}
            automates the process of improving agent system prompts with
            algorithms, replacing manual trial-and-error with data-driven
            iteration.
          </p>

          <p>
            <Link href="/genai">MLflow</Link> provides the complete open-source
            toolkit for agent optimization:{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>tracing</Link> for
            full execution visibility,{" "}
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
                <a href="#why-agent-optimization-matters">
                  Why Agent Optimization Matters
                </a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#optimization-strategies">
                  Agent Optimization Techniques
                </a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#common-use-cases">Common Use Cases</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#how-to-implement">
                  How to Implement Agent Optimization
                </a>
              </li>
              <li style={{ marginBottom: "0" }}>
                <a href="#faq">FAQ</a>
              </li>
            </ul>
          </div>

          <h2 id="why-agent-optimization-matters">
            Why Agent Optimization Matters
          </h2>

          <p>
            AI agents face unique optimization challenges that traditional
            software profiling and testing can't address. Because agents combine
            multi-step reasoning, tool use, and LLM calls into complex execution
            paths, standard debugging and monitoring tools fall short:
          </p>

          <div className="grid-2">
            <div className="card">
              <h3>Compounding Costs</h3>
              <p>
                <strong>Problem:</strong> Agents make multiple LLM calls per
                request across reasoning steps and tool invocations. Token costs
                compound at every step, and without visibility, API bills grow
                unpredictably.
              </p>
              <p>
                <strong>Solution:</strong>{" "}
                <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>Tracing</Link>{" "}
                captures token usage per span across the full agent execution
                graph, and an <Link href="/ai-gateway">AI Gateway</Link>{" "}
                enforces rate limits and budget controls across providers.
              </p>
            </div>

            <div className="card">
              <h3>Unreliable Decision-Making</h3>
              <p>
                <strong>Problem:</strong> Agents can select the wrong tools,
                enter redundant reasoning loops, hallucinate intermediate steps,
                or produce incorrect final responses. Traditional testing can't
                catch these failures.
              </p>
              <p>
                <strong>Solution:</strong>{" "}
                <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                  Evaluation with LLM judges
                </Link>{" "}
                continuously assesses agent quality across correctness,
                relevance, tool use, and safety, catching regressions before
                users report them.
              </p>
            </div>

            <div className="card">
              <h3>Cascading Latency</h3>
              <p>
                <strong>Problem:</strong> Each reasoning step and tool call adds
                latency. Agent workflows with sequential LLM calls and external
                API requests compound delays, leading to slow response times
                that frustrate users.
              </p>
              <p>
                <strong>Solution:</strong>{" "}
                <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
                  Per-span latency tracing
                </Link>{" "}
                across the full execution graph identifies bottlenecks so you
                can parallelize calls, cache repeated operations, or use faster
                models for non-critical steps.
              </p>
            </div>

            <div className="card">
              <h3>Too Many Knobs to Tune</h3>
              <p>
                <strong>Problem:</strong> Agents have many interacting surfaces
                to optimize: system prompts, tool descriptions, model selection
                per step, few-shot examples, and retrieval parameters. Manual
                tuning doesn't scale.
              </p>
              <p>
                <strong>Solution:</strong>{" "}
                <Link href="/prompt-optimization">
                  Automated prompt optimization
                </Link>{" "}
                uses algorithms to improve agent prompts across hundreds of
                examples, replacing manual trial-and-error with data-driven
                iteration.
              </p>
            </div>
          </div>

          <h2 id="optimization-strategies">Agent Optimization Techniques</h2>

          <p>
            Each of the challenges above has a corresponding optimization
            approach. Here's what actually works:
          </p>

          <ul>
            <li>
              <strong>Cut compounding costs</strong>: Use{" "}
              <Link
                href="/llm-tracing"
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                tracing
              </Link>{" "}
              to see how many tokens each reasoning step and tool call uses.
              You'll spot the waste fast: redundant reasoning loops, duplicate
              tool calls, verbose prompts. Remove them, and costs drop. Then set
              up budget policies and alerting through the{" "}
              <Link
                href="/ai-gateway"
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                AI Gateway
              </Link>{" "}
              to catch cost spikes before they eat through your budget.
            </li>
            <li>
              <strong>Fix unreliable decision-making</strong>: Set up{" "}
              <Link
                href="/llm-evaluation"
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                evaluation
              </Link>{" "}
              to score agent outputs with{" "}
              <Link
                href="/llm-as-a-judge"
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                LLM judges
              </Link>{" "}
              and code-based metrics across correctness, relevance, and safety.
              Get a quality baseline, make a change (better prompt, different
              model, improved tool descriptions), re-evaluate, and see if it
              actually helped.
            </li>
            <li>
              <strong>Reduce cascading latency</strong>: Every sequential step
              adds delay. Use{" "}
              <Link
                href="/llm-tracing"
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                per-span latency tracing
              </Link>{" "}
              to find the slowest steps, then parallelize independent tool
              calls, cache repeated lookups, and use lighter models for
              lower-complexity steps.
            </li>
            <li>
              <strong>Automate prompt tuning</strong>: System prompts, tool
              descriptions, few-shot examples. Too many surfaces to tweak by
              hand.{" "}
              <Link
                href="/prompt-optimization"
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Prompt optimization
              </Link>{" "}
              runs your agent across test examples, finds where prompts fall
              short, generates better versions, and picks the best one.
            </li>
          </ul>

          <h2 id="common-use-cases">Common Use Cases for Agent Optimization</h2>

          <p>
            These techniques apply to any agent architecture. Here are the most
            common scenarios where teams see the biggest improvements:
          </p>

          <ul>
            <li>
              <strong>RAG agents:</strong> Agents that retrieve documents before
              generating answers have two places to go wrong: bad retrieval and
              bad generation. Use <Link href="/llm-evaluation">evaluation</Link>{" "}
              with groundedness and relevance judges to figure out which side is
              failing, then improve the retrieval strategy or generation prompts
              accordingly.
            </li>
            <li>
              <strong>Multi-step tool-use agents:</strong> Agents that chain
              together multiple tool calls are prone to picking the wrong tool,
              calling tools unnecessarily, or getting stuck in loops. Use{" "}
              <Link href="/llm-tracing">tracing</Link> to see the full execution
              path and find where the agent goes off track.
            </li>
            <li>
              <strong>Customer-facing chatbots:</strong> When agents talk
              directly to users, quality and latency matter most. Set up{" "}
              <Link href="/llm-evaluation">evaluation</Link> to catch bad
              responses before users see them, and use{" "}
              <Link href="/llm-tracing">latency tracing</Link> to keep response
              times fast.
            </li>
            <li>
              <strong>Agents running at scale:</strong> When you have many
              agents or high request volume, costs add up quickly. Use the{" "}
              <Link href="/ai-gateway">AI Gateway</Link> for budget policies and
              model routing, and run{" "}
              <Link href="/prompt-optimization">prompt optimization</Link> to
              get better results from cheaper models.
            </li>
          </ul>

          <h2 id="how-to-implement">How to Implement Agent Optimization</h2>

          <p>
            <Link href="/genai">MLflow</Link> provides the complete open-source
            toolkit for agent optimization. Start with tracing to gain full
            execution visibility, add evaluation to measure agent quality, then
            apply targeted optimizations based on what the data shows.
          </p>

          <p style={{ marginTop: "32px", marginBottom: "0px" }}>
            <strong>Optimize agent prompts automatically</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton code={OPTIMIZE_CODE} />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={OPTIMIZE_CODE}
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
            <strong>Evaluate agent quality with LLM judges</strong>
          </p>

          <figure
            style={{
              margin: "8px 0 0 0",
              borderRadius: "8px",
              overflow: "hidden",
              border: "1px solid #e5e7eb",
            }}
          >
            <video
              width="100%"
              controls
              autoPlay
              loop
              muted
              playsInline
              style={{
                borderRadius: "8px",
                boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
              }}
            >
              <source
                src="/img/releases/3.9.0/judge_builder_ui.mp4"
                type="video/mp4"
              />
              Your browser does not support the video tag.
            </video>
            <figcaption
              style={{
                marginTop: "12px",
                padding: "0 12px 12px",
                fontSize: "14px",
                color: "#6b7280",
                fontStyle: "italic",
              }}
            >
              Build custom LLM judges in the MLflow UI to score agent traces on
              quality dimensions that matter for your use case
            </figcaption>
          </figure>

          <div className="info-box">
            <p>
              <Link href="/genai" style={{ color: "#007bff" }}>
                <strong>MLflow</strong>
              </Link>{" "}
              is the largest open-source{" "}
              <strong>AI engineering platform</strong>, with over 30 million
              monthly downloads. Thousands of organizations use MLflow to debug,
              evaluate, monitor, and optimize production-quality AI agents while
              controlling costs and managing access to models and data. Backed
              by the Linux Foundation and licensed under Apache 2.0, MLflow
              provides a complete agent optimization toolkit with no vendor
              lock-in.{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL}>Get started &rarr;</Link>
            </p>
          </div>

          <h2>Open Source vs. Proprietary Agent Optimization</h2>

          <p>
            When choosing tools for agent optimization, the decision between
            open source and proprietary platforms has significant implications
            for cost, flexibility, and data ownership.
          </p>

          <p>
            <strong>
              Open Source (<Link href="/genai">MLflow</Link>):
            </strong>{" "}
            With MLflow, you maintain complete control over your agent
            optimization data and infrastructure. Trace data, evaluation
            results, and prompt versions stay on your own systems. There are no
            per-seat fees, no usage limits, and no vendor lock-in. MLflow
            integrates with any LLM provider and agent framework (LangGraph,
            CrewAI, OpenAI Agents, and more) through OpenTelemetry-compatible
            tracing.
          </p>

          <p>
            <strong>Proprietary SaaS Tools:</strong> Commercial agent
            optimization and observability platforms offer convenience but at
            the cost of flexibility and control. They typically charge per seat
            or per trace volume, which becomes expensive at scale with agents
            that generate many traces per request. Your trace data and
            evaluation results are sent to their servers, raising privacy and
            compliance concerns. You're locked into their ecosystem, making it
            difficult to switch providers or customize workflows.
          </p>

          <p>
            <strong>Why Teams Choose Open Source:</strong> Organizations
            building AI agents at scale choose MLflow because it provides
            production-grade tracing, evaluation, prompt optimization, and cost
            management, compliance, and governance without giving up control of
            their data, cost predictability, or flexibility. The Apache 2.0
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
                Agent Tracing Documentation
              </Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                Agent Evaluation Guide
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
