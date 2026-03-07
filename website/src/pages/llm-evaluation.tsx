import { useState } from "react";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../components/Header/Header";
import { SocialLinksFooter } from "../components/SocialLinksFooter/SocialLinksFooter";
import { ArticleSidebar } from "../components/ArticleSidebar/ArticleSidebar";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import EvaluationHero from "@site/static/img/GenAI_evaluations/GenAI_evaluations_hero.png";
import { CopyButton } from "../components/CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";

const SEO_TITLE = "LLM Evaluation and Agent Evaluation | MLflow AI Platform";
const SEO_DESCRIPTION =
  "Agent evaluation and LLM evaluation systematically assess how well agents and LLM applications perform. Explore evaluation on MLflow's open-source AI platform.";

const faqs: {
  question: string;
  answer: React.ReactNode;
  answerText?: string;
}[] = [
  // General concept questions
  {
    question: "What is agent evaluation?",
    answer:
      "Agent evaluation is the systematic process of measuring how well autonomous AI agents perform their intended tasks. Extending beyond LLM evaluation, agent evaluation must assess multi-step reasoning, tool selection accuracy, error recovery, and task completion across complex workflows. This includes evaluating whether agents choose the right tools, use them correctly, handle edge cases gracefully, and achieve their objectives efficiently. MLflow provides specialized scorers and evaluation frameworks designed specifically for agentic systems.",
  },
  {
    question: "What is LLM evaluation?",
    answer:
      "LLM evaluation measures the quality of outputs from large language models across dimensions like accuracy, relevance, safety, and coherence. It goes beyond traditional software testing because LLM outputs are non-deterministic and open-ended. Evaluation uses automated judges (other LLMs that score outputs), human feedback, and code-based metrics to assess whether responses meet quality standards. MLflow supports both built-in LLM judges for common quality dimensions and custom judges tailored to your specific use case.",
  },
  {
    question: "What is an LLM judge?",
    answer: (
      <>
        An LLM judge is a language model used to automatically evaluate the
        outputs of an agent or LLM application. Instead of relying solely on
        human review (which is slow and expensive), LLM judges can assess
        thousands of responses for qualities like correctness, relevance,
        safety, and helpfulness.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/"}>
          MLflow provides built-in judges
        </Link>{" "}
        for common evaluation criteria and tools to create custom judges aligned
        with your domain expertise and quality standards.
      </>
    ),
    answerText:
      "An LLM judge is a language model used to automatically evaluate the outputs of an agent or LLM application. Instead of relying solely on human review (which is slow and expensive), LLM judges can assess thousands of responses for qualities like correctness, relevance, safety, and helpfulness. MLflow provides built-in judges for common evaluation criteria and tools to create custom judges aligned with your domain expertise and quality standards.",
  },
  {
    question:
      "How is AI evaluation different from traditional software testing?",
    answer:
      "Traditional software testing verifies deterministic outputs: given input X, expect output Y. AI evaluation must handle non-deterministic systems where the same input can produce many valid (or invalid) outputs. Instead of exact matching, AI evaluation assesses quality dimensions like relevance, factual accuracy, safety, and user satisfaction. It requires statistical approaches (pass rates across datasets), semantic comparison (meaning rather than exact text), and often human or LLM judges to assess subjective quality criteria.",
  },
  {
    question: "What are LLM evaluation metrics?",
    answer: (
      <>
        LLM evaluation metrics are quantitative measures of output quality.
        Common metrics include correctness (factual accuracy), relevance
        (answers the question asked), groundedness (supported by provided
        context), safety (free from harmful content), and coherence (logically
        structured).{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          MLflow provides 70+ built-in metrics
        </Link>{" "}
        covering these dimensions, plus APIs to define custom metrics using
        Python code or LLM judges tailored to your specific requirements.
      </>
    ),
    answerText:
      "LLM evaluation metrics are quantitative measures of output quality. Common metrics include correctness (factual accuracy), relevance (answers the question asked), groundedness (supported by provided context), safety (free from harmful content), and coherence (logically structured). MLflow provides 70+ built-in metrics covering these dimensions, plus APIs to define custom metrics using Python code or LLM judges tailored to your specific requirements.",
  },
  {
    question: "Do I need evaluation for my agent?",
    answer:
      "Yes, if you're building production agents. Without evaluation, you have no objective way to know if your agent actually works well, whether changes improve or degrade quality, or when production behavior drifts from expected standards. Evaluation enables confident iteration: test prompt changes before deployment, catch regressions automatically, and maintain quality as models and data evolve. Even simple agents benefit from basic evaluation to prevent embarrassing failures and build user trust.",
  },
  {
    question: "When should I evaluate my agent or LLM application?",
    answer:
      "Evaluate throughout the development lifecycle. During development, evaluate iteratively as you refine prompts, models, and logic. Before deployment, run comprehensive evaluations against benchmark datasets to establish baseline quality. In production, continuously monitor with automated judges to detect regressions, drift, or emerging failure patterns. After incidents, use evaluation to understand root causes and verify fixes. MLflow supports all these scenarios with batch evaluation, inline evaluation during development, and production monitoring.",
  },
  {
    question: "How is agent evaluation different from LLM evaluation?",
    answer: (
      <>
        LLM evaluation assesses single-turn input/output pairs: given a prompt,
        is the response accurate, relevant, and safe? Agent evaluation is far
        more complex because agents take multiple steps, use tools, and make
        decisions. You must evaluate the entire trajectory: Did the agent choose
        the right tools? Did it recover from errors? Did it complete the goal
        efficiently? LLM evaluation uses metrics like correctness and relevance.
        Agent evaluation adds trajectory metrics like tool call efficiency,
        reasoning quality, and task completion rate.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          MLflow supports both
        </Link>{" "}
        with specialized scorers for each use case.
      </>
    ),
    answerText:
      "LLM evaluation assesses single-turn input/output pairs: given a prompt, is the response accurate, relevant, and safe? Agent evaluation is far more complex because agents take multiple steps, use tools, and make decisions. You must evaluate the entire trajectory: Did the agent choose the right tools? Did it recover from errors? Did it complete the goal efficiently? LLM evaluation uses metrics like correctness and relevance. Agent evaluation adds trajectory metrics like tool call efficiency, reasoning quality, and task completion rate. MLflow supports both with specialized scorers for each use case.",
  },
  {
    question: "How do I evaluate agent tool use and reasoning?",
    answer:
      "Evaluating agents requires assessing both individual steps and end-to-end outcomes. MLflow's tracing captures the complete execution graph: which tools were called, with what arguments, and in what order. You can then evaluate tool selection accuracy (did it choose the right tool?), argument correctness (were parameters valid?), reasoning quality (did the chain of thought make sense?), and final outcome (was the goal achieved?). Use trajectory-based scorers to evaluate the full agent path, not just the final answer.",
  },
  {
    question: "How do I evaluate RAG applications?",
    answer: (
      <>
        RAG (Retrieval-Augmented Generation) evaluation requires assessing both
        retrieval quality and generation quality. For retrieval, measure whether
        the right documents were retrieved (context relevance) and whether all
        relevant information was found (recall). For generation, evaluate
        whether the response is grounded in the retrieved context (faithfulness)
        and doesn't hallucinate facts not present in the sources (groundedness).
        MLflow provides built-in judges for{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/"}>
          context relevance, faithfulness, and groundedness
        </Link>
        , plus integration with RAGAS and DeepEval metrics for comprehensive RAG
        evaluation.
      </>
    ),
    answerText:
      "RAG (Retrieval-Augmented Generation) evaluation requires assessing both retrieval quality and generation quality. For retrieval, measure whether the right documents were retrieved (context relevance) and whether all relevant information was found (recall). For generation, evaluate whether the response is grounded in the retrieved context (faithfulness) and doesn't hallucinate facts not present in the sources (groundedness). MLflow provides built-in judges for context relevance, faithfulness, and groundedness, plus integration with RAGAS and DeepEval metrics for comprehensive RAG evaluation.",
  },
  {
    question: "How do I build an evaluation dataset?",
    answer: (
      <>
        Start with real examples from your application. Collect production
        traces that represent typical usage, edge cases, and known failure
        modes. Add expected outputs (ground truth) where definitive answers
        exist, or rely on LLM judges for open-ended evaluation.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "datasets/"}>
          MLflow Evaluation Datasets
        </Link>{" "}
        let you version your data, track which evaluations used which versions,
        and incrementally add examples as you discover new failure patterns.
        Quality datasets are often small but representative, covering the full
        range of your application's expected behavior.
      </>
    ),
    answerText:
      "Start with real examples from your application. Collect production traces that represent typical usage, edge cases, and known failure modes. Add expected outputs (ground truth) where definitive answers exist, or rely on LLM judges for open-ended evaluation. MLflow Evaluation Datasets let you version your data, track which evaluations used which versions, and incrementally add examples as you discover new failure patterns. Quality datasets are often small but representative, covering the full range of your application's expected behavior.",
  },
  {
    question: "When should I use LLM judges vs human evaluation?",
    answer: (
      <>
        Use <strong>LLM judges</strong> for: high-volume evaluation (thousands
        of examples), consistent and reproducible scoring, well-defined criteria
        (safety, relevance, factuality), rapid iteration during development, and
        continuous production monitoring. Use <strong>human evaluation</strong>{" "}
        for: calibrating LLM judges (ensuring they align with expert judgment),
        evaluating subjective qualities (tone, brand voice, creativity), edge
        cases where LLM judges may fail, and building ground truth datasets. The
        best approach combines both:{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          use human feedback to validate and improve LLM judges
        </Link>
        , then deploy judges at scale. MLflow supports both human feedback
        collection and LLM-based evaluation in an integrated workflow.
      </>
    ),
    answerText:
      "Use LLM judges for: high-volume evaluation (thousands of examples), consistent and reproducible scoring, well-defined criteria (safety, relevance, factuality), rapid iteration during development, and continuous production monitoring. Use human evaluation for: calibrating LLM judges (ensuring they align with expert judgment), evaluating subjective qualities (tone, brand voice, creativity), edge cases where LLM judges may fail, and building ground truth datasets. The best approach combines both: use human feedback to validate and improve LLM judges, then deploy judges at scale. MLflow supports both human feedback collection and LLM-based evaluation in an integrated workflow.",
  },
  // MLflow-specific questions
  {
    question: "What is the best agent evaluation tool?",
    answer:
      "The best agent evaluation tool depends on your requirements. MLflow is the leading open-source option, offering comprehensive evaluation capabilities without vendor lock-in. It supports any agent framework (LangGraph, CrewAI, ADK, Pydantic AI, etc.), any LLM provider (OpenAI, Anthropic, Bedrock, etc.), and provides both built-in and custom evaluation metrics. Unlike proprietary tools, MLflow is free, gives you full control over your data, and integrates with your existing infrastructure. With 20,000+ GitHub stars and over 30 million monthly downloads, it's trusted by thousands of organizations.",
  },
  {
    question: "How do I get started with MLflow agent evaluation?",
    answer: (
      <>
        Getting started takes minutes. Install MLflow, create an evaluation
        dataset from your test cases, and run{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/quickstart/"}>
          mlflow.genai.evaluate()
        </Link>{" "}
        with built-in scorers like Correctness and Safety. View results in the
        MLflow UI to identify failures and track improvements. As you iterate,
        add custom scorers for your specific quality criteria and expand your
        evaluation dataset. The{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          evaluation documentation
        </Link>{" "}
        provides step-by-step guides and examples for every agent framework.
      </>
    ),
    answerText:
      "Getting started takes minutes. Install MLflow, create an evaluation dataset from your test cases, and run mlflow.genai.evaluate() with built-in scorers like Correctness and Safety. View results in the MLflow UI to identify failures and track improvements. As you iterate, add custom scorers for your specific quality criteria and expand your evaluation dataset. The evaluation documentation provides step-by-step guides and examples for every agent framework.",
  },
  {
    question: "What kind of LLM judges does MLflow provide?",
    answer: (
      <>
        MLflow includes{" "}
        <Link
          href={
            MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/llm-judge/predefined/"
          }
        >
          70+ built-in judges
        </Link>{" "}
        covering response quality (Safety, Correctness, RelevanceToQuery,
        Groundedness, Fluency), RAG (RetrievalRelevance, RetrievalGroundedness),
        agent behavior (ToolCallEfficiency, RoleAdherence), and multi-turn
        conversations (ConversationalSafety). Beyond built-in judges, MLflow
        supports{" "}
        <Link
          href={
            MLFLOW_GENAI_DOCS_URL +
            "eval-monitor/scorers/llm-judge/custom-judges/"
          }
        >
          custom LLM judges
        </Link>{" "}
        and{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/custom/"}>
          code-based scorers
        </Link>{" "}
        for domain-specific evaluation. MLflow also integrates with DeepEval and
        RAGAS for additional metrics.
      </>
    ),
    answerText:
      "MLflow includes 70+ built-in judges covering response quality (Safety, Correctness, RelevanceToQuery, Groundedness, Fluency), RAG (RetrievalRelevance, RetrievalGroundedness), agent behavior (ToolCallEfficiency, RoleAdherence), and multi-turn conversations (ConversationalSafety). Beyond built-in judges, MLflow supports custom LLM judges and code-based scorers for domain-specific evaluation. MLflow also integrates with DeepEval and RAGAS for additional metrics.",
  },
  {
    question: "How do I create custom evaluation metrics in MLflow?",
    answer: (
      <>
        MLflow offers two approaches for custom metrics. For code-based logic
        (regex patterns, length checks, JSON validation), use the{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/"}>
          @scorer decorator
        </Link>{" "}
        to wrap any Python function. For semantic evaluation requiring judgment,
        use{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/"}>
          make_judge()
        </Link>{" "}
        to create custom LLM judges with your criteria and feedback categories.
        Both approaches integrate seamlessly with MLflow's evaluation framework
        and tracking UI.
      </>
    ),
    answerText:
      "MLflow offers two approaches for custom metrics. For code-based logic (regex patterns, length checks, JSON validation), use the @scorer decorator to wrap any Python function. For semantic evaluation requiring judgment, use make_judge() to create custom LLM judges with your criteria and feedback categories. Both approaches integrate seamlessly with MLflow's evaluation framework and tracking UI.",
  },
  {
    question: "How do I compare different agent versions with MLflow?",
    answer: (
      <>
        MLflow makes version comparison straightforward. Run the same evaluation
        dataset against multiple agent versions, then use the{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          Evaluation UI
        </Link>{" "}
        to compare results side-by-side. View aggregate metrics (pass rates,
        average scores) to understand overall impact, then drill into individual
        examples where versions differ to understand why. The Trace Comparison
        view shows step-by-step differences in reasoning and tool use, helping
        you identify exactly what changed between versions.
      </>
    ),
    answerText:
      "MLflow makes version comparison straightforward. Run the same evaluation dataset against multiple agent versions, then use the Evaluation UI to compare results side-by-side. View aggregate metrics (pass rates, average scores) to understand overall impact, then drill into individual examples where versions differ to understand why. The Trace Comparison view shows step-by-step differences in reasoning and tool use, helping you identify exactly what changed between versions.",
  },
  {
    question: "What LLM providers does MLflow evaluation support?",
    answer:
      "MLflow evaluation works with any LLM provider. This includes OpenAI, Anthropic (Claude), AWS Bedrock, Google Vertex AI (Gemini), Azure OpenAI, Mistral, Cohere, Together AI, Anyscale, and local models via vLLM or Ollama. You can use different providers for your application and your evaluation judges. MLflow's provider-agnostic design means you're never locked into a single vendor and can switch providers or use multiple providers as needed.",
  },
  {
    question: "How does MLflow compare to other evaluation tools?",
    answer:
      "MLflow differentiates through openness and completeness. Unlike proprietary tools that charge per evaluation or lock you into their ecosystem, MLflow is 100% open source under Apache 2.0. It provides end-to-end evaluation: built-in judges, custom metric APIs, Evaluation Datasets, version comparison, and production monitoring. MLflow integrates with any LLM and agent framework, stores data where you choose, and is backed by the Linux Foundation rather than a single vendor. For teams prioritizing flexibility and data sovereignty, MLflow is the clear choice.",
  },
  {
    question: "Is MLflow evaluation free?",
    answer:
      "Yes. MLflow is completely free and open source under the Apache 2.0 license, backed by the Linux Foundation. All evaluation features (built-in judges, custom scorers, Evaluation Datasets, the evaluation UI, and production monitoring) are included at no cost. You can use MLflow in commercial applications without licensing fees. The only costs are your own infrastructure (which you control) and any LLM API calls for running judges. Managed MLflow is also available on Databricks and other platforms if you prefer hosted solutions.",
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
    "Open-source platform for agent evaluation, LLM evaluation, experiment tracking, and deployment.",
  url: "https://mlflow.org",
  license: "https://www.apache.org/licenses/LICENSE-2.0",
};

export default function AgentEvaluation() {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://mlflow.org/llm-evaluation" />
        <link rel="canonical" href="https://mlflow.org/llm-evaluation" />
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
          <h1>LLM Evaluation and Agent Evaluation</h1>

          <p>
            <a href="#llm-evaluation">LLM evaluation</a> systematically measures
            the quality of LLM applications across dimensions like correctness,
            relevance, safety, and coherence.{" "}
            <a href="#agent-evaluation">Agent evaluation</a> extends LLM
            evaluation to also assess multi-step reasoning, tool selection, task
            completion, and beyond for autonomous{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>agents</Link>.
          </p>

          <p>
            Evaluation gives engineering teams confidence that their agents and
            LLM applications actually work well: not just whether they run, but
            whether they produce correct, safe, and useful results. As agents
            move from prototypes to production-critical applications, evaluation
            becomes essential for maintaining quality and enabling continuous
            improvement.
          </p>

          <p>
            Unlike traditional software, agents and LLM applications are{" "}
            <strong>non-deterministic</strong>: the same input can produce
            different outputs. This makes exact-match testing insufficient. AI
            evaluation uses{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/"}>
              LLM judges
            </Link>
            , human feedback, and code-based metrics to assess quality
            dimensions like correctness, relevance, safety, and helpfulness
            across representative datasets.
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
                src="/img/releases/3.10.0/inline-eval.mp4"
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
                <a href="#llm-evaluation">LLM Evaluation</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#agent-evaluation">Agent Evaluation</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#agent-vs-llm">LLM vs Agent Evaluation</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#evaluation-lifecycle">Evaluation Lifecycle</a>
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

          <h2>Why LLM and Agent Evaluation Matters</h2>

          <p>
            Agents, LLM applications, and RAG systems introduce unique
            challenges that traditional software testing can't address:
          </p>

          <div className="grid-2">
            <div className="card">
              <h3>Quality Assurance</h3>
              <p>
                <strong>Problem:</strong> Agent outputs are non-deterministic
                and can include hallucinations or irrelevant responses.
              </p>
              <p>
                <strong>Solution:</strong> Automated LLM judges continuously
                assess quality dimensions like correctness, relevance, and
                safety across every response.
              </p>
            </div>

            <div className="card">
              <h3>Regression Detection</h3>
              <p>
                <strong>Problem:</strong> Prompt changes, model updates, or data
                drift can silently degrade quality without obvious errors.
              </p>
              <p>
                <strong>Solution:</strong> Run evaluations against benchmark
                datasets before deployment and continuously in production to
                catch regressions early.
              </p>
            </div>

            <div className="card">
              <h3>Agent Debugging</h3>
              <p>
                <strong>Problem:</strong> Multi-step agents make complex
                decisions about tool use, data access, and control flow that are
                hard to understand and debug.
              </p>
              <p>
                <strong>Solution:</strong> Evaluate agent trajectories
                end-to-end, assessing tool selection accuracy, reasoning
                quality, and task completion.
              </p>
            </div>

            <div className="card">
              <h3>Safety & Compliance</h3>
              <p>
                <strong>Problem:</strong> Agents can produce harmful, off-topic,
                or policy-violating outputs that are hard to catch with static
                rules.
              </p>
              <p>
                <strong>Solution:</strong> Use LLM judges to assess safety,
                toxicity, and policy compliance across every response.
              </p>
            </div>
          </div>

          <h2 id="llm-evaluation">LLM Evaluation</h2>

          <p>
            LLM evaluation focuses on measuring the quality of outputs from
            large language models and LLM-powered applications. This includes
            assessing whether responses are accurate, relevant to the user's
            question, grounded in provided context, free from harmful content,
            and helpful for the user's goals.
          </p>

          <p>
            For single-turn LLM applications (content generators, summarization
            tools, translation, classification), evaluation helps you understand
            which prompts produce the best results, identify quality issues
            before they reach users, and track quality over time.{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/"}>
              LLM judges
            </Link>{" "}
            automate this assessment, enabling evaluation at scale without
            requiring human review of every response.
          </p>

          <p>
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
              MLflow's evaluation framework
            </Link>{" "}
            provides built-in judges for common quality dimensions (safety,
            correctness, relevance, groundedness) and APIs to create custom
            judges tailored to your specific requirements and domain expertise.
          </p>

          <h2 id="agent-evaluation">Agent Evaluation</h2>

          <p>
            Agent evaluation extends LLM evaluation to multi-step agentic
            systems. While LLM evaluation assesses individual responses, agent
            evaluation must assess the complete trajectory: how agents reason
            about tasks, which tools they select, how they handle errors, and
            whether they achieve their goals efficiently.
          </p>

          <p>
            Agents built with frameworks like LangGraph, CrewAI, ADK, or
            Pydantic AI can behave unpredictably: getting stuck in loops, making
            incorrect tool choices, or producing inconsistent outputs across
            runs. Agent evaluation captures the full execution graph, enabling
            you to assess whether the agent chose the right tools, used them
            with correct arguments, recovered gracefully from errors, and
            completed objectives efficiently.
          </p>

          <p>
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
              MLflow supports agent evaluation
            </Link>{" "}
            through trajectory-based scorers that assess the complete agent
            path, not just the final answer. Combined with tracing that captures
            every step, you can debug agent failures, optimize prompts and tool
            selection logic, and build confidence that agents behave correctly
            in production.
          </p>

          <h2 id="agent-vs-llm">
            LLM Evaluation vs Agent Evaluation: Key Differences
          </h2>

          <p>
            Understanding the distinction between agent evaluation and LLM
            evaluation is critical for choosing the right evaluation strategy.
            While they share common foundations, they differ significantly in
            scope, metrics, and complexity.
          </p>

          <div
            style={{
              overflowX: "auto",
              margin: "32px 0",
            }}
          >
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: "14px",
                lineHeight: "1.6",
              }}
            >
              <thead>
                <tr
                  style={{
                    backgroundColor: "#f8fafc",
                    borderBottom: "2px solid #e2e8f0",
                  }}
                >
                  <th
                    style={{
                      padding: "16px",
                      textAlign: "left",
                      fontWeight: "600",
                      color: "#1a1a1a",
                    }}
                  >
                    Aspect
                  </th>
                  <th
                    style={{
                      padding: "16px",
                      textAlign: "left",
                      fontWeight: "600",
                      color: "#1a1a1a",
                    }}
                  >
                    LLM Evaluation
                  </th>
                  <th
                    style={{
                      padding: "16px",
                      textAlign: "left",
                      fontWeight: "600",
                      color: "#1a1a1a",
                    }}
                  >
                    Agent Evaluation
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                  <td
                    style={{
                      padding: "16px",
                      fontWeight: "500",
                      color: "#1a1a1a",
                    }}
                  >
                    Scope
                  </td>
                  <td style={{ padding: "16px", color: "#3d3d3d" }}>
                    Single input/output pair
                  </td>
                  <td style={{ padding: "16px", color: "#3d3d3d" }}>
                    Multi-step trajectory with tool calls
                  </td>
                </tr>
                <tr
                  style={{
                    borderBottom: "1px solid #e2e8f0",
                    backgroundColor: "#fafafa",
                  }}
                >
                  <td
                    style={{
                      padding: "16px",
                      fontWeight: "500",
                      color: "#1a1a1a",
                    }}
                  >
                    What You Evaluate
                  </td>
                  <td style={{ padding: "16px", color: "#3d3d3d" }}>
                    Response quality only
                  </td>
                  <td style={{ padding: "16px", color: "#3d3d3d" }}>
                    Reasoning + tool use + final outcome
                  </td>
                </tr>
                <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                  <td
                    style={{
                      padding: "16px",
                      fontWeight: "500",
                      color: "#1a1a1a",
                    }}
                  >
                    Key Metrics
                  </td>
                  <td style={{ padding: "16px", color: "#3d3d3d" }}>
                    Correctness, relevance, safety, fluency
                  </td>
                  <td style={{ padding: "16px", color: "#3d3d3d" }}>
                    Tool call accuracy, task completion, efficiency, error
                    recovery
                  </td>
                </tr>
                <tr
                  style={{
                    borderBottom: "1px solid #e2e8f0",
                    backgroundColor: "#fafafa",
                  }}
                >
                  <td
                    style={{
                      padding: "16px",
                      fontWeight: "500",
                      color: "#1a1a1a",
                    }}
                  >
                    Typical Use Cases
                  </td>
                  <td style={{ padding: "16px", color: "#3d3d3d" }}>
                    Summarization, translation, single-turn Q&A, content
                    generation, classification
                  </td>
                  <td style={{ padding: "16px", color: "#3d3d3d" }}>
                    Chatbots, autonomous assistants, coding agents, RAG systems,
                    research agents, workflow automation
                  </td>
                </tr>
                <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                  <td
                    style={{
                      padding: "16px",
                      fontWeight: "500",
                      color: "#1a1a1a",
                    }}
                  >
                    Failure Modes
                  </td>
                  <td style={{ padding: "16px", color: "#3d3d3d" }}>
                    Hallucinations, irrelevance, unsafe content
                  </td>
                  <td style={{ padding: "16px", color: "#3d3d3d" }}>
                    Infinite loops, wrong tool selection, incomplete goals,
                    inefficient paths
                  </td>
                </tr>
                <tr style={{ backgroundColor: "#fafafa" }}>
                  <td
                    style={{
                      padding: "16px",
                      fontWeight: "500",
                      color: "#1a1a1a",
                    }}
                  >
                    MLflow Scorers
                  </td>
                  <td style={{ padding: "16px", color: "#3d3d3d" }}>
                    <Link
                      href={
                        MLFLOW_GENAI_DOCS_URL +
                        "eval-monitor/scorers/llm-judge/predefined/#response-quality"
                      }
                    >
                      Safety
                    </Link>
                    ,{" "}
                    <Link
                      href={
                        MLFLOW_GENAI_DOCS_URL +
                        "eval-monitor/scorers/llm-judge/predefined/#response-quality"
                      }
                    >
                      Correctness
                    </Link>
                    ,{" "}
                    <Link
                      href={
                        MLFLOW_GENAI_DOCS_URL +
                        "eval-monitor/scorers/llm-judge/predefined/#response-quality"
                      }
                    >
                      RelevanceToQuery
                    </Link>
                    ,{" "}
                    <Link
                      href={
                        MLFLOW_GENAI_DOCS_URL +
                        "eval-monitor/scorers/llm-judge/predefined/#rag"
                      }
                    >
                      Groundedness
                    </Link>
                  </td>
                  <td style={{ padding: "16px", color: "#3d3d3d" }}>
                    <Link
                      href={
                        MLFLOW_GENAI_DOCS_URL +
                        "eval-monitor/scorers/llm-judge/predefined/#tool-call"
                      }
                    >
                      ToolCallEfficiency
                    </Link>
                    ,{" "}
                    <Link
                      href={
                        MLFLOW_GENAI_DOCS_URL +
                        "eval-monitor/scorers/llm-judge/predefined/#multi-turn"
                      }
                    >
                      RoleAdherence
                    </Link>
                    ,{" "}
                    <Link
                      href={
                        MLFLOW_GENAI_DOCS_URL +
                        "eval-monitor/scorers/llm-judge/predefined/#multi-turn"
                      }
                    >
                      ConversationalSafety
                    </Link>
                    ,{" "}
                    <Link
                      href={
                        MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/custom/"
                      }
                    >
                      custom trajectory scorers
                    </Link>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <h2 id="evaluation-lifecycle">The Evaluation Lifecycle</h2>

          <p>
            Agent and LLM evaluation isn't a one-time activity. It's a
            continuous cycle that spans the entire development and deployment
            process. Here's how evaluation fits into each stage:
          </p>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              gap: "20px",
              margin: "32px 0",
            }}
          >
            <div
              style={{
                background: "#ffffff",
                border: "1px solid #e5e7eb",
                borderRadius: "4px",
                padding: "24px",
                boxShadow:
                  "0 0 0 1px rgba(50, 50, 93, 0.05), 0 0 14px 5px rgba(50, 50, 93, 0.08), 0 0 10px 3px rgba(0, 0, 0, 0.05)",
              }}
            >
              <div
                style={{
                  fontSize: "28px",
                  fontWeight: "700",
                  marginBottom: "8px",
                  color: "#0194e2",
                }}
              >
                1
              </div>
              <div
                style={{
                  fontWeight: "600",
                  marginBottom: "8px",
                  color: "#1a1a1a",
                  fontSize: "16px",
                }}
              >
                Build & Experiment
              </div>
              <div
                style={{
                  fontSize: "14px",
                  color: "#505050",
                  lineHeight: "1.5",
                }}
              >
                Run evaluations as you develop. Test changes to prompts, tools,
                and logic and compare results instantly.
              </div>
            </div>

            <div
              style={{
                background: "#ffffff",
                border: "1px solid #e5e7eb",
                borderRadius: "4px",
                padding: "24px",
                boxShadow:
                  "0 0 0 1px rgba(50, 50, 93, 0.05), 0 0 14px 5px rgba(50, 50, 93, 0.08), 0 0 10px 3px rgba(0, 0, 0, 0.05)",
              }}
            >
              <div
                style={{
                  fontSize: "28px",
                  fontWeight: "700",
                  marginBottom: "8px",
                  color: "#0194e2",
                }}
              >
                2
              </div>
              <div
                style={{
                  fontWeight: "600",
                  marginBottom: "8px",
                  color: "#1a1a1a",
                  fontSize: "16px",
                }}
              >
                Benchmark & Validate
              </div>
              <div
                style={{
                  fontSize: "14px",
                  color: "#505050",
                  lineHeight: "1.5",
                }}
              >
                Run comprehensive evaluations against curated datasets before
                deployment. Establish baseline quality metrics.
              </div>
            </div>

            <div
              style={{
                background: "#ffffff",
                border: "1px solid #e5e7eb",
                borderRadius: "4px",
                padding: "24px",
                boxShadow:
                  "0 0 0 1px rgba(50, 50, 93, 0.05), 0 0 14px 5px rgba(50, 50, 93, 0.08), 0 0 10px 3px rgba(0, 0, 0, 0.05)",
              }}
            >
              <div
                style={{
                  fontSize: "28px",
                  fontWeight: "700",
                  marginBottom: "8px",
                  color: "#0194e2",
                }}
              >
                3
              </div>
              <div
                style={{
                  fontWeight: "600",
                  marginBottom: "8px",
                  color: "#1a1a1a",
                  fontSize: "16px",
                }}
              >
                Monitor in Production
              </div>
              <div
                style={{
                  fontSize: "14px",
                  color: "#505050",
                  lineHeight: "1.5",
                }}
              >
                Continuously evaluate production traces with LLM judges. Detect
                regressions and quality drift automatically.
              </div>
            </div>

            <div
              style={{
                background: "#ffffff",
                border: "1px solid #e5e7eb",
                borderRadius: "4px",
                padding: "24px",
                boxShadow:
                  "0 0 0 1px rgba(50, 50, 93, 0.05), 0 0 14px 5px rgba(50, 50, 93, 0.08), 0 0 10px 3px rgba(0, 0, 0, 0.05)",
              }}
            >
              <div
                style={{
                  fontSize: "28px",
                  fontWeight: "700",
                  marginBottom: "8px",
                  color: "#0194e2",
                }}
              >
                4
              </div>
              <div
                style={{
                  fontWeight: "600",
                  marginBottom: "8px",
                  color: "#1a1a1a",
                  fontSize: "16px",
                }}
              >
                Learn & Iterate
              </div>
              <div
                style={{
                  fontSize: "14px",
                  color: "#505050",
                  lineHeight: "1.5",
                }}
              >
                Convert failures into test cases. Collect human feedback to
                improve judges. Repeat the cycle.
              </div>
            </div>
          </div>

          <h2>Common Use Cases for AI Evaluation</h2>

          <p>
            AI evaluation solves real-world problems across the AI development
            lifecycle:
          </p>

          <ul>
            <li>
              <strong>Pre-deployment Testing:</strong> Before releasing new
              prompts, models, or agent logic, run comprehensive evaluations
              against benchmark datasets. Compare quality metrics to previous
              versions to ensure changes improve, not degrade, your application.
            </li>
            <li>
              <strong>Continuous Quality Monitoring:</strong> In production,
              continuously evaluate responses with automated judges to detect
              quality regressions, emerging failure patterns, or drift from
              expected behavior before users notice.
            </li>
            <li>
              <strong>Debugging Failures:</strong> When your agent or LLM
              application produces incorrect outputs, evaluation pinpoints the
              root cause. Was the retrieval poor? The reasoning flawed? The tool
              selection wrong? Evaluation results combined with{" "}
              <Link href="/llm-tracing">traces</Link> reveal exactly what went
              wrong.
            </li>
            <li>
              <strong>A/B Testing Prompt Changes:</strong> Before deploying
              prompt modifications to production, run{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                side-by-side evaluations
              </Link>{" "}
              with LLM judges. Compare quality metrics to ensure changes improve
              output quality, or use{" "}
              <Link href="/prompt-optimization">prompt optimization</Link> to
              automate the improvement process entirely.
            </li>
            <li>
              <strong>Building Regression Datasets:</strong> Convert production
              failures and edge cases into evaluation examples. Over time, build
              a comprehensive regression dataset that catches known failure
              modes before they reach production again.
            </li>
            <li>
              <strong>Safety and Compliance:</strong> Use safety scorers to
              detect harmful, biased, or policy-violating outputs. Maintain
              audit trails of evaluation results for regulatory compliance and
              incident investigation.
            </li>
          </ul>

          <h2 id="key-components">Key Components of AI Evaluation</h2>

          <p>
            A comprehensive AI evaluation platform combines six capabilities:
          </p>

          <ul>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                LLM Judges
              </Link>
              : Automated scorers that use language models to assess output
              quality across dimensions like correctness, relevance, safety, and
              helpfulness.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Custom Scorers
              </Link>
              : Code-based metrics using Python functions for deterministic
              checks like format validation, length limits, and regex patterns.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "datasets/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Evaluation Datasets
              </Link>
              : Curated sets of test cases with inputs and optional expected
              outputs that represent your application's typical usage and edge
              cases.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Evaluation UI
              </Link>
              : Visual interface to review results, compare versions, and drill
              into individual examples to understand failures.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "tracing/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Tracing Integration
              </Link>
              : Evaluate production traces to monitor quality continuously and
              debug failures with full execution context.
            </li>
            <li>
              <Link
                href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}
                style={{ color: "#007bff", fontWeight: "600" }}
              >
                Human Feedback
              </Link>
              : Collect expert reviews and end-user ratings to validate LLM
              judges and identify blind spots in automated evaluation.
            </li>
          </ul>

          <h2 id="how-to-implement">How to Implement Agent Evaluation</h2>

          <p>
            Modern open-source AI platforms like{" "}
            <Link href="/genai">MLflow</Link> make it easy to add comprehensive
            evaluation to your agents and LLM applications with minimal code.
          </p>

          <p>
            With just a few lines of code, you can evaluate your application
            against datasets using built-in or custom scorers. Results are
            tracked in MLflow, where you can compare versions, drill into
            failures, collect human feedback, and monitor quality over time. You
            can evaluate during development (testing new prompts), before
            deployment (comprehensive benchmark testing), and in production
            (continuous monitoring).
          </p>

          <p>
            Here are quick examples of evaluating with MLflow. Check out the{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
              MLflow evaluation documentation
            </Link>{" "}
            for comprehensive guides and framework-specific examples.
          </p>

          <p style={{ marginTop: "32px", marginBottom: "0px" }}>
            <strong>Evaluation with Built-in Judges</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton
                code={`import mlflow
from mlflow.genai.scorers import Safety, Correctness, RelevanceToQuery

# Evaluate your agent or LLM application
results = mlflow.genai.evaluate(
    data="my_eval_dataset",      # Your evaluation dataset
    predict_fn=my_agent,          # Your agent or LLM app
    scorers=[
        Safety(),                 # Check for harmful content
        Correctness(),            # Check factual accuracy
        RelevanceToQuery(),       # Check response relevance
    ],
)

# View results in MLflow UI or programmatically
print(f"Safety pass rate: {results.metrics['safety/pass_rate']}")
print(f"Correctness pass rate: {results.metrics['correctness/pass_rate']}")`}
              />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={`import mlflow
from mlflow.genai.scorers import Safety, Correctness, RelevanceToQuery

# Evaluate your agent or LLM application
results = mlflow.genai.evaluate(
    data="my_eval_dataset",      # Your evaluation dataset
    predict_fn=my_agent,          # Your agent or LLM app
    scorers=[
        Safety(),                 # Check for harmful content
        Correctness(),            # Check factual accuracy
        RelevanceToQuery(),       # Check response relevance
    ],
)

# View results in MLflow UI or programmatically
print(f"Safety pass rate: {results.metrics['safety/pass_rate']}")
print(f"Correctness pass rate: {results.metrics['correctness/pass_rate']}")`}
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
            <strong>Evaluation with Custom LLM Judges</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton
                code={`from mlflow.genai.judges import make_judge
from typing import Literal

# Create a custom judge for your specific criteria
conversation_quality_judge = make_judge(
    name="conversation_quality",
    instructions=(
        "Analyze the {{ conversation }} for signs of user frustration, "
        "unresolved questions, incomplete answers, or factual errors. "
        "Consider the full context of the interaction."
    ),
    feedback_value_type=Literal["high_quality", "medium_quality", "low_quality"],
)

# Use in evaluation
results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[conversation_quality_judge],
)`}
              />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={`from mlflow.genai.judges import make_judge
from typing import Literal

# Create a custom judge for your specific criteria
conversation_quality_judge = make_judge(
    name="conversation_quality",
    instructions=(
        "Analyze the {{ conversation }} for signs of user frustration, "
        "unresolved questions, incomplete answers, or factual errors. "
        "Consider the full context of the interaction."
    ),
    feedback_value_type=Literal["high_quality", "medium_quality", "low_quality"],
)

# Use in evaluation
results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[conversation_quality_judge],
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
            <strong>Evaluation with Custom Code-based Metrics</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton
                code={`from mlflow.genai.scorers import scorer

@scorer
def response_length(inputs, outputs):
    """Check response is within acceptable length limits."""
    word_count = len(outputs["response"].split())
    return {
        "score": 50 <= word_count <= 500,
        "rationale": f"Response has {word_count} words",
    }

@scorer
def contains_required_sections(inputs, outputs):
    """Check response includes all required sections."""
    response = outputs["response"].lower()
    required = ["summary", "recommendation", "next steps"]
    missing = [s for s in required if s not in response]
    return {
        "score": len(missing) == 0,
        "rationale": f"Missing sections: {missing}" if missing else "All sections present",
    }

# Use in evaluation
results = mlflow.genai.evaluate(
    data=traces_to_evaluate,
    scorers=[response_length, contains_required_sections],
)`}
              />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={`from mlflow.genai.scorers import scorer

@scorer
def response_length(inputs, outputs):
    """Check response is within acceptable length limits."""
    word_count = len(outputs["response"].split())
    return {
        "score": 50 <= word_count <= 500,
        "rationale": f"Response has {word_count} words",
    }

@scorer
def contains_required_sections(inputs, outputs):
    """Check response includes all required sections."""
    response = outputs["response"].lower()
    required = ["summary", "recommendation", "next steps"]
    missing = [s for s in required if s not in response]
    return {
        "score": len(missing) == 0,
        "rationale": f"Missing sections: {missing}" if missing else "All sections present",
    }

# Use in evaluation
results = mlflow.genai.evaluate(
    data=traces_to_evaluate,
    scorers=[response_length, contains_required_sections],
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

          <div style={{ margin: "40px 0", textAlign: "center" }}>
            <img
              src={EvaluationHero}
              alt="MLflow Evaluation UI showing LLM judge results with pass rates and individual assessments"
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
              The MLflow Evaluation UI displays results, enabling version
              comparison and failure analysis
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
              Apache 2.0, MLflow provides a complete evaluation solution with no
              vendor lock-in.{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL}>Get started →</Link>
            </p>
          </div>

          <h2>Open Source vs. Proprietary Evaluation Tools</h2>

          <p>
            When choosing an AI evaluation platform, the decision between open
            source and proprietary SaaS tools has significant long-term
            implications for your team, infrastructure, and data ownership.
          </p>

          <p>
            <strong>
              Open Source (<Link href="/genai">MLflow</Link>):
            </strong>{" "}
            With MLflow, you maintain complete control over your evaluation
            infrastructure and data. Deploy on your own infrastructure or use
            managed versions on Databricks, AWS, or other platforms. There are
            no per-evaluation fees, no usage limits, and no vendor lock-in. Your
            evaluation data stays under your control, and you can customize
            judges and metrics to your exact needs.
          </p>

          <p>
            <strong>Proprietary SaaS Tools:</strong> Commercial evaluation
            platforms offer convenience but at the cost of flexibility and
            control. They typically charge per evaluation or per seat, which can
            become expensive at scale. Your data is sent to their servers,
            raising privacy and compliance concerns. You're locked into their
            ecosystem, making it difficult to switch providers or customize
            functionality.
          </p>

          <p>
            <strong>Why Teams Choose Open Source:</strong> Organizations
            building production agents increasingly choose MLflow because it
            offers enterprise-grade evaluation without compromising on data
            sovereignty, cost predictability, or flexibility. The Apache 2.0
            license and Linux Foundation backing ensure MLflow remains truly
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
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                Agent Evaluation Documentation
              </Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/"}>
                LLM Judges Guide
              </Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/"}>
                Custom Scorers Documentation
              </Link>
            </li>
            <li>
              <Link href="/prompt-optimization">Prompt Optimization Guide</Link>
            </li>
            <li>
              <Link href="/llm-tracing">LLM Tracing FAQ</Link>
            </li>
            <li>
              <Link href="/ai-observability">AI Observability FAQ</Link>
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
