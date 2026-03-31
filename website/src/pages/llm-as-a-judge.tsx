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
  "LLM-as-a-Judge Evaluation for LLMs & Agents | MLflow Agent Platform";
const SEO_DESCRIPTION =
  "Learn how to use LLM-as-a-judge to evaluate LLM and agent quality at scale with MLflow--the comprehensive, open-source agent engineering and ops platform.";

const BUILTIN_CODE = `import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery, RetrievalGroundedness

eval_data = [
    {
        "inputs": {"question": "What is MLflow?"},
        "outputs": "MLflow is an open-source platform for managing the ML lifecycle.",
        "expectations": {
            "expected_facts": ["open-source platform", "ML lifecycle management"],
        },
    },
]

results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[
        Correctness(),
        RelevanceToQuery(),
        RetrievalGroundedness(),
    ],
)`;

const THIRD_PARTY_CODE = `from mlflow.genai.scorers.deepeval import AnswerRelevancy
from mlflow.genai.scorers.ragas import FactualCorrectness
from mlflow.genai.scorers.phoenix import Toxicity
from mlflow.genai.scorers import Correctness

results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[
        Correctness(),
        AnswerRelevancy(model="openai:/gpt-5"),
        FactualCorrectness(model="openai:/gpt-5"),
        Toxicity(model="openai:/gpt-5"),
    ],
)`;

const CUSTOM_CODE = `from mlflow.genai.judges import make_judge
from typing import Literal

domain_accuracy = make_judge(
    name="domain_accuracy",
    instructions="Evaluate whether the {{ outputs }} provides accurate domain-specific information for the given {{ inputs }}. Return 'accurate' if correct, 'inaccurate' if not.",
    feedback_value_type=Literal["accurate", "inaccurate"],
    model="openai:/gpt-5",
)

results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[domain_accuracy],
)`;

const MEMALIGN_CODE = `from mlflow.genai.judges.optimizers import MemAlignOptimizer

optimizer = MemAlignOptimizer(reflection_lm="openai:/gpt-5")

aligned_judge = my_judge.align(
    traces=alignment_traces,
    optimizer=optimizer,
)

results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[aligned_judge],
)`;

const faqs: {
  question: string;
  answer: React.ReactNode;
  answerText?: string;
}[] = [
  {
    question: "What is LLM-as-a-judge?",
    answer:
      "LLM-as-a-judge is an evaluation technique where one LLM evaluates the quality of another LLM's outputs. Instead of relying on human reviewers or simple metrics, you use a judge model (like GPT, Claude, or Gemini) to assess correctness, relevance, safety, and groundedness, producing both a score and a justification.",
  },
  {
    question: "When should I use LLM-as-a-judge evaluation?",
    answer:
      "Use LLM-as-a-judge when you need scalable, consistent evaluation that goes beyond simple metrics. It's essential for assessing subjective qualities like helpfulness, coherence, and appropriateness that BLEU, ROUGE, and exact match can't measure. It's also the standard approach for production monitoring where manual review can't scale.",
  },
  {
    question: "Which models work best as judges?",
    answer:
      "The most capable models like GPT, Claude, and Gemini typically perform best as judges. They follow complex evaluation criteria, provide detailed reasoning, and catch subtle quality issues. Use smaller or faster models for cost-sensitive scenarios with simpler criteria.",
  },
  {
    question: "How accurate are LLM judges compared to humans?",
    answer:
      "Properly calibrated LLM judges achieve over 80% agreement with human evaluators on correctness and readability, and 95% agreement when measured within one-score distance. This matches typical human-to-human inter-rater reliability. Judge optimization with MemAlign can improve agreement by an additional 30-50%.",
  },
  {
    question: "How do I get started with LLM-as-a-judge in MLflow?",
    answer: (
      <>
        Install MLflow, prepare an evaluation dataset with inputs and outputs,
        and run <code>mlflow.genai.evaluate()</code> with built-in scorers like{" "}
        <code>Correctness</code> and <code>RetrievalGroundedness</code>. MLflow
        provides 50+ built-in judges that work immediately, or you can create
        custom judges with the{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          Judge Builder UI
        </Link>
        —no code required.
      </>
    ),
    answerText:
      "Install MLflow, prepare an evaluation dataset with inputs and outputs, and run mlflow.genai.evaluate() with built-in scorers like Correctness and RetrievalGroundedness. MLflow provides 50+ built-in judges that work immediately, or you can create custom judges with the Judge Builder UI—no code required.",
  },
  {
    question: "What are common pitfalls with LLM-as-a-judge?",
    answer:
      "LLM judges can exhibit biases: position bias (favoring certain answer positions in comparisons), verbosity bias (preferring longer responses), and self-preference bias (favoring outputs from the same model family). Mitigate these with clear rubrics, calibration examples, and validation against human feedback. MLflow's judge optimization automates this calibration process.",
  },
  {
    question: "How do I define good evaluation criteria?",
    answer:
      'Good criteria are specific, measurable, and include examples. Instead of "Is this response good?", use "Does the response directly answer the user\'s question with accurate information supported by the retrieved documents?" Include 3-5 example judgments with scores and justifications to calibrate the judge.',
  },
  {
    question: "Can I use LLM-as-a-judge for agent evaluation?",
    answer: (
      <>
        Yes. MLflow provides specialized judges for agent evaluation:{" "}
        <code>ToolCallCorrectness</code> assesses whether agents choose
        appropriate tools with correct arguments, and{" "}
        <code>ToolCallEfficiency</code> identifies redundant calls and wasted
        token spend. For conversational agents, multi-turn judges evaluate
        dialogue quality across complete sessions. See the{" "}
        <Link
          href={
            MLFLOW_GENAI_DOCS_URL +
            "eval-monitor/scorers/llm-judge/predefined"
          }
        >
          built-in judges reference
        </Link>{" "}
        for the full list.
      </>
    ),
    answerText:
      "Yes. MLflow provides specialized judges for agent evaluation: ToolCallCorrectness assesses whether agents choose appropriate tools with correct arguments, and ToolCallEfficiency identifies redundant calls and wasted token spend. For conversational agents, multi-turn judges evaluate dialogue quality across complete sessions.",
  },
  {
    question: "Can I create judges without writing code?",
    answer: (
      <>
        Yes. MLflow's{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/llm-judge/custom-judges/create-custom-judge/#step-2-define-custom-judges"}>
          Judge Builder UI
        </Link>{" "}
        lets you create judges visually. Navigate to your experiment's Judges
        tab, define evaluation instructions in natural language, test against
        sample traces, and deploy. The UI is ideal for rapid prototyping and
        iterating on evaluation criteria.
      </>
    ),
    answerText:
      "Yes. MLflow's Judge Builder UI lets you create judges visually. Navigate to your experiment's Judges tab, define evaluation instructions in natural language, test against sample traces, and deploy. The UI is ideal for rapid prototyping and iterating on evaluation criteria.",
  },
  {
    question:
      "How do I improve judge accuracy when scores don't match my quality standards?",
    answer: (
      <>
        Use MLflow's judge optimization with{" "}
        <Link
          href={
            MLFLOW_GENAI_DOCS_URL +
            "eval-monitor/scorers/llm-judge/memalign/"
          }
        >
          MemAlign
        </Link>
        . Provide 20-50 examples where humans have scored outputs, and MemAlign
        automatically refines the judge's instructions using a dual-memory
        system. This improves agreement with human evaluators by 30-50%,
        transforming generic judges into domain-specific evaluators.
      </>
    ),
    answerText:
      "Use MLflow's judge optimization with MemAlign. Provide 20-50 examples where humans have scored outputs, and MemAlign automatically refines the judge's instructions using a dual-memory system. This improves agreement with human evaluators by 30-50%, transforming generic judges into domain-specific evaluators.",
  },
  {
    question: "Can I use LLM-as-a-judge for production monitoring?",
    answer: (
      <>
        Yes. MLflow's online evaluation runs judges continuously against
        production{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>traces</Link>. Set
        quality thresholds and receive alerts when scores drop. MLflow's
        in-UI trace evaluation lets you run judges on any trace directly from
        the interface without code.
      </>
    ),
    answerText:
      "Yes. MLflow's online evaluation runs judges continuously against production traces. Set quality thresholds and receive alerts when scores drop. MLflow's in-UI trace evaluation lets you run judges on any trace directly from the interface without code.",
  },
  {
    question: "How do I evaluate multi-turn conversations?",
    answer: (
      <>
        MLflow supports{" "}
        <a href="https://mlflow.org/blog/multiturn-evaluation">
          multi-turn evaluation and simulation
        </a>
        . Use session-level judges like <code>ConversationCompleteness</code>,{" "}
        <code>KnowledgeRetention</code>, and <code>UserFrustration</code> to
        evaluate complete conversation threads. Simulation mode replaces manual
        testing with an LLM that plays the user role, evaluating hundreds of
        conversations in minutes.
      </>
    ),
    answerText:
      "MLflow supports multi-turn evaluation and simulation. Use session-level judges like ConversationCompleteness, KnowledgeRetention, and UserFrustration to evaluate complete conversation threads. Simulation mode replaces manual testing with an LLM that plays the user role, evaluating hundreds of conversations in minutes.",
  },
  {
    question: "How much does LLM-as-a-judge cost?",
    answer:
      "Costs depend on your judge model and evaluation volume. Track spending using MLflow's trace cost tracking. Strategies to reduce costs: use smaller judge models for simpler criteria, evaluate a sample of production traffic rather than 100%, and cache judgments for repeated inputs.",
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
    "Open-source platform for LLM-as-a-judge evaluation, AI observability, experiment tracking, and deployment.",
  url: "https://mlflow.org",
  license: "https://www.apache.org/licenses/LICENSE-2.0",
};

export default function LLMAsAJudge() {
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
          content="https://mlflow.org/llm-as-a-judge"
        />
        <link rel="canonical" href="https://mlflow.org/llm-as-a-judge" />
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
          <h1>LLM-as-a-Judge for LLM and Agent Evaluation</h1>

          <p>
            LLM-as-a-judge evaluates the outputs of LLM applications and
            agents across quality dimensions like correctness, relevance,
            groundedness, safety, and helpfulness. Any model can act as a
            judge (OpenAI, Claude, Gemini, open-source, etc.), and each
            assessment produces both a score and a written justification.
          </p>

          <p>
            LLM-as-a-judge gives engineering teams automated quality
            assessment at production scale. Traditional metrics like BLEU
            and ROUGE measure token overlap but miss whether a response
            hallucinated or violated tone guidelines. Human reviewers catch
            these issues but can only evaluate a limited number of outputs
            per day. As LLM applications move from prototypes to production,
            judge-based evaluation becomes essential for maintaining quality
            and catching regressions.
          </p>

          <p>
            <Link href="/genai">MLflow's</Link>{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
              evaluation framework
            </Link>{" "}
            supports LLM-as-a-judge with built-in judges, custom judge
            creation, and automatic tracking of every evaluation run across
            model versions, prompt variants, and system changes.
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
                <a href="#why-llm-as-a-judge-matters">
                  Why LLM-as-a-Judge Matters
                </a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#out-of-the-box-judges">Out-of-the-Box Judges</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#custom-judges">Custom Judges</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#judge-optimization">Judge Optimization</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#common-use-cases">
                  Common Use Cases for LLM-as-a-Judge
                </a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#how-to-implement">
                  How to Implement LLM-as-a-Judge
                </a>
              </li>
              <li style={{ marginBottom: "0" }}>
                <a href="#faq">FAQ</a>
              </li>
            </ul>
          </div>

          {/* ---- Why LLM-as-a-Judge Matters ---- */}
          <h2 id="why-llm-as-a-judge-matters">
            Why LLM-as-a-Judge Matters
          </h2>

          <p>
            Agents, LLM applications, and RAG systems introduce evaluation
            challenges that traditional testing can't address:
          </p>

          <div className="grid-2">
            <div className="card">
              <h3>Human Evaluation Doesn't Scale</h3>
              <p>
                <strong>Problem:</strong> Manual review creates bottlenecks
                and inconsistency. Different reviewers score the same output
                differently, and coverage is always incomplete.
              </p>
              <p>
                <strong>Solution:</strong> LLM judges apply the same criteria
                to every output, achieving over 80% agreement with human
                evaluators and eliminating reviewer inconsistency.
              </p>
            </div>

            <div className="card">
              <h3>Traditional Metrics Miss Nuance</h3>
              <p>
                <strong>Problem:</strong> BLEU and ROUGE measure token overlap
                but can't assess whether a response is actually helpful,
                appropriate, or safe for the user.
              </p>
              <p>
                <strong>Solution:</strong> LLM judges understand context and
                intent. They evaluate the qualities users actually care
                about: accuracy, helpfulness, tone, and policy compliance.
              </p>
            </div>

            <div className="card">
              <h3>Production Monitoring is Critical</h3>
              <p>
                <strong>Problem:</strong> Quality regressions in production go
                undetected until users report them. Manual spot-checking
                can't keep up.
              </p>
              <p>
                <strong>Solution:</strong> Run judges continuously against
                production{" "}
                <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>traces</Link>,
                catching degradation within seconds. Set thresholds and alert
                when scores drop.
              </p>
            </div>

            <div className="card">
              <h3>Iteration Requires Measurement</h3>
              <p>
                <strong>Problem:</strong> Without quantitative metrics, teams
                can't measure whether prompt changes or model upgrades
                improve quality. Progress is guesswork.
              </p>
              <p>
                <strong>Solution:</strong> LLM judges provide quantitative
                scores. Baseline: 3.2 correctness. New prompt: 3.8. MLflow
                tracks all runs for easy comparison.
              </p>
            </div>
          </div>

          {/* ---- Out-of-the-Box Judges ---- */}
          <h2 id="out-of-the-box-judges">Out-of-the-Box Judges</h2>

          <p>
            Some quality checks are common across every LLM application:
            correctness, safety, groundedness, relevance. MLflow ships with
            built-in judges for these, each returning a score and a written
            justification for every output it evaluates.
          </p>

          <p>
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "eval-monitor/scorers/llm-judge/predefined"
              }
              style={{ color: "#007bff", fontWeight: "600" }}
            >
              Grounding judges
            </Link>{" "}
            verify that responses are supported by source documents and
            catch hallucinations.{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "eval-monitor/scorers/llm-judge/predefined"
              }
              style={{ color: "#007bff", fontWeight: "600" }}
            >
              Response quality judges
            </Link>{" "}
            cover correctness against expected answers, safety screening
            for toxic or harmful outputs, and adherence to{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "eval-monitor/scorers/llm-judge/guidelines/"
              }
              style={{ color: "#007bff", fontWeight: "600" }}
            >
              custom guidelines
            </Link>{" "}
            you define yourself.
          </p>

          <p>
            For conversational applications, single-turn evaluation isn't
            enough. A chatbot can answer individual questions well but still
            lose context, go in circles, or frustrate users over the course of
            a conversation. MLflow's{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL + "eval-monitor/"
              }
              style={{ color: "#007bff", fontWeight: "600" }}
            >
              multi-turn judges
            </Link>{" "}
            evaluate complete conversation sessions for context retention,
            resolution, and user satisfaction. And for agentic systems,{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "eval-monitor/scorers/llm-judge/predefined"
              }
              style={{ color: "#007bff", fontWeight: "600" }}
            >
              tool call judges
            </Link>{" "}
            assess whether agents are choosing the right tools and using
            them efficiently.
          </p>

          <h2 id="third-party-judge-integrations">Third-Party Judge Integrations</h2>

          <p>
            Beyond its own judges, MLflow integrates with{" "}
            <strong>DeepEval</strong>, <strong>RAGAS</strong>, and{" "}
            <strong>Arize Phoenix</strong>, giving you access to 20+ additional
            evaluation metrics. Third-party judges are imported directly
            from their packages and
            work alongside MLflow's built-in judges in a single evaluation
            run.
          </p>

          {/* ---- Custom Judges ---- */}
          <h2 id="custom-judges">Custom Judges</h2>

          <p>
            Built-in judges cover the most common quality dimensions, but
            every team has criteria specific to what they're building. A
            healthcare company needs to verify medical accuracy. A financial
            services firm needs to check regulatory compliance. An e-commerce
            platform needs to ensure product recommendations match user intent.
            Built-in judges can't cover these.
          </p>

          <p>
            The fastest way to create a custom judge is through MLflow's{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "eval-monitor/scorers/llm-judge/custom-judges/create-custom-judge/#step-2-define-custom-judges"
              }
              style={{ color: "#007bff", fontWeight: "600" }}
            >
              Judge Builder UI
            </Link>
            . Define your evaluation criteria in plain language, test the judge
            against sample traces to see how it scores, iterate on the
            instructions until it matches your expectations, and deploy. You
            can also{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "eval-monitor/scorers/llm-judge/custom-judges/"
              }
              style={{ color: "#007bff", fontWeight: "600" }}
            >
              define judges in code
            </Link>{" "}
            for version control and integration into your evaluation
            pipelines.
          </p>

          {/* ---- Judge Optimization ---- */}
          <h2 id="judge-optimization">Judge Optimization</h2>

          <p>
            Even well-designed judges can disagree with your team's quality
            assessments. A judge might score a response as "correct" when your
            domain experts would flag it as incomplete, or pass outputs that
            don't meet your organization's specific standards. MLflow's judge
            optimization uses human feedback to automatically refine judge
            instructions, improving agreement with human evaluators by
            30-50%.
          </p>

          <p>
            Start with a built-in or custom judge, collect human feedback
            on 20-50 example outputs via the MLflow UI or programmatically,
            run alignment, and deploy the optimized judge. MLflow's{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "eval-monitor/scorers/llm-judge/memalign/"
              }
            >
              MemAlign optimizer
            </Link>{" "}
            (experimental) uses a dual-memory system that learns from a
            handful of feedback examples instead of requiring hundreds of
            labeled samples. One memory stores general guidelines extracted
            from your feedback; the other stores specific edge cases that
            the guidelines alone don't cover.
          </p>

          <p>
            Judge optimization is most valuable when you're moving from
            development to production and need judges that reflect your
            organization's actual quality bar, not just generic best
            practices.
          </p>

          {/* ---- Common Use Cases ---- */}
          <h2 id="common-use-cases">
            Common Use Cases for LLM-as-a-Judge
          </h2>

          <p>
            Every production LLM system needs evaluation. If the check is
            deterministic (exact match, regex, JSON schema validation), a{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/"}>
              code-based scorer
            </Link>{" "}
            is the right tool. But most of the things
            that matter in production (did the response hallucinate? did it
            follow your brand guidelines? was it actually helpful?) can only
            be assessed by something that understands language.
          </p>

          <ul>
            <li>
              <strong>Your customer support bot keeps making up return
              policies.</strong> Run{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/llm-judge/predefined"}>
                RetrievalGroundedness
              </Link>{" "}
              on every response to catch when the model invents facts that
              aren't in your knowledge base. Identify whether the problem is
              bad retrieval or bad generation.
            </li>
            <li>
              <strong>You rewrote your prompt and need to know if it's
              actually better.</strong> Evaluate both versions across 500 test
              inputs with Correctness and RelevanceToQuery judges. MLflow
              tracks both runs side by side so you can compare scores and
              deploy with confidence.
            </li>
            <li>
              <strong>Your agent is burning through API credits.</strong> ToolCallEfficiency
              identifies redundant tool calls and unnecessary reasoning loops.
              Teams commonly find agents making 3-4x more LLM calls than
              needed before optimizing.
            </li>
            <li>
              <strong>Legal needs to sign off before you ship to
              production.</strong> Define compliance rules with Guidelines
              judges: "never provide specific medical dosage
              recommendations" or "always include a disclaimer for financial
              advice." Run these on every output automatically.
            </li>
            <li>
              <strong>Users are abandoning your chatbot mid-conversation.</strong>{" "}
              Multi-turn judges like UserFrustration and KnowledgeRetention
              reveal whether the agent is losing context, going in circles,
              or failing to resolve issues across conversation turns.
            </li>
          </ul>

          {/* ---- How to Implement ---- */}
          <h2 id="how-to-implement">
            How to Implement LLM-as-a-Judge
          </h2>

          <p>
            Getting started with LLM-as-a-judge in MLflow takes a few lines
            of code. Check out the{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
              evaluation documentation
            </Link>{" "}
            for detailed guides and API reference.
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
              <CopyButton code={BUILTIN_CODE} />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={BUILTIN_CODE}
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
            <strong>Evaluation with Third-Party Judges</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton code={THIRD_PARTY_CODE} />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={THIRD_PARTY_CODE}
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
            <strong>Evaluation with Custom Judges</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton code={CUSTOM_CODE} />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={CUSTOM_CODE}
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
            <strong>Judge Alignment with MemAlign</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton code={MEMALIGN_CODE} />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={MEMALIGN_CODE}
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
              <strong>
                AI engineering platform for agents, LLMs, and ML models
              </strong>
              , with over 30 million monthly downloads. Thousands of
              organizations use MLflow to debug, evaluate, monitor, and optimize
              production-quality AI agents and LLM applications while
              controlling costs and managing access to models and data. Backed
              by the Linux Foundation and licensed under Apache 2.0, MLflow
              provides built-in judges, custom judge creation, and judge
              optimization with no vendor lock-in.{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                Get started →
              </Link>
            </p>
          </div>

          {/* ---- FAQ ---- */}
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

          {/* ---- Related Resources ---- */}
          <h2>Related Resources</h2>

          <ul>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                LLM Evaluation Documentation
              </Link>
            </li>
            <li>
              <Link
                href={
                  MLFLOW_GENAI_DOCS_URL +
                  "eval-monitor/scorers/llm-judge/predefined"
                }
              >
                Built-in LLM Judges Reference
              </Link>
            </li>
            <li>
              <Link
                href={
                  MLFLOW_GENAI_DOCS_URL +
                  "eval-monitor/scorers/llm-judge/custom-judges/"
                }
              >
                Custom Judge Creation Guide
              </Link>
            </li>
            <li>
              <Link href="/llm-evaluation">LLM & Agent Evaluation</Link>
            </li>
            <li>
              <Link href="/ai-observability">AI Observability</Link>
            </li>
            <li>
              <Link href="/llm-tracing">LLM Tracing</Link>
            </li>
            <li>
              <Link href="/genai">MLflow for Agents and LLMs</Link>
            </li>
          </ul>
        </div>

        <ArticleSidebar />
        <SocialLinksFooter />
      </div>
    </>
  );
}
