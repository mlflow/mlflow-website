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
  "Prompt Optimization: Automate Prompt Engineering | MLflow AI Platform";
const SEO_DESCRIPTION =
  "Automate prompt engineering with MLflow. Use optimizers like GEPA to systematically improve prompts for LLM applications without manual guesswork.";

const faqs: {
  question: string;
  answer: React.ReactNode;
  answerText?: string;
}[] = [
  {
    question:
      "Why use MLflow for prompt optimization instead of using GEPA or DSPy directly?",
    answer:
      "MLflow wraps optimizers like GEPA and DSPy behind a single API, so you can try different algorithms without rewriting your code. More importantly, MLflow tracks every optimization run: each prompt version is saved in the Prompt Registry with full diff history, evaluation scores are logged for comparison across runs, and traces let you inspect individual predictions. This means you can see exactly what changed, measure whether it helped, and roll back to any previous version at any time.",
  },
  {
    question: "What is prompt optimization?",
    answer:
      "Prompt optimization is the automated process of improving prompts for LLM applications using data-driven algorithms instead of manual trial-and-error. Optimizers analyze prompt performance on training data, identify failure patterns, and generate improved prompt variants iteratively until quality converges.",
  },
  {
    question: "How is prompt optimization different from prompt engineering?",
    answer:
      "Prompt engineering is a manual process where a human writes and tweaks prompts based on intuition and spot-checking outputs. Prompt optimization automates this by running algorithms that systematically test, analyze, and improve prompts across hundreds of examples. It replaces guesswork with measurable, reproducible improvement.",
  },
  {
    question: "What is GEPA?",
    answer:
      "GEPA (Gradient-free Estimated Prompt-optimization Algorithm) is an optimization algorithm that iteratively improves prompts by evaluating them on training examples, analyzing failure patterns, generating improved variants, and selecting the best performer. It works with any LLM application whose prompts are registered in the MLflow Prompt Registry.",
  },
  {
    question: "What is DSPy?",
    answer:
      "DSPy is a framework for programming language models that provides optimization techniques like MIPROv2 and SIMBA. MLflow integrates with DSPy so you can use its optimizers on any agent framework through the MLflow Prompt Registry, with full tracking, versioning, and comparison in the MLflow UI.",
  },
  {
    question: "How much training data do I need for prompt optimization?",
    answer:
      "GEPA works well with 50-100 labeled examples. Each example should include inputs and an expected output so the optimizer can measure prompt quality and identify failure patterns.",
  },
  {
    question: "Is prompt optimization free with MLflow?",
    answer:
      "The MLflow optimization APIs are 100% free and open source under the Apache 2.0 license. However, the optimizers call LLMs during the optimization process (for reflection and evaluation), so you will incur API costs from your LLM provider.",
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
    "Open-source AI platform with automated prompt optimization for LLM applications and agents.",
  url: "https://mlflow.org",
  license: "https://www.apache.org/licenses/LICENSE-2.0",
};

const GEPA_CODE = `import mlflow
from mlflow.genai.optimize import GepaPromptOptimizer
from mlflow.genai.scorers import Correctness

# Register a base prompt in the MLflow Prompt Registry
base_prompt = mlflow.genai.register_prompt(
    name="qa-prompt",
    template=(
        "Answer the question based on the context.\\n\\n"
        "Context: {{ context }}\\n"
        "Question: {{ question }}\\n\\n"
        "Answer:"
    ),
)

# Prepare training data with expected outputs
train_data = [
    {
        "inputs": {"context": "MLflow is an open source AI platform.",
                   "question": "What is MLflow?"},
        "expectations": {"expected_response": "An open source AI platform"},
    },
    # ... more labeled examples (50-100 recommended)
]

# Run GEPA optimization
result = mlflow.genai.optimize_prompts(
    predict_fn=my_predict_fn,
    train_data=train_data,
    prompt_uris=[base_prompt.uri],
    optimizer=GepaPromptOptimizer(
        reflection_model="openai:/gpt-5.2",
        max_metric_calls=500,
    ),
    # LLM judge that scores each candidate prompt's responses;
    # the optimizer uses these scores as a reward signal
    # to guide its search and identify prompt improvements
    scorers=[Correctness()],
    enable_tracking=True,
)

# Print the optimized prompt
optimized = mlflow.genai.load_prompt(result.optimized_prompts[0].uri)
print(optimized.template)`;

export default function PromptOptimization() {
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
          content="https://mlflow.org/prompt-optimization"
        />
        <link rel="canonical" href="https://mlflow.org/prompt-optimization" />
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
            margin: 48px 0 28px 0 !important;
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
          .steps-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin: 32px 0;
          }
          .step-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            padding: 24px;
            box-shadow: 0 0 0 1px rgba(50, 50, 93, 0.05), 0 0 14px 5px rgba(50, 50, 93, 0.08), 0 0 10px 3px rgba(0, 0, 0, 0.05);
          }
          .step-number {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
            color: #0194e2;
          }
          .step-title {
            font-weight: 600;
            margin-bottom: 8px;
            color: #1a1a1a;
            font-size: 16px;
          }
          .step-desc {
            font-size: 14px;
            color: #505050;
            line-height: 1.5;
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
            .steps-grid {
              grid-template-columns: 1fr 1fr;
              gap: 16px;
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
          <h1>Prompt Optimization</h1>

          <p>
            Prompt optimization automates the prompt engineering process for LLM
            applications. Traditional prompt engineering relies on manual "guess
            and check": a human writes a prompt, spot-checks a few outputs,
            tweaks wording, and repeats. Prompt optimization replaces this with
            algorithms that systematically analyze performance on training data,
            identify failure patterns, and generate improved prompt variants
            until quality converges.
          </p>

          <h2 id="why-optimize">Why Optimize Prompts?</h2>

          <p>
            Manual prompt engineering is slow, inconsistent, and hard to scale.
            Prompt optimization addresses these challenges:
          </p>

          <div className="grid-2">
            <div className="card">
              <h3>Manual Guesswork</h3>
              <p>
                <strong>Problem:</strong> Engineers tweak prompts based on
                intuition and spot-check a handful of outputs, missing failure
                patterns that only appear at scale.
              </p>
              <p>
                <strong>Solution:</strong> Optimizers evaluate prompts across
                hundreds of examples, systematically finding and fixing issues
                that humans miss.
              </p>
            </div>

            <div className="card">
              <h3>Unreproducible Results</h3>
              <p>
                <strong>Problem:</strong> There's no record of which prompt
                variants were tried, what worked, or why changes were made.
                Knowledge is lost when team members change.
              </p>
              <p>
                <strong>Solution:</strong> Every optimization run, prompt
                version, and metric is tracked in MLflow, creating a complete
                audit trail.
              </p>
            </div>

            <div className="card">
              <h3>Scaling to New Tasks</h3>
              <p>
                <strong>Problem:</strong> A team with 10 LLM features can't
                afford to have an engineer manually tuning each prompt
                indefinitely.
              </p>
              <p>
                <strong>Solution:</strong> Automated optimization runs in
                minutes to hours, and can be re-run whenever models change or
                requirements evolve.
              </p>
            </div>

            <div className="card">
              <h3>Diminishing Returns</h3>
              <p>
                <strong>Problem:</strong> Human prompt engineers often plateau
                quickly, unable to identify the subtle instruction changes that
                would improve edge cases.
              </p>
              <p>
                <strong>Solution:</strong> Algorithms like GEPA analyze failure
                patterns at scale and generate targeted improvements for
                specific edge cases.
              </p>
            </div>
          </div>

          <h2 id="how-it-works">How Does Prompt Optimization Work?</h2>

          <p>
            Prompt optimizers work by running an automated loop that no human
            could replicate at scale. Instead of one person tweaking a prompt
            and eyeballing a few outputs, an optimizer tests the prompt against
            hundreds of examples, uses an LLM to figure out what went wrong,
            rewrites the prompt to fix those problems, and repeats. Leading
            optimizers like GEPA and DSPy's MIPROv2 all follow this same
            four-step cycle:
          </p>

          <div className="steps-grid">
            <div className="step-card">
              <div className="step-number">1</div>
              <div className="step-title">Evaluate</div>
              <div className="step-desc">
                Run the current prompt against training examples and score each
                output. This produces a performance baseline and a set of
                failure cases where the prompt fell short.
              </div>
            </div>

            <div className="step-card">
              <div className="step-number">2</div>
              <div className="step-title">Analyze Failures</div>
              <div className="step-desc">
                An LLM reads each failure, including the full execution trace,
                and diagnoses why the prompt failed: missing instructions,
                ambiguous wording, or unhandled edge cases.
              </div>
            </div>

            <div className="step-card">
              <div className="step-number">3</div>
              <div className="step-title">Generate Candidates</div>
              <div className="step-desc">
                The optimizer generates new prompt variants that target the
                specific weaknesses found in step 2, while maintaining diversity
                to avoid local optima.
              </div>
            </div>

            <div className="step-card">
              <div className="step-number">4</div>
              <div className="step-title">Select & Repeat</div>
              <div className="step-desc">
                Evaluate all candidates, select the best as the new baseline,
                and repeat the loop until quality converges or a budget limit is
                reached.
              </div>
            </div>
          </div>

          <h2 id="how-to-implement">How to Implement Prompt Optimization</h2>

          <p>
            <Link href="/genai">MLflow</Link> provides a unified{" "}
            <Link
              href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/optimize-prompts/"}
            >
              prompt optimization API
            </Link>{" "}
            (
            <Link
              href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/optimize-prompts/"}
            >
              <code>mlflow.genai.optimize_prompts</code>
            </Link>
            ) that wraps optimizers like GEPA and DSPy behind a single
            interface, so you can try different algorithms without changing your
            code. The process integrates tightly with the{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/"}>
              MLflow Prompt Registry
            </Link>
            : you register a base prompt, the optimizer generates and tests
            improved variants, and the best result is saved as a new prompt
            version. Every run, metric, and trace is tracked automatically, so
            you can compare performance across versions, inspect individual
            predictions, and roll back to any previous prompt at any time.
          </p>

          <p>
            The workflow works with any agent framework (
            <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/openai-agent.html">
              OpenAI Agents SDK
            </a>
            ,{" "}
            <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langchain.html">
              LangChain
            </a>
            ,{" "}
            <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langgraph.html">
              LangGraph
            </a>
            ,{" "}
            <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/crewai.html">
              CrewAI
            </a>
            , etc.) and any LLM provider. You provide training data with
            expected outputs, choose an optimizer algorithm, and MLflow handles
            the rest.
          </p>

          <h3 id="example">Example</h3>

          <p>
            The following example uses{" "}
            <Link
              href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/optimize-prompts/"}
            >
              GEPA
            </Link>{" "}
            (Gradient-free Estimated Prompt-optimization Algorithm) to optimize
            a question-answering prompt. GEPA evaluates the current prompt on
            training examples, analyzes failure patterns, generates improved
            variants, selects the best performer, and repeats until convergence.
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton code={GEPA_CODE} />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={GEPA_CODE}
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
            With <code>enable_tracking=True</code>, every optimization run is
            logged to MLflow. You can compare evaluation scores across runs, see
            exactly how the optimizer improved the prompt, and roll back to any
            previous version at any time.
          </p>

          <div style={{ margin: "32px 0" }}>
            <img
              src="/img/prompt-opt-eval-comparison.png"
              alt="MLflow evaluation comparison showing pass rate improving from 46% to 60% after prompt optimization"
              style={{
                width: "100%",
                borderRadius: "4px",
                border: "1px solid #e5e7eb",
                boxShadow:
                  "0 0 0 1px rgba(50, 50, 93, 0.05), 0 0 14px 5px rgba(50, 50, 93, 0.08), 0 0 10px 3px rgba(0, 0, 0, 0.05)",
              }}
            />
            <p
              style={{
                fontSize: "14px",
                color: "#6b7280",
                marginTop: "12px",
                textAlign: "center",
              }}
            >
              MLflow tracks performance results across prompt versions for easy
              comparison.
            </p>
          </div>

          <div style={{ margin: "32px 0" }}>
            <img
              src="/img/prompt-opt-prompt-comparison.png"
              alt="MLflow prompt version comparison showing diff between base and optimized prompt"
              style={{
                width: "100%",
                borderRadius: "4px",
                border: "1px solid #e5e7eb",
                boxShadow:
                  "0 0 0 1px rgba(50, 50, 93, 0.05), 0 0 14px 5px rgba(50, 50, 93, 0.08), 0 0 10px 3px rgba(0, 0, 0, 0.05)",
              }}
            />
            <p
              style={{
                fontSize: "14px",
                color: "#6b7280",
                marginTop: "12px",
                textAlign: "center",
              }}
            >
              Optimized prompts are versioned in the MLflow Prompt Registry with
              full diff highlighting.
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
              Apache 2.0, MLflow provides automated prompt optimization with no
              vendor lock-in.{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL}>Get started →</Link>
            </p>
          </div>

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
              <Link
                href={
                  MLFLOW_GENAI_DOCS_URL + "prompt-registry/optimize-prompts/"
                }
              >
                Prompt Optimization Documentation
              </Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/"}>
                Prompt Registry Documentation
              </Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                Evaluation and Monitoring Guide
              </Link>
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
