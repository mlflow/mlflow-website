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
import PromptHero from "@site/static/img/GenAI_prompts/prompt-registry-hero.png";
import PromptCreateUI from "@site/static/img/GenAI_prompts/GenAI_prompts_1.png";

const SEO_TITLE = "Prompt Registry for LLMs & Agents | MLflow Agent Platform";
const SEO_DESCRIPTION =
  "Learn prompt management, prompt versioning, and prompt engineering with MLflow—the comprehensive, open-source agent engineering and ops platform.";

const faqs: {
  question: string;
  answer: React.ReactNode;
  answerText?: string;
}[] = [
  {
    question: "What is a prompt registry?",
    answer:
      "A prompt registry is a centralized repository for storing, versioning, and managing prompt templates across their lifecycle in LLM and agent applications. It treats prompts as first-class artifacts with version control, commit messages, metadata, and environment aliases (development, staging, production). A prompt registry decouples prompts from application code, enabling teams to iterate on prompts without redeploying applications.",
  },
  {
    question: "What is prompt management?",
    answer:
      "Prompt management is the overarching discipline of organizing, versioning, testing, evaluating, and deploying prompts across an organization's AI applications. It encompasses prompt registries (centralized storage), prompt versioning (change tracking), prompt evaluation (quality testing), and prompt optimization (automated improvement). Effective prompt management reduces engineering bottlenecks and ensures consistent prompt quality across teams.",
  },
  {
    question: "What is prompt versioning?",
    answer:
      "Prompt versioning is the practice of systematically tracking and controlling changes to prompts over time. Unlike code versioning, prompt versioning must account for the non-deterministic nature of LLM outputs — the same prompt change can produce wildly different results. Good prompt versioning includes commit messages, diff views for comparing versions, and the ability to roll back to previous versions when quality degrades.",
  },
  {
    question: "What is prompt engineering?",
    answer: (
      <>
        Prompt engineering is the process of designing, structuring, and
        optimizing the text instructions sent to LLMs to produce desired
        outputs. A <Link href="/prompt-registry">prompt registry</Link> supports
        prompt engineering by providing version control, evaluation tools, and
        collaboration features that make the iterative process of refining
        prompts more systematic and reproducible.
      </>
    ),
    answerText:
      "Prompt engineering is the process of designing, structuring, and optimizing the text instructions sent to LLMs to produce desired outputs. A prompt registry supports prompt engineering by providing version control, evaluation tools, and collaboration features that make the iterative process of refining prompts more systematic and reproducible.",
  },
  {
    question:
      "How is a prompt registry different from storing prompts in code?",
    answer:
      "Storing prompts in application code tightly couples prompt changes to code deployments. Every prompt edit requires a code review, merge, and deploy cycle. A prompt registry decouples prompts from code: teams can update prompts through a UI or API, version changes independently, evaluate new versions against quality benchmarks, and promote them through environments (dev → staging → production) without touching application code. This dramatically speeds up prompt iteration cycles.",
  },
  {
    question: "Does MLflow support prompt optimization?",
    answer: (
      <>
        Yes. MLflow includes{" "}
        <Link
          href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/prompt-optimization/"}
        >
          automatic prompt optimization
        </Link>{" "}
        powered by the GEPA (Generalized Efficient Prompt Adaptation) algorithm.
        You define your evaluation criteria, provide a dataset, and MLflow
        automatically generates improved prompt variants and selects the best
        one. This has been shown to improve accuracy by 10-15% without manual
        prompt engineering.
      </>
    ),
    answerText:
      "Yes. MLflow includes automatic prompt optimization powered by the GEPA (Generalized Efficient Prompt Adaptation) algorithm. You define your evaluation criteria, provide a dataset, and MLflow automatically generates improved prompt variants and selects the best one. This has been shown to improve accuracy by 10-15% without manual prompt engineering.",
  },
  {
    question:
      "How do I evaluate prompt changes in MLflow before deploying them?",
    answer: (
      <>
        MLflow integrates prompt versioning with{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>evaluation</Link>.
        When you create a new prompt version, you can run it against a test
        dataset using LLM judges that score quality metrics like relevance,
        correctness, and safety. Compare scores across prompt versions
        side-by-side before promoting the new version to production.
      </>
    ),
    answerText:
      "MLflow integrates prompt versioning with evaluation. When you create a new prompt version, you can run it against a test dataset using LLM judges that score quality metrics like relevance, correctness, and safety. Compare scores across prompt versions side-by-side before promoting the new version to production.",
  },
  {
    question: "Can non-technical team members edit prompts in MLflow?",
    answer:
      "Yes. MLflow's Prompt Registry provides a UI-based editor where domain experts, product managers, and other non-technical team members can create and edit prompts directly. Changes are versioned with commit messages, so engineers maintain full visibility into what changed and why. This eliminates engineering bottlenecks and lets the people closest to the domain iterate on prompt quality.",
  },
  {
    question: "How does MLflow's prompt registry work with agents?",
    answer: (
      <>
        Agents built with frameworks like LangGraph, CrewAI, or OpenAI Agents
        SDK rely on system prompts and tool instructions that define agent
        behavior. The prompt registry stores and versions these prompts
        separately from agent code. When you load a prompt at runtime using{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/"}>
          mlflow.genai.load_prompt()
        </Link>
        , you can specify an alias (e.g., &quot;production&quot;) to control
        which version your agent uses. This lets you update agent instructions
        without redeploying agent code.
      </>
    ),
    answerText:
      'Agents built with frameworks like LangGraph, CrewAI, or OpenAI Agents SDK rely on system prompts and tool instructions that define agent behavior. The prompt registry stores and versions these prompts separately from agent code. When you load a prompt at runtime using mlflow.genai.load_prompt(), you can specify an alias (e.g., "production") to control which version your agent uses. This lets you update agent instructions without redeploying agent code.',
  },
  {
    question: "How do I get started with prompt management in MLflow?",
    answer: (
      <>
        Getting started with MLflow's Prompt Registry takes just a few lines of
        code. Install MLflow with{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/"}>
          pip install &apos;mlflow[genai]&apos;
        </Link>
        , register your first prompt with{" "}
        <code>mlflow.genai.register_prompt()</code>, and load it in your
        application with <code>mlflow.genai.load_prompt()</code>. You can also
        create and manage prompts through the MLflow UI. See the{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/"}>
          prompt registry documentation
        </Link>{" "}
        for complete examples.
      </>
    ),
    answerText:
      "Getting started with MLflow's Prompt Registry takes just a few lines of code. Install MLflow with pip install 'mlflow[genai]', register your first prompt with mlflow.genai.register_prompt(), and load it in your application with mlflow.genai.load_prompt(). You can also create and manage prompts through the MLflow UI. See the prompt registry documentation for complete examples.",
  },
  {
    question:
      "How does MLflow's prompt registry integrate with tracing and observability?",
    answer: (
      <>
        MLflow automatically links prompts to{" "}
        <Link href="/llm-tracing">traces</Link>. When your application loads a
        prompt from the registry and uses it in an LLM call, MLflow records
        which prompt version was used in the trace. This creates a complete
        audit trail connecting prompt versions to application behavior, making
        it easy to debug quality issues and understand the impact of prompt
        changes on production metrics.
      </>
    ),
    answerText:
      "MLflow automatically links prompts to traces. When your application loads a prompt from the registry and uses it in an LLM call, MLflow records which prompt version was used in the trace. This creates a complete audit trail connecting prompt versions to application behavior, making it easy to debug quality issues and understand the impact of prompt changes on production metrics.",
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
    "Open-source platform for prompt management, AI observability, experiment tracking, evaluation, and deployment.",
  url: "https://mlflow.org",
  license: "https://www.apache.org/licenses/LICENSE-2.0",
};

export default function PromptRegistry() {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://mlflow.org/prompt-registry" />
        <link rel="canonical" href="https://mlflow.org/prompt-registry" />
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
          <h1>Prompt Registry for LLM and Agent Applications</h1>

          <p>
            A prompt registry is a centralized repository for storing,
            versioning, and managing the prompt templates that power LLM and
            agent applications. For teams managing{" "}
            <a href="#prompt-management">prompt engineering</a> at scale, a
            prompt registry is the foundational infrastructure — it provides{" "}
            <a href="#prompt-versioning">prompt versioning</a> with diff views
            and commit messages, evaluation integration for quality testing, and
            environment aliases for safe deployments. Prompt registries decouple
            prompts from application code, enabling faster iteration without
            redeployments. A prompt registry is a core component of{" "}
            <Link href="/ai-observability">AI observability</Link> and{" "}
            <Link href="/llmops">LLMOps</Link>.
          </p>

          <div style={{ margin: "40px 0", textAlign: "center" }}>
            <img
              src={PromptHero}
              alt="MLflow Prompt Registry UI showing versioned prompt templates with version history sidebar and prompt details"
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
              MLflow Prompt Registry: create, version, and manage prompt
              templates with a built-in UI
            </p>
          </div>

          <h2 id="prompt-management">Prompt Management</h2>

          <p>
            Prompt management is the discipline of organizing, versioning,
            testing, and deploying prompts across an organization&apos;s AI
            applications. It encompasses all the operational work around prompt
            engineering: storing prompt templates in a central registry,{" "}
            <a href="#prompt-versioning">versioning every change</a> with commit
            messages and diffs, evaluating prompt quality against benchmarks,
            and promoting tested prompts through environments. As LLM-powered
            applications scale from prototypes to production, managing prompt
            engineering workflows becomes as critical as managing code.
          </p>

          <p>
            Effective prompt management solves three problems. First, it
            eliminates engineering bottlenecks: domain experts can iterate on
            prompts through a UI without waiting for code deployments. Second,
            it ensures consistency: every team member works from the same
            versioned prompts rather than local copies scattered across
            notebooks and scripts. Third, it enables quality gates: new prompt
            versions can be evaluated against benchmarks before reaching
            production.{" "}
            <Link
              href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/"}
              style={{ color: "#007bff", fontWeight: "600" }}
            >
              MLflow&apos;s Prompt Registry
            </Link>{" "}
            provides a complete prompt management solution: a centralized
            registry with version control, a UI for non-technical editors,{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
              evaluation integration
            </Link>{" "}
            for quality testing, and environment aliases for safe deployment
            workflows.
          </p>

          <h2 id="prompt-versioning">Prompt Versioning</h2>

          <p>
            Prompt versioning is one of the most important parts of{" "}
            <a href="#prompt-management">prompt management</a>. It tracks every
            change to a prompt template with commit messages, timestamps, and
            metadata. Unlike code versioning, prompt versioning must account for
            the non-deterministic nature of LLM outputs: a small wording change
            in a prompt can dramatically alter model behavior. This makes robust
            versioning essential for any prompt engineering workflow.
          </p>

          <p>
            Good prompt versioning provides diff views to compare versions
            side-by-side, immutable version history for reproducibility, the
            ability to roll back to any previous version, and integration with
            evaluation tools to measure the impact of each change.{" "}
            <Link
              href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/"}
              style={{ color: "#007bff", fontWeight: "600" }}
            >
              MLflow&apos;s prompt versioning
            </Link>{" "}
            goes beyond simple version control: each version can be tagged with
            aliases like &quot;production&quot; or &quot;staging,&quot; allowing
            teams to promote tested prompt versions through environments without
            redeploying application code.
          </p>

          <h2 id="why-prompt-registry-matters">
            Why a Prompt Registry Matters
          </h2>

          <p>
            AI applications — agents, LLM applications, and RAG systems — rely
            on prompts as core configuration. As prompt engineering efforts
            scale, teams without a prompt registry face compounding problems:
          </p>

          <div className="grid-2">
            <div className="card">
              <h3>Slow Iteration Cycles</h3>
              <p>
                <strong>Problem:</strong> Prompts hardcoded in application code
                require a full deploy cycle for every change, even minor wording
                tweaks.
              </p>
              <p>
                <strong>Solution:</strong> Decouple prompts from code. Load them
                at runtime from the registry so changes take effect immediately.
              </p>
            </div>

            <div className="card">
              <h3>No Quality Gates</h3>
              <p>
                <strong>Problem:</strong> Prompt changes go straight to
                production without testing, causing regressions in output
                quality, safety, or accuracy.
              </p>
              <p>
                <strong>Solution:</strong> Evaluate new prompt versions against
                benchmarks before promoting them to production.
              </p>
            </div>

            <div className="card">
              <h3>Team Bottlenecks</h3>
              <p>
                <strong>Problem:</strong> Only engineers can update prompts,
                blocking domain experts who understand the content best.
              </p>
              <p>
                <strong>Solution:</strong> Provide a UI for non-technical team
                members to edit prompts, with version control for full
                visibility.
              </p>
            </div>

            <div className="card">
              <h3>No Reproducibility</h3>
              <p>
                <strong>Problem:</strong> Without version history, teams
                can&apos;t reproduce past behavior, debug regressions, or
                understand why outputs changed.
              </p>
              <p>
                <strong>Solution:</strong> Store every prompt version immutably
                with commit messages, metadata, and the ability to roll back.
              </p>
            </div>
          </div>

          <h2>Common Use Cases for Prompt Registries</h2>

          <p>
            Prompt registries solve real-world prompt engineering challenges
            across the AI development lifecycle:
          </p>

          <ul>
            <li>
              <strong>Iterating on Agent System Prompts:</strong> When building{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>agents</Link>{" "}
              with LangGraph, CrewAI, or OpenAI Agents SDK, the system prompt
              defines agent behavior. A prompt registry lets you version and
              test system prompt changes without redeploying your agent, then
              roll back if quality degrades.
            </li>
            <li>
              <strong>A/B Testing Prompt Variants:</strong> Before deploying a
              prompt change, create multiple versions and{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                evaluate them
              </Link>{" "}
              against the same test dataset. Compare quality scores side-by-side
              to pick the best variant before it reaches users.
            </li>
            <li>
              <strong>Multi-Environment Deployment:</strong> Use aliases like
              &quot;development,&quot; &quot;staging,&quot; and
              &quot;production&quot; to promote prompt versions through
              environments. Test changes in staging before they reach production
              users, and roll back instantly if quality degrades.
            </li>
            <li>
              <strong>Enabling Domain Expert Collaboration:</strong> Product
              managers, legal teams, and domain experts often have the best
              understanding of what a prompt should say. A prompt registry lets
              them edit prompts through a UI while engineers maintain version
              control and deployment governance.
            </li>
            <li>
              <strong>Automated Prompt Optimization:</strong> Instead of
              manually iterating on prompts, use{" "}
              <Link
                href={
                  MLFLOW_GENAI_DOCS_URL + "prompt-registry/prompt-optimization/"
                }
              >
                MLflow&apos;s automatic prompt optimization
              </Link>{" "}
              to generate improved variants programmatically. Define your
              evaluation criteria, provide a dataset, and let the optimizer find
              a better prompt.
            </li>
            <li>
              <strong>Compliance and Audit Trails:</strong> Regulated industries
              need to track what prompts were used when and by whom. A prompt
              registry provides a complete audit trail of every version change,
              who made it, and which environments it was deployed to.
            </li>
          </ul>

          <h2 id="how-to-implement">How to Implement a Prompt Registry</h2>

          <p>
            MLflow offers both a UI and an API for prompt engineering workflows.
            Non-technical team members can create and edit prompts directly
            through the UI, while engineers can use the Python API for
            programmatic workflows. Here are quick examples showing both
            approaches. Check out the{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/"}>
              MLflow prompt registry documentation
            </Link>{" "}
            for complete guides including prompt evaluation and optimization
            workflows.
          </p>

          <div style={{ margin: "40px 0", textAlign: "center" }}>
            <img
              src={PromptCreateUI}
              alt="MLflow Prompt Registry create prompt dialog showing name, template, and commit message fields"
              style={{
                maxWidth: "560px",
                width: "100%",
                borderRadius: "8px",
                boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
                display: "block",
                margin: "0 auto",
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
              Non-technical users can create prompts directly through the MLflow
              UI — no code required
            </p>
          </div>

          <p style={{ marginTop: "32px", marginBottom: "0px" }}>
            <strong>Register a prompt</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton
                code={`import mlflow

# Register a prompt template with variables
prompt = mlflow.genai.register_prompt(
    name="qa-assistant",
    template="Answer the question based on the context.\\n\\nContext: {{context}}\\n\\nQuestion: {{question}}",
    commit_message="Initial QA prompt template",
)

print(f"Registered prompt version: {prompt.version}")`}
              />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={`import mlflow

# Register a prompt template with variables
prompt = mlflow.genai.register_prompt(
    name="qa-assistant",
    template="Answer the question based on the context.\\n\\nContext: {{context}}\\n\\nQuestion: {{question}}",
    commit_message="Initial QA prompt template",
)

print(f"Registered prompt version: {prompt.version}")`}
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
            <strong>Load and use a prompt at runtime</strong>
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

# Load the production version of a prompt
prompt = mlflow.genai.load_prompt("prompts:/qa-assistant/production")

# Fill in template variables
filled_prompt = prompt.format(
    context="MLflow is an open source AI platform.",
    question="What is MLflow?",
)

# Use with any LLM provider
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-5-2",
    messages=[{"role": "user", "content": filled_prompt}],
)`}
              />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={`import mlflow
from openai import OpenAI

# Load the production version of a prompt
prompt = mlflow.genai.load_prompt("prompts:/qa-assistant/production")

# Fill in template variables
filled_prompt = prompt.format(
    context="MLflow is an open source AI platform.",
    question="What is MLflow?",
)

# Use with any LLM provider
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-5-2",
    messages=[{"role": "user", "content": filled_prompt}],
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
            <strong>Evaluate prompt versions</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton
                code={`import mlflow
from mlflow.genai.scorers import Correctness

# Define evaluation data with expected outputs
eval_data = [
    {
        "inputs": {"question": "What is MLflow?"},
        "outputs": {"response": "MLflow is an open source AI platform."},
        "expectations": {"expected_response": "MLflow is an open source AI platform."},
    },
]

# Score outputs with LLM judges
results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[Correctness()],
)`}
              />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={`import mlflow
from mlflow.genai.scorers import Correctness

# Define evaluation data with expected outputs
eval_data = [
    {
        "inputs": {"question": "What is MLflow?"},
        "outputs": {"response": "MLflow is an open source AI platform."},
        "expectations": {"expected_response": "MLflow is an open source AI platform."},
    },
]

# Score outputs with LLM judges
results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[Correctness()],
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
              is the largest open-source{" "}
              <strong>
                AI engineering platform for agents, LLMs, and ML models
              </strong>
              , with over 30 million monthly downloads. Thousands of
              organizations use MLflow to manage prompts, evaluate AI quality,
              trace agent behavior, and deploy production-grade AI applications.
              Backed by the Linux Foundation and licensed under Apache 2.0,
              MLflow provides a complete prompt management solution with no
              vendor lock-in.{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/"}>
                Get started →
              </Link>
            </p>
          </div>

          <h2>Open Source vs. Proprietary Prompt Registries</h2>

          <p>
            When choosing a prompt registry, the decision between open source
            and proprietary SaaS tools has significant long-term implications
            for your team, data ownership, and costs.
          </p>

          <p>
            <strong>
              Open Source (<Link href="/genai">MLflow</Link>):
            </strong>{" "}
            With MLflow, you maintain complete control over your prompts and
            infrastructure. Deploy on your own servers or use managed versions
            on Databricks, AWS, or other platforms. There are no per-seat fees,
            no usage limits, and no vendor lock-in. Your prompt data stays under
            your control, and you get integrated evaluation, tracing, and
            optimization in the same platform. MLflow works with any LLM
            provider and agent framework.
          </p>

          <p>
            <strong>Proprietary SaaS Tools:</strong> Commercial prompt
            management platforms offer convenience but at the cost of
            flexibility and control. They typically charge per seat or per API
            call, which can become expensive as teams grow. Your prompt
            templates and version history are stored on their servers, raising
            IP and compliance concerns. Most proprietary tools only support
            their own ecosystem, making it difficult to integrate with your
            existing evaluation and observability tools.
          </p>

          <p>
            <strong>Why Teams Choose Open Source:</strong> Organizations
            building production AI applications increasingly choose MLflow
            because it offers a complete prompt management platform (registry,
            versioning, evaluation, optimization) without compromising on data
            sovereignty, cost predictability, or flexibility. The Apache 2.0
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
              <Link href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/"}>
                Prompt Registry Documentation
              </Link>
            </li>
            <li>
              <Link
                href={
                  MLFLOW_GENAI_DOCS_URL + "prompt-registry/prompt-optimization/"
                }
              >
                Prompt Optimization Guide
              </Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
                LLM Evaluation Documentation
              </Link>
            </li>
            <li>
              <Link href="/llm-evaluation">Agent Evaluation FAQ</Link>
            </li>
            <li>
              <Link href="/ai-observability">AI Observability</Link>
            </li>
            <li>
              <Link href="/llmops">LLMOps and AgentOps</Link>
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
