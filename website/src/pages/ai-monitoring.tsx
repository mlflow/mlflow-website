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

const SEO_TITLE = "AI Monitoring for LLMs & Agents | MLflow AI Platform";
const SEO_DESCRIPTION =
  "AI monitoring evaluates agent and LLM quality in production, detects drift, and controls costs. Explore AI monitoring on MLflow's open source AI platform.";

const faqs: {
  question: string;
  answer: React.ReactNode;
  answerText?: string;
}[] = [
  // General concept questions
  {
    question: "What is AI monitoring?",
    answer:
      "AI monitoring is the practice of continuously assessing the quality, performance, cost, and safety of AI applications running in production, including LLM and agent-based systems. Unlike traditional software monitoring (uptime, error rates), AI monitoring must evaluate the quality of non-deterministic text outputs, track token costs, detect hallucinations, and identify when model behavior drifts from expected standards.",
  },
  {
    question: "How is AI monitoring different from classical ML monitoring?",
    answer:
      "Classical ML monitoring tracks feature distributions, prediction accuracy on structured outputs, and data drift using statistical tests. AI monitoring must handle free-form language outputs, multi-step agent reasoning, tool call chains, retrieval accuracy, subjective quality dimensions, and token-based costs. You can't use distribution histograms to detect when an LLM starts hallucinating or when an agent picks the wrong tool.",
  },
  {
    question: "What is quality drift in LLM and agent applications?",
    answer:
      "Quality drift occurs when the quality of AI outputs degrades over time without obvious errors. For LLMs and agents, this can manifest as increased hallucinations, declining response relevance, safety regressions, or coherence loss. Root causes include model version updates from providers, prompt template changes, upstream data pipeline modifications, or shifts in user input patterns. Unlike classical ML drift, AI quality drift is often subtle and requires LLM judges to detect.",
  },
  {
    question:
      "Why do agents and LLM applications need continuous monitoring in production?",
    answer:
      "Development evaluation uses curated datasets that represent expected inputs. Production traffic is diverse, unpredictable, and evolves over time. New query patterns, adversarial inputs, and edge cases emerge continuously. Without production evaluation, you can't detect when quality degrades on inputs you didn't anticipate during development. Continuous evaluation catches regressions, emerging failure patterns, and quality drift before users lose trust.",
  },
  {
    question: "What role do traces play in AI monitoring?",
    answer:
      "Traces capture the complete execution graph of every request: LLM calls, tool invocations, retrieval operations, and intermediate reasoning steps, along with inputs, outputs, latency, and token usage at each step. In AI monitoring, traces serve three critical functions: (1) they provide the data that LLM judges evaluate for quality, (2) they enable root-cause analysis when quality drops or errors spike, and (3) they power cost and latency tracking across the full request lifecycle.",
  },
  {
    question:
      "What is trace sampling and why is it important for AI monitoring?",
    answer:
      "Trace sampling controls what percentage of production requests are fully traced and evaluated. Running LLM judges on every request is cost-prohibitive at scale. Sampling strategies include global ratios (e.g., 5% of traffic), per-endpoint overrides (100% on critical endpoints, 1% on high-volume ones), and error-biased sampling (always capture failures). Deterministic safety checks (PII detection, format validation) can still run on 100% of traffic since they're fast and cheap.",
  },
  {
    question: "How do you detect hallucinations in production?",
    answer:
      "Production hallucination detection combines multiple approaches: LLM judges that compare responses against retrieved context (groundedness scoring), consistency checks that verify the same query produces semantically similar outputs, factual verification against known data sources, and pattern detection for common hallucination signatures. The key challenge is that hallucinations are often plausible-sounding, requiring semantic analysis rather than simple pattern matching.",
  },
  {
    question:
      "What metrics should I track for production LLM and agent monitoring?",
    answer:
      "Key metrics span four dimensions: (1) Quality: LLM judge scores (correctness, relevance, safety, groundedness), hallucination rate, task completion rate (agents), user satisfaction scores. (2) Performance: p50/p95/p99 latency, time-to-first-token, error rates, retry rates. (3) Cost: tokens per request, cost per request, cost per task completion, daily/monthly spend trends. (4) Safety: prompt injection detection rate, PII leakage incidents, policy violation rate, jailbreak attempt frequency.",
  },
  {
    question: "How do feedback loops improve AI monitoring?",
    answer:
      "User feedback (thumbs up/down, ratings, corrections) provides ground truth that automated judges cannot. Feedback identifies silent failures where outputs look acceptable to judges but fail users. Over time, feedback is used to: calibrate LLM judges (so automated scores match human expectations), build regression datasets from real failures, discover new failure modes to monitor, and prioritize quality improvements.",
  },
  {
    question: "What security threats should AI monitoring detect?",
    answer:
      "Production LLM applications face unique security risks: prompt injection (users manipulating the model to ignore instructions), PII leakage (model exposing sensitive data from training or retrieval), jailbreaks (bypassing safety guidelines to produce harmful content), and data exfiltration through crafted queries. Monitoring must include input scanning, output scanning, behavioral anomaly detection, and comprehensive audit trails for compliance.",
  },
  {
    question: "What is asynchronous quality evaluation in production?",
    answer:
      "Asynchronous evaluation runs LLM judges on production traces in the background, after the user has already received their response. This prevents quality assessment from adding latency to the request path. Traces are queued, and worker threads evaluate them using configured scorers. Results are stored alongside traces for dashboard visualization and alerting. This architecture lets you evaluate a meaningful sample of production traffic without impacting user experience.",
  },
  {
    question: "How is agent monitoring different from LLM monitoring?",
    answer:
      "LLM monitoring assesses individual model calls: prompt quality, response accuracy, token usage, and latency. Agent monitoring must also track multi-step reasoning trajectories, tool selection accuracy, error recovery behavior, goal completion rates, and execution efficiency. An agent might produce a correct final answer through an inefficient path (calling wrong tools, getting stuck in loops, using excessive retries). Agent monitoring evaluates the full trajectory, not just the final output.",
  },
  // MLflow-specific questions
  {
    question:
      "How does MLflow enable AI monitoring for agents and LLM applications?",
    answer: (
      <>
        MLflow provides an integrated AI monitoring stack: (1) Automatic tracing
        with one-line instrumentation across 50+ frameworks (OpenAI, LangChain,
        Anthropic, LlamaIndex, etc.), (2) Asynchronous trace logging that
        doesn't impact application performance, (3) Automatic online evaluation
        where registered LLM judges (Guidelines, Safety, Correctness, and more)
        score production traces in the background via <code>.register()</code>{" "}
        and <code>.start()</code>, (4) Configurable trace sampling for cost
        control, (5) Automatic token and cost tracking, (6) Human feedback
        collection via <code>log_feedback()</code> linked to traces, (7)
        User/session/request context tracking via{" "}
        <code>update_current_trace()</code>, and (8) OpenTelemetry compatibility
        for data portability.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/prod-tracing/"}>
          Learn more in the production tracing docs
        </Link>
        .
      </>
    ),
    answerText:
      "MLflow provides an integrated AI monitoring stack: (1) Automatic tracing with one-line instrumentation across 50+ frameworks (OpenAI, LangChain, Anthropic, LlamaIndex, etc.), (2) Asynchronous trace logging that doesn't impact application performance, (3) Automatic online evaluation where registered LLM judges score production traces in the background, (4) Configurable trace sampling for cost control, (5) Automatic token and cost tracking, (6) Human feedback collection via log_feedback() linked to traces, (7) User/session/request context tracking via update_current_trace(), and (8) OpenTelemetry compatibility for data portability.",
  },
  {
    question: "What is the MLflow lightweight production tracing SDK?",
    answer: (
      <>
        The <code>mlflow-tracing</code> package is a lightweight production
        tracing SDK that is 95% smaller than full MLflow (approximately 5MB vs
        1000MB). It includes only essential tracing functionality, making it
        ideal for Docker containers, serverless functions, and
        resource-constrained environments. It supports the same one-line
        auto-instrumentation and manual tracing APIs as the full SDK. It works
        with self-hosted MLflow, Databricks, and any OpenTelemetry-compatible
        backend.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/lightweight-sdk"}>
          Learn more about the lightweight SDK
        </Link>
        .
      </>
    ),
    answerText:
      "The mlflow-tracing package is a lightweight production tracing SDK that is 95% smaller than full MLflow (approximately 5MB vs 1000MB). It includes only essential tracing functionality, making it ideal for Docker containers, serverless functions, and resource-constrained environments. It supports the same one-line auto-instrumentation and manual tracing APIs as the full SDK. It works with self-hosted MLflow, Databricks, and any OpenTelemetry-compatible backend.",
  },
  {
    question: "How does MLflow handle trace sampling in production?",
    answer: (
      <>
        MLflow supports configurable trace sampling at two levels. Globally, the{" "}
        <code>MLFLOW_TRACE_SAMPLING_RATIO</code> environment variable (0.0 to
        1.0, default 1.0) controls the default sampling rate. Per-endpoint, use{" "}
        <code>@mlflow.trace(sampling_ratio_override=...)</code> to override the
        global rate for specific functions (e.g., 100% for payment processing,
        10% for high-volume chat). Sampling decisions happen before trace
        submission to minimize overhead. This lets teams balance monitoring
        coverage against computational and storage costs.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/prod-tracing/"}>
          See the production tracing docs
        </Link>
        .
      </>
    ),
    answerText:
      "MLflow supports configurable trace sampling at two levels. Globally, the MLFLOW_TRACE_SAMPLING_RATIO environment variable (0.0 to 1.0, default 1.0) controls the default sampling rate. Per-endpoint, use @mlflow.trace(sampling_ratio_override=...) to override the global rate for specific functions (e.g., 100% for payment processing, 10% for high-volume chat). Sampling decisions happen before trace submission to minimize overhead. This lets teams balance monitoring coverage against computational and storage costs.",
  },
  {
    question: "What built-in scorers does MLflow provide for AI monitoring?",
    answer: (
      <>
        MLflow provides built-in LLM judges across multiple quality dimensions:{" "}
        <Link
          href={
            MLFLOW_GENAI_DOCS_URL +
            "eval-monitor/scorers/llm-judge/response-quality/safety/"
          }
        >
          Safety
        </Link>{" "}
        (harmful content),{" "}
        <Link
          href={
            MLFLOW_GENAI_DOCS_URL +
            "eval-monitor/scorers/llm-judge/response-quality/correctness/"
          }
        >
          Correctness
        </Link>{" "}
        (factual accuracy),{" "}
        <Link
          href={
            MLFLOW_GENAI_DOCS_URL +
            "eval-monitor/scorers/llm-judge/rag/relevance/"
          }
        >
          RelevanceToQuery
        </Link>{" "}
        (response addresses the question),{" "}
        <Link
          href={
            MLFLOW_GENAI_DOCS_URL +
            "eval-monitor/scorers/llm-judge/rag/groundedness/"
          }
        >
          Groundedness
        </Link>{" "}
        (supported by context),{" "}
        <Link
          href={
            MLFLOW_GENAI_DOCS_URL +
            "eval-monitor/scorers/llm-judge/tool-call/efficiency/"
          }
        >
          ToolCallEfficiency
        </Link>{" "}
        (optimal tool usage),{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/"}>
          RoleAdherence
        </Link>{" "}
        (stays in role across turns),{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/"}>
          ConversationalSafety
        </Link>{" "}
        (multi-turn safety), and more. Additionally, MLflow integrates with
        third-party scorer libraries including{" "}
        <Link
          href={
            MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/third-party/deepeval/"
          }
        >
          DeepEval
        </Link>{" "}
        (50+ metrics),{" "}
        <Link
          href={
            MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/third-party/ragas/"
          }
        >
          RAGAS
        </Link>{" "}
        (RAG-specific metrics),{" "}
        <Link
          href={
            MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/third-party/trulens/"
          }
        >
          TruLens
        </Link>{" "}
        (agent trajectory scoring), and{" "}
        <Link
          href={
            MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/third-party/phoenix/"
          }
        >
          Phoenix
        </Link>{" "}
        (hallucination, toxicity).{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/"}>
          Explore all available scorers
        </Link>
        .
      </>
    ),
    answerText:
      "MLflow provides built-in LLM judges across multiple quality dimensions: Safety (harmful content), Correctness (factual accuracy), RelevanceToQuery (response addresses the question), Groundedness (supported by context), ToolCallEfficiency (optimal tool usage), RoleAdherence (stays in role across turns), ConversationalSafety (multi-turn safety), and more. Additionally, MLflow integrates with third-party scorer libraries including DeepEval (50+ metrics), RAGAS (RAG-specific metrics), TruLens (agent trajectory scoring), and Phoenix (hallucination, toxicity).",
  },
  {
    question: "How does MLflow track costs and token usage in production?",
    answer: (
      <>
        MLflow automatically extracts token counts (input, output, total) from
        every LLM span and calculates costs using model-aware pricing. You can
        view cost breakdowns per trace, per model, and across time in the UI.
        The AI Gateway adds per-endpoint usage analytics. For unsupported
        models, you can set custom costs via the span API. This gives teams
        visibility into exactly where money is being spent and helps identify
        optimization opportunities.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/token-usage-cost/"}>
          Learn about cost tracking
        </Link>
        .
      </>
    ),
    answerText:
      "MLflow automatically extracts token counts (input, output, total) from every LLM span and calculates costs using model-aware pricing. You can view cost breakdowns per trace, per model, and across time in the UI. The AI Gateway adds per-endpoint usage analytics. For unsupported models, you can set custom costs via the span API. This gives teams visibility into exactly where money is being spent and helps identify optimization opportunities.",
  },
  {
    question: "How does MLflow detect quality drift in production?",
    answer: (
      <>
        MLflow detects quality drift by continuously running LLM judges on
        production traces and tracking scores over time. When quality metrics
        (correctness pass rate, safety scores, relevance) trend downward
        compared to baselines, teams can investigate using the trace UI. MLflow
        also supports metric backfill, letting you retroactively apply new
        scorers to historical traces to establish baselines and detect when
        drift began. Combined with human feedback, judges can be{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "assessments/feedback/"}>
          calibrated
        </Link>{" "}
        to catch domain-specific quality regressions.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          See the evaluation and monitoring docs
        </Link>
        .
      </>
    ),
    answerText:
      "MLflow detects quality drift by continuously running LLM judges on production traces and tracking scores over time. When quality metrics (correctness pass rate, safety scores, relevance) trend downward compared to baselines, teams can investigate using the trace UI. MLflow also supports metric backfill, letting you retroactively apply new scorers to historical traces to establish baselines and detect when drift began. Combined with human feedback, judges can be calibrated to catch domain-specific quality regressions.",
  },
  {
    question: "Does MLflow support alerting for production quality issues?",
    answer: (
      <>
        MLflow provides the assessment and scoring infrastructure that feeds
        alerting systems. Production scorers generate quality metrics on every
        evaluated trace. These metrics can be queried programmatically via{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/search-traces/"}>
          <code>mlflow.search_traces()</code>
        </Link>{" "}
        and aggregated into time-series dashboards. Teams integrate these
        metrics with their existing alerting tools (PagerDuty, Slack, etc.) to
        trigger alerts on quality score drops, cost anomalies, latency spikes,
        or safety incidents. The AI Gateway also supports rate limiting and cost
        budgets as proactive guardrails.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>Learn more</Link>.
      </>
    ),
    answerText:
      "MLflow provides the assessment and scoring infrastructure that feeds alerting systems. Production scorers generate quality metrics on every evaluated trace. These metrics can be queried programmatically via mlflow.search_traces() and aggregated into time-series dashboards. Teams integrate these metrics with their existing alerting tools (PagerDuty, Slack, etc.) to trigger alerts on quality score drops, cost anomalies, latency spikes, or safety incidents. The AI Gateway also supports rate limiting and cost budgets as proactive guardrails.",
  },
  {
    question:
      "Can I use MLflow AI monitoring with any LLM provider or agent framework?",
    answer:
      "Yes. MLflow supports any LLM provider (OpenAI, Anthropic, AWS Bedrock, Google Gemini, Azure OpenAI, Mistral, Cohere, Ollama, and more) and any agent framework (LangChain, LangGraph, LlamaIndex, CrewAI, AutoGen, DSPy, Pydantic AI, and more). One-line auto-instrumentation is available for 50+ libraries. MLflow is fully OpenTelemetry-compatible, so you can export traces to any OTel-compatible backend. SDKs are available for Python, JavaScript, and TypeScript.",
  },
  {
    question: "How does MLflow handle human feedback in AI monitoring?",
    answer: (
      <>
        MLflow integrates human feedback directly into the monitoring pipeline.
        Users can provide ratings (thumbs up/down, star ratings) linked to
        specific traces via{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "assessments/feedback/"}>
          <code>log_feedback()</code>
        </Link>
        . This feedback serves multiple purposes: ground-truthing LLM judges,
        identifying silent failures that automated scoring misses, and improving
        judges over time.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "assessments/feedback/"}>
          MemAlign
        </Link>{" "}
        learns evaluation guidelines from just a handful of natural-language
        feedback examples, enabling judges to align with domain experts without
        expensive retraining.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
          Learn about feedback collection
        </Link>
        .
      </>
    ),
    answerText:
      "MLflow integrates human feedback directly into the monitoring pipeline. Users can provide ratings (thumbs up/down, star ratings) linked to specific traces via log_feedback(). This feedback serves multiple purposes: ground-truthing LLM judges, identifying silent failures that automated scoring misses, and improving judges over time. MemAlign learns evaluation guidelines from just a handful of natural-language feedback examples, enabling judges to align with domain experts without expensive retraining.",
  },
  {
    question:
      "How does MLflow handle security threats like prompt injection and PII leakage?",
    answer: (
      <>
        MLflow's AI monitoring captures complete audit trails of all inputs,
        outputs, and model interactions for compliance and incident
        investigation. Built-in safety scorers detect harmful content, PII
        exposure, and policy violations in outputs. The AI Gateway adds
        real-time guardrails that filter inputs for prompt injection attempts
        and scan outputs for PII, toxicity, or policy violations before they
        reach users. Combined with tracing, you can investigate security
        incidents with full execution context.{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "governance/ai-gateway/"}>
          Learn about AI Gateway guardrails
        </Link>
        .
      </>
    ),
    answerText:
      "MLflow's AI monitoring captures complete audit trails of all inputs, outputs, and model interactions for compliance and incident investigation. Built-in safety scorers detect harmful content, PII exposure, and policy violations in outputs. The AI Gateway adds real-time guardrails that filter inputs for prompt injection attempts and scan outputs for PII, toxicity, or policy violations before they reach users. Combined with tracing, you can investigate security incidents with full execution context.",
  },
  {
    question: "Is MLflow AI monitoring free and open source?",
    answer:
      "Yes. MLflow is 100% open source under the Apache 2.0 license, backed by the Linux Foundation. All AI monitoring features (tracing, evaluation, scorers, cost tracking, feedback collection) are free, including for commercial use. There are no per-trace fees, no usage limits, and no vendor lock-in. You can self-host MLflow or use managed versions on Databricks, AWS, and other platforms. Your production data stays under your control.",
  },
  {
    question: "How do I get started with MLflow AI monitoring?",
    answer: (
      <>
        MLflow AI monitoring combines tracing, automatic LLM judge evaluation,
        human feedback collection, and cost tracking into a unified stack. See
        the <a href="#how-to-implement">How to Implement AI Monitoring</a>{" "}
        section above for a step-by-step walkthrough with code examples, or jump
        straight to the{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/prod-tracing/"}>
          production tracing documentation
        </Link>{" "}
        for detailed setup guides and framework-specific examples.
      </>
    ),
    answerText:
      "MLflow AI monitoring combines tracing, automatic LLM judge evaluation, human feedback collection, and cost tracking into a unified stack. See the How to Implement AI Monitoring section above for a step-by-step walkthrough with code examples, or jump straight to the production tracing documentation for detailed setup guides and framework-specific examples.",
  },
  {
    question:
      "What is the difference between MLflow's development evaluation and AI monitoring?",
    answer:
      "Development evaluation runs scorers and LLM judges against curated benchmark datasets to validate quality before deployment. AI monitoring runs the same scorers and LLM judges on live production traffic to detect issues that emerge from real-world usage. MLflow unifies both: use the same scorers, LLM judges, and evaluation APIs in development and production to ensure consistent quality standards. Development evaluation gives you confidence to deploy; AI monitoring gives you confidence it's still working well.",
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
    "Open-source platform for AI monitoring, LLM monitoring, agent monitoring, evaluation, and deployment.",
  url: "https://mlflow.org",
  license: "https://www.apache.org/licenses/LICENSE-2.0",
};

const CODE_PRODUCTION_TRACING = `import mlflow
import os
from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
@mlflow.trace
def handle_chat(request: Request, chat_request: ChatRequest):
    # Attach production context to every trace
    mlflow.update_current_trace(
        client_request_id=request.headers.get("X-Request-ID"),
        tags={
            "mlflow.trace.session": request.headers.get("X-Session-ID"),
            "mlflow.trace.user": request.headers.get("X-User-ID"),
            "environment": "production",
            "app_version": os.getenv("APP_VERSION", "1.0.0"),
            "deployment_id": os.getenv("DEPLOYMENT_ID", "unknown"),
        },
    )

    response = generate_response(chat_request.message)
    return {"response": response}`;

const CODE_ONLINE_EVAL = `import mlflow
from mlflow.genai.scorers import Guidelines, ScorerSamplingConfig

mlflow.set_experiment("production-genai-app")

# Create a production judge with evaluation criteria
safety_judge = Guidelines(
    name="safety_check",
    guidelines=(
        "The response must not contain PII, harmful content, "
        "or hallucinated information."
    ),
    model="gateway:/my-llm-endpoint",
)

# Register and start automatic evaluation on production traces
registered_judge = safety_judge.register(name="production_safety_check")
registered_judge.start(
    sampling_config=ScorerSamplingConfig(
        sample_rate=0.1,  # Evaluate 10% of traces
        filter_string="metadata.environment = 'production'",
    ),
)`;

const CODE_FEEDBACK = `import mlflow
from mlflow.entities import AssessmentSource
from fastapi import FastAPI

app = FastAPI()

@app.post("/feedback")
def submit_feedback(trace_id: str, is_correct: bool, user_id: str):
    mlflow.log_feedback(
        trace_id=trace_id,
        name="response_is_correct",
        value=is_correct,
        source=AssessmentSource(
            source_type="HUMAN",
            source_id=user_id,
        ),
    )`;

export default function AiMonitoring() {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://mlflow.org/ai-monitoring" />
        <link rel="canonical" href="https://mlflow.org/ai-monitoring" />
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
          <h1>AI Monitoring for LLMs and Agents</h1>

          <p>
            AI monitoring is the practice of continuously evaluating the
            quality, performance, cost, and safety of AI applications running in
            production, including LLM and agent-based systems. It goes beyond
            uptime and error rates to assess the <em>quality</em> of
            non-deterministic AI outputs, track token costs and latency, and
            detect when behavior drifts from expected standards.{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/prod-tracing/"}>
              Production tracing
            </Link>{" "}
            captures the execution data that makes this possible.
          </p>

          <p>
            Unlike classical ML monitoring (which tracks feature distributions
            and prediction accuracy on structured data), AI monitoring must
            evaluate free-form language outputs, multi-step agent reasoning,
            tool call chains, retrieval accuracy, and token costs. Traditional
            monitoring can tell you the system is running; AI monitoring tells
            you whether it's <em>working well</em>.
          </p>

          <p>
            <Link href="/genai">MLflow</Link> provides a complete AI monitoring
            stack: automatic online evaluation with LLM judges that score traces
            asynchronously, configurable trace sampling for cost control, user
            and session context tracking for debugging, human feedback
            collection, and built-in scorers for hallucination detection,
            safety, and more.{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/"}>
              Explore the evaluation and monitoring docs
            </Link>
            .
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
                src="https://mlflow.org/docs/latest/images/llms/tracing/overview_demo.mp4"
                type="video/mp4"
              />
              Your browser does not support the video tag.
            </video>
          </div>

          <h2 id="why-monitoring">Why AI Monitoring Matters</h2>

          <p>
            Agents and LLM applications in production face challenges that don't
            exist during development:
          </p>

          <div className="grid-2">
            <div className="card">
              <h3>Quality Drift Detection</h3>
              <p>
                <strong>Problem:</strong> Agent outputs degrade silently from
                model updates, prompt changes, or shifting user inputs.
              </p>
              <p>
                <strong>Solution:</strong> Continuous LLM judges and human
                feedback detect quality regressions before users lose trust.
              </p>
            </div>

            <div className="card">
              <h3>Cost and Latency Control</h3>
              <p>
                <strong>Problem:</strong> Token costs and latency can spiral
                without visibility into per-request spending and response times.
              </p>
              <p>
                <strong>Solution:</strong> Automatic cost/token tracking with
                per-model breakdowns and anomaly detection.
              </p>
            </div>

            <div className="card">
              <h3>Safety and Security</h3>
              <p>
                <strong>Problem:</strong> Production agents face prompt
                injection, PII leakage, jailbreaks, and policy violations that
                don't exist in development.
              </p>
              <p>
                <strong>Solution:</strong> Real-time safety scoring with
                deterministic and LLM-based detectors on every request.
              </p>
            </div>

            <div className="card">
              <h3>Production Debugging</h3>
              <p>
                <strong>Problem:</strong> When quality drops or errors spike,
                tracing the root cause across multi-step agent workflows is
                complex.
              </p>
              <p>
                <strong>Solution:</strong> Full execution traces with assessment
                scores enable rapid root-cause analysis.
              </p>
            </div>
          </div>

          <h2 id="use-cases">AI Monitoring Use Cases</h2>

          <ul>
            <li>
              <strong>Hallucination detection in RAG systems:</strong> Run
              groundedness scorers on production traces to catch when retrieval
              quality degrades or the model starts generating claims unsupported
              by the retrieved context.
            </li>
            <li>
              <strong>Agent tool selection monitoring:</strong> Track whether
              agents pick the right tools and complete tasks efficiently. Detect
              loops, unnecessary retries, and incorrect tool selections that
              waste tokens and degrade user experience.
            </li>
            <li>
              <strong>Cost optimization:</strong> Identify expensive queries,
              track per-model spend trends, and find opportunities to switch to
              cheaper models for low-complexity requests without sacrificing
              quality.
            </li>
            <li>
              <strong>Safety regression detection:</strong> After model or
              prompt updates, compare safety scores against pre-deployment
              baselines to catch regressions before they affect users at scale.
            </li>
            <li>
              <strong>A/B testing prompt changes:</strong> Compare quality
              scores, latency, and cost across prompt variants using production
              trace data to make data-driven decisions about which version to
              keep.
            </li>
            <li>
              <strong>Compliance and audit in regulated industries:</strong>{" "}
              Healthcare, finance, and legal teams need to prove their AI
              systems behave correctly and safely. AI monitoring provides full
              audit trails of every input, output, and model interaction for
              regulatory review.
            </li>
            <li>
              <strong>Latency SLA monitoring:</strong> For user-facing chatbots,
              coding assistants, and real-time agents where response time
              directly impacts user experience. Track p50/p95/p99 latency and
              time-to-first-token to catch performance regressions before they
              affect retention.
            </li>
          </ul>

          <h2 id="how-to-implement">How to Implement AI Monitoring</h2>

          <p>
            <Link href="/genai">MLflow</Link> provides an open-source AI
            monitoring stack that covers tracing, automatic quality evaluation
            with LLM judges, cost and token tracking, human feedback collection,
            and real-time safety guardrails, compatible with any LLM provider
            and any agent framework. Here's how to set it up.
          </p>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              gap: "20px",
              margin: "32px -60px",
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
                Trace Every Request
              </div>
              <div
                style={{
                  fontSize: "14px",
                  color: "#505050",
                  lineHeight: "1.5",
                }}
              >
                Add production tracing with{" "}
                <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
                  <code>@mlflow.trace</code>
                </Link>{" "}
                to capture execution graphs. Attach user, session, and
                deployment context.
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
                Score Traces Automatically
              </div>
              <div
                style={{
                  fontSize: "14px",
                  color: "#505050",
                  lineHeight: "1.5",
                }}
              >
                Set up{" "}
                <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/"}>
                  automatic LLM judge evaluation
                </Link>{" "}
                to score production traces for safety, correctness, and quality
                drift in the background.
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
                Collect Human Feedback
              </div>
              <div
                style={{
                  fontSize: "14px",
                  color: "#505050",
                  lineHeight: "1.5",
                }}
              >
                Use{" "}
                <Link href={MLFLOW_GENAI_DOCS_URL + "assessments/feedback/"}>
                  <code>mlflow.log_feedback()</code>
                </Link>{" "}
                to record user ratings linked to traces. Catch quality issues
                that automated judges miss and calibrate scoring over time.
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
                Track Costs &<br />
                Enforce Guardrails
              </div>
              <div
                style={{
                  fontSize: "14px",
                  color: "#505050",
                  lineHeight: "1.5",
                }}
              >
                Integrate automatic{" "}
                <Link
                  href={MLFLOW_GENAI_DOCS_URL + "tracing/token-usage-cost/"}
                >
                  token and cost tracking
                </Link>{" "}
                per request. The{" "}
                <Link href={MLFLOW_GENAI_DOCS_URL + "governance/ai-gateway/"}>
                  AI Gateway
                </Link>{" "}
                adds real-time safety guardrails.
              </div>
            </div>
          </div>

          <p style={{ marginTop: "32px", marginBottom: "0px" }}>
            <strong>Trace production requests with context</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton code={CODE_PRODUCTION_TRACING} />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={CODE_PRODUCTION_TRACING}
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
            <strong>Register judges for automatic online evaluation</strong>
          </p>

          <div
            style={{
              margin: "8px 0",
              borderRadius: "8px",
              overflow: "hidden",
              border: "1px solid #e5e7eb",
            }}
          >
            <video width="100%" controls autoPlay loop muted playsInline>
              <source
                src="https://mlflow.org/docs/latest/images/llms/tracing/automatic-evaluation-ui-setup.mp4"
                type="video/mp4"
              />
              Your browser does not support the video tag.
            </video>
          </div>

          <p style={{ marginTop: "32px", marginBottom: "0px" }}>
            <strong>Collect user feedback on traces</strong>
          </p>

          <div
            className="rounded-lg border border-white/10 overflow-hidden"
            style={{ backgroundColor: CODE_BG, margin: "8px 0" }}
          >
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
              <span className="text-xs text-white/50 font-mono">python</span>
              <CopyButton code={CODE_FEEDBACK} />
            </div>
            <div className="p-3 overflow-x-auto">
              <Highlight
                theme={customNightOwl}
                code={CODE_FEEDBACK}
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
              <strong>AI engineering platform</strong>, with over 30 million
              monthly downloads. Thousands of organizations use MLflow to debug,
              evaluate, monitor, and optimize production-quality AI agents and
              LLM applications while controlling costs and managing access to
              models and data. Backed by the Linux Foundation and licensed under
              Apache 2.0, MLflow provides a complete AI monitoring solution with
              no vendor lock-in.{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL}>Get started →</Link>
            </p>
          </div>

          <h2>Open Source vs. Proprietary AI Monitoring Tools</h2>

          <p>
            When choosing an AI monitoring platform for agents and LLM
            applications, the decision between open source and proprietary SaaS
            tools has significant long-term implications for your team,
            infrastructure, and data ownership.
          </p>

          <p>
            <strong>
              Open Source (<Link href="/genai">MLflow</Link>):
            </strong>{" "}
            With MLflow, you maintain complete control over your production
            traces and monitoring data. Deploy on your own infrastructure or use
            managed versions on Databricks, AWS, or other platforms. There are
            no per-trace fees, no usage limits, and no vendor lock-in. Your
            production data stays under your control, and OpenTelemetry
            compatibility ensures you can export traces to any backend.
          </p>

          <p>
            <strong>Proprietary SaaS Tools:</strong> Commercial monitoring
            platforms offer convenience but at the cost of flexibility and
            control. They typically charge per trace or per seat, which can
            become expensive at scale. Your production data is sent to their
            servers, raising privacy and compliance concerns for sensitive
            traces. You're locked into their ecosystem, making it difficult to
            switch providers or customize functionality.
          </p>

          <p>
            <strong>Why Teams Choose Open Source:</strong> Organizations running
            production agents increasingly choose MLflow because it offers
            enterprise-grade monitoring without compromising on data
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
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/prod-tracing/"}>
                Production Tracing Documentation
              </Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/lightweight-sdk"}>
                Lightweight SDK Documentation
              </Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/"}>
                Scorers Documentation
              </Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/token-usage-cost/"}>
                Token Usage and Cost Tracking
              </Link>
            </li>
            <li>
              <Link href="/llm-tracing">LLM Tracing FAQ</Link>
            </li>
            <li>
              <Link href="/llm-evaluation">Agent Evaluation FAQ</Link>
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
