import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import {
  Layout,
  AboveTheFold,
  BelowTheFold,
  Section,
  HeroImage,
  ProcessSection,
} from "../../components";
import { StickyFeaturesGrid } from "../../components/ProductTabs/StickyFeaturesGrid";
import type { Feature } from "../../components/ProductTabs/features";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import CardHero from "@site/static/img/GenAI_evaluations/GenAI_evaluations_hero.png";
import Card1 from "@site/static/img/GenAI_evaluations/GenAI_evaluations_1.png";
import Card2 from "@site/static/img/GenAI_evaluations/GenAI_evaluations_2.png";
import Card3 from "@site/static/img/GenAI_evaluations/GenAI_evaluations_3.png";
import Card4 from "@site/static/img/GenAI_evaluations/GenAI_evaluations_4.png";
import Card5 from "@site/static/img/GenAI_evaluations/GenAI_evaluations_5.png";
import Card6 from "@site/static/img/GenAI_evaluations/GenAI_evaluations_6.png";
import CardFeedback from "@site/static/img/GenAI_evaluations/GenAI_evaluations_feedback.png";
import CardCustomJudge from "@site/static/img/GenAI_evaluations/GenAI_evaluations_custom_judge.png";

const qualityFeatures: Feature[] = [
  {
    id: "prebuilt-judges",
    title: "Pre-built LLM judges",
    description:
      "Quickly start with built-in LLM judges for safety, hallucination, retrieval quality, and relevance. Our research-backed judges provide accurate, reliable quality evaluation aligned with human expertise.",
    imageSrc: Card1,
    codeSnippet: `import mlflow
from mlflow.genai.scorers import (
    Safety,
    Correctness,
    RelevanceToQuery,
)

results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[Safety(), Correctness(), RelevanceToQuery()],
)`,
  },
  {
    id: "custom-judges",
    title: "Customized LLM judges",
    description:
      "Adapt our base model to create custom LLM judges tailored to your business needs, aligning with your human expert's judgment.",
    imageSrc: CardCustomJudge,
    imageZoom: 98,
    imageFit: "contain",
    codeSnippet: `from mlflow.genai.judges import make_judge
from typing import Literal

judge = make_judge(
    name="conversation_quality",
    instructions=(
        "Analyze the {{ conversation }} "
        "for signs of user frustration, "
        "unresolved questions, or "
        "incorrect answers."
    ),
    feedback_value_type=Literal[
        "high_quality", "low_quality"
    ],
)`,
  },
  {
    id: "human-feedback",
    title: "Collect human feedback",
    description:
      "Gather feedback from end users and domain experts directly within your application. Use human annotations to validate LLM judge accuracy, identify blind spots, and continuously improve evaluation quality.",
    imageSrc: CardFeedback,
    imageZoom: 95,
    codeSnippet: `import mlflow

# Log human feedback on a trace
mlflow.log_feedback(
    trace_id=trace.info.trace_id,
    name="user_rating",
    value="positive",
    source=mlflow.feedback.HUMAN,
)`,
  },
];

const iterativeFeatures: Feature[] = [
  {
    id: "test-variants",
    title: "Test new agent versions",
    description:
      "MLflow's GenAI evaluation API lets you test new agent versions (prompts, models, code) against evaluation and regression datasets. Each version is linked to its evaluation results, enabling tracking of improvements over time.",
    imageSrc: Card3,
    codeSnippet: `import mlflow

# Evaluate your app against a dataset
results = mlflow.genai.evaluate(
    data="my_eval_dataset",
    predict_fn=my_agent,
    scorers=[Correctness(), RelevanceToQuery()],
)
print(f"Pass rate: {results.metrics['correctness/pass_rate']}")`,
  },
  {
    id: "code-based-metrics",
    title: "Customize with code-based metrics",
    description:
      "Customize evaluation to measure any aspect of your app's quality or performance using our custom metrics API. Convert any Python function—from regex to custom logic—into a metric.",
    codeSnippet: `from mlflow.genai.scorers import scorer

@scorer
def response_length(request, response):
    """Check response is within length limits."""
    length = len(response.text.split())
    return length <= 500

results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[response_length],
)`,
  },
  {
    id: "evaluation-ui",
    title: "Identify root causes with evaluation review UIs",
    description:
      "Use MLflow's Evaluation UI to visualize a summary of your evals and view results record-by-record to quickly identify root causes and further improvement opportunities.",
    imageSrc: Card5,
    codeSnippet: `import mlflow

# Run evaluation and review in the UI
results = mlflow.genai.evaluate(
    data="regression_dataset",
    predict_fn=my_agent,
    scorers=[Correctness(), Safety()],
)
# Open MLflow UI → Evaluation tab
# Filter by failed assessments to find root causes`,
  },
  {
    id: "compare-versions",
    title: "Compare versions side-by-side",
    description:
      "Compare evaluations across agent versions to understand if your changes improved or regressed quality. Review individual questions side-by-side in the Trace Comparison UI to find differences, debug regressions, and inform your next version.",
    imageSrc: Card6,
    codeSnippet: `import mlflow

# Evaluate two versions of your agent
for version in [agent_v1, agent_v2]:
    results = mlflow.genai.evaluate(
        data="regression_dataset",
        predict_fn=version,
        scorers=[Correctness()],
    )
# Compare results in MLflow UI → Compare tab`,
  },
];

const SEO_TITLE = "Agent & LLM Evaluation | MLflow AI Platform";
const SEO_DESCRIPTION =
  "Evaluate AI agent and LLM quality with MLflow's AI Engineering Platform. LLM judges, custom metrics, human feedback, and side-by-side version comparison.";

export default function Evaluations() {
  return (
    <Layout>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta
          property="og:url"
          content="https://mlflow.org/genai/evaluations"
        />
        <link rel="canonical" href="https://mlflow.org/genai/evaluations" />
      </Head>
      <AboveTheFold
        sectionLabel="Evaluations"
        title="Agent Evaluation"
        body={
          <>
            Confidently{" "}
            <Link
              to="/llm-evaluation"
              className="text-blue-400 !underline decoration-blue-400/50 underline-offset-2 hover:text-blue-300 hover:decoration-blue-300/50 transition-colors"
            >
              evaluate
            </Link>{" "}
            quality in development and production to identify issues and
            iteratively test improvements.
          </>
        }
        hasGetStartedButton={`${MLFLOW_GENAI_DOCS_URL}eval-monitor/quickstart/`}
        minHeight="none"
      >
        <HeroImage src={CardHero} alt="" />
      </AboveTheFold>

      <Section
        id="quality-metrics"
        title="Find quality issues using LLM judges and human feedback"
      >
        <StickyFeaturesGrid features={qualityFeatures} colorTheme="red" />
      </Section>

      <Section title="Iteratively improve quality">
        <StickyFeaturesGrid features={iterativeFeatures} colorTheme="red" />
      </Section>

      <ProcessSection
        colorTheme="red"
        subtitle="From zero to evaluating your agent in minutes. No complex setup required."
        getStartedLink={`${MLFLOW_GENAI_DOCS_URL}eval-monitor/quickstart/`}
        steps={[
          {
            number: "1",
            title: "Start MLflow Server",
            description:
              "One command to get started. Docker setup is also available.",
            time: "~30 seconds",
            code: "uvx mlflow server",
            language: "bash",
          },
          {
            number: "2",
            title: "Enable Tracing",
            description:
              "Add minimal code to start capturing traces from your agent or LLM app.",
            time: "~30 seconds",
            code: `import mlflow

mlflow.set_tracking_uri(
    "http://localhost:5000"
)
mlflow.openai.autolog()`,
            language: "python",
          },
          {
            number: "3",
            title: "Run your code",
            description:
              "Run your code as usual. Explore traces and metrics in the MLflow UI.",
            time: "~1 minute",
            code: `from openai import OpenAI

client = OpenAI()
client.responses.create(
    model="gpt-5-mini",
    input="Hello!",
)`,
            language: "python",
          },
          {
            number: "4",
            title: "Evaluate with LLM Judges",
            description:
              "Run built-in LLM judges to automatically score your app's quality.",
            time: "~1 minute",
            code: `from mlflow.genai.scorers import (
    Safety,
    Correctness,
)

traces = mlflow.search_traces()
mlflow.genai.evaluate(
    data=traces,
    scorers=[
        Safety(),
        Correctness(),
    ],
)`,
            language: "python",
          },
        ]}
      />
      <BelowTheFold contentType="genai" hideGetStarted />
    </Layout>
  );
}
