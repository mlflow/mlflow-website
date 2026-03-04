import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import {
  Layout,
  AboveTheFold,
  BelowTheFold,
  Section,
  HeroImage,
  Button,
  HighlightedKeyword,
  ProcessSection,
  EcosystemList,
} from "../../components";
import { StickyFeaturesGrid } from "../../components/ProductTabs/StickyFeaturesGrid";
import type { Feature } from "../../components/ProductTabs/features";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import { genaiIntegrations } from "./index";
import CardHero from "@site/static/img/GenAI_observability/GenAI_observability_hero.png";
import Card1 from "@site/static/img/GenAI_observability/GenAI_observability_1.png";
import Card2 from "@site/static/img/GenAI_observability/GenAI_observability_2.png";
import Card3 from "@site/static/img/GenAI_observability/GenAI_observability_3.png";
import Card4 from "@site/static/img/GenAI_observability/GenAI_observability_token_usage.png";

const tracingFeatures: Feature[] = [
  {
    id: "end-to-end-observability",
    title: "End to end observability",
    description:
      "Capture your agent or LLM application's inputs, outputs, and step-by-step execution: prompts, retrievals, tool calls, and more.",
    imageSrc: Card1,
    quickstartLink: `${MLFLOW_GENAI_DOCS_URL}tracing/quickstart/`,
    codeSnippet: `import mlflow
import openai

mlflow.openai.autolog()

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
# View full trace in MLflow UI`,
  },
  {
    id: "visualize-execution-flow",
    title: "Visualize execution flow",
    description:
      "Deep dive into your agent or LLM application's logic and latency with a comprehensive and intuitive UI for effective debugging.",
    imageSrc: Card3,
    codeSnippet: `import mlflow

mlflow.openai.autolog()

# Complex multi-step execution is fully captured
response = agent.run("Summarize sales data")

# Open MLflow UI → Traces tab
# See execution DAG, timeline, and I/O values`,
  },
  {
    id: "quality-monitoring",
    title: "Quality monitoring",
    description:
      "Track and analyze the quality of your agent or LLM application over time, and take action to fix issues before impact spreads.",
    imageSrc: Card2,
    codeSnippet: `import mlflow

mlflow.openai.autolog()

# Find traces where users were frustrated
traces = mlflow.search_traces(
    filter_string='feedback.user_frustration = "true"',
)
for t in traces:
    print(f"Request: {t.data.request}")`,
  },
  {
    id: "understand-many-traces",
    title: "Spot trends and patterns at scale",
    description:
      "Zoom out with a simplified summary UI to quickly review many traces at once to understand how your agent or LLM application is performing overall.",
    imageSrc: Card4,
    imageFit: "contain",
    imageZoom: 100,
    codeSnippet: `import mlflow

# Search traces and analyze token usage
traces = mlflow.search_traces(
    order_by=["timestamp DESC"],
    max_results=100,
)
total = sum(
    t.info.token_usage["total_tokens"]
    for t in traces
)
print(f"Total tokens: {total}")`,
  },
];

const SEO_TITLE = "LLM & Agent Observability | MLflow AI Platform";
const SEO_DESCRIPTION =
  "Debug and monitor AI agents and LLM apps with MLflow's AI Engineering Platform. Full execution traces, quality monitoring, and trend analysis. Auto-trace 50+ frameworks.";

export default function Observability() {
  return (
    <Layout>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://mlflow.org/genai/observability" />
        <link rel="canonical" href="https://mlflow.org/genai/observability" />
      </Head>
      <AboveTheFold
        sectionLabel="Observability"
        title="LLM and Agent Observability"
        minHeight="small"
        body={[
          <>
            Gain visibility into your agent or LLM application's logic to{" "}
            <HighlightedKeyword>debug issues</HighlightedKeyword>,{" "}
            <HighlightedKeyword>improve quality</HighlightedKeyword> and{" "}
            <HighlightedKeyword>understand user behavior</HighlightedKeyword>.
          </>,
        ]}
        bodyColor="white"
        actions={
          <div className="flex flex-wrap justify-center items-center gap-4">
            <Link to={`${MLFLOW_GENAI_DOCS_URL}tracing/quickstart/`}>
              <Button variant="primary" size="medium">
                Get Started
              </Button>
            </Link>
            <Link to={`${MLFLOW_GENAI_DOCS_URL}tracing/`}>
              <Button variant="outline" size="medium">
                View Docs
              </Button>
            </Link>
          </div>
        }
      >
        <div className="relative w-full max-w-[1320px] mx-auto px-4">
          <video
            className="rounded-[16px]"
            width="100%"
            autoPlay
            loop
            muted
            playsInline
          >
            <source
              src={
                require("@site/static/img/releases/3.9.0/overview_demo.mp4")
                  .default
              }
              type="video/mp4"
            />
          </video>
        </div>
      </AboveTheFold>

      <Section title="Rich, detailed traces for every request">
        <StickyFeaturesGrid features={tracingFeatures} colorTheme="red" />
      </Section>

      <EcosystemList
        title="Automatic tracing for your entire stack"
        body={
          <>
            Auto-trace 50+{" "}
            <Link
              to="/llmops"
              className="text-blue-400 !underline decoration-blue-400/50 underline-offset-2 hover:text-blue-300 hover:decoration-blue-300/50 transition-colors"
            >
              LLM
            </Link>{" "}
            providers and agent frameworks with a single line of code.{" "}
            <Link
              to="/llm-tracing"
              className="text-blue-400 !underline decoration-blue-400/50 underline-offset-2 hover:text-blue-300 hover:decoration-blue-300/50 transition-colors"
            >
              LLM tracing
            </Link>{" "}
            captures every execution step, and MLflow is{" "}
            <a
              href="https://mlflow.org/docs/latest/genai/tracing/opentelemetry/"
              className="text-blue-400 !underline decoration-blue-400/50 underline-offset-2 hover:text-blue-300 hover:decoration-blue-300/50 transition-colors"
            >
              OpenTelemetry compatible
            </a>
            , supporting any programming language, agent, or LLM.
          </>
        }
        items={genaiIntegrations.map((item) => ({
          ...item,
          src: item.src.startsWith("/") ? item.src : `/${item.src}`,
        }))}
      />

      <ProcessSection
        subtitle="From zero to full observability in minutes. No complex setup or major code changes required."
        colorTheme="red"
      />
      <BelowTheFold contentType="genai" hideGetStarted hideSocialWidget />
    </Layout>
  );
}
