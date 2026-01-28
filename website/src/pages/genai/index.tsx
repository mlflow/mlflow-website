import {
  Layout,
  LogosCarousel,
  LatestNews,
  AboveTheFold,
  BelowTheFold,
  ValuePropWidget,
  StickyGrid,
} from "../../components";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import Card1 from "@site/static/img/GenAI_home/GenAI_home_1.png";
import Card2 from "@site/static/img/GenAI_home/GenAI_home_2.png";
import Card3 from "@site/static/img/GenAI_home/GenAI_home_3.png";
import CardGateway from "@site/static/img/GenAI_gateway/GenAI_gateway_1.png";

export default function GenAi(): JSX.Element {
  return (
    <Layout>
      <AboveTheFold
        title="Ship high-quality GenAI, fast"
        body={[
          "Enhance your GenAI application with end-to-end tracking, observability, evaluations, all in one integrated platform.",
        ]}
        hasGetStartedButton={MLFLOW_GENAI_DOCS_URL}
        bodyColor="white"
      />

      <StickyGrid
        cards={[
          {
            title: "Debug with tracing",
            body: [
              "Debug and iterate on GenAI applications using MLflow's tracing, which captures your app's entire execution, including prompts, retrievals, tool calls.",
              "MLflow's open-source, OpenTelemetry-compatible tracing SDK helps avoid vendor lock-in.",
            ],
            cta: {
              href: "/genai/observability",
              text: "Learn more",
            },
            image: <img src={Card1} alt="MLflow tracing" />,
          },

          {
            title: "Accurately measure free-form language with LLM judges",
            body: "Utilize LLM-as-a-judge metrics, mimicking human expertise, to assess and enhance GenAI quality. Access pre-built judges for common metrics like hallucination or relevance, or develop custom judges tailored to your business needs and expert insights.",
            cta: {
              href: "/genai/evaluations",
              text: "Learn more",
            },
            image: <img src={Card2} alt="MLflow LLM judges" />,
          },
          {
            title: "Prompt Registry",
            body: [
              "Version, compare, iterate on, and discover prompt templates directly through the MLflow UI. Reuse prompts across multiple versions of your agent or application code, and view rich lineage identifying which versions are using each prompt.",
            ],
            cta: {
              href: "/genai/prompt-registry",
              text: "Learn more",
            },
            image: <img src={Card3} alt="MLflow LLM judges" />,
          },

          {
            title: "AI Gateway",
            body: [
              "Standardize access to multiple LLM providers with unified endpoints, centralized key management, and rate limiting.",
            ],
            cta: {
              href: "/genai/ai-gateway",
              text: "Learn more",
            },
            image: <img src={CardGateway} alt="MLflow AI Gateway" />,
          },
        ]}
      />

      <LogosCarousel />
      <ValuePropWidget />
      <BelowTheFold contentType="genai">
        <LatestNews />
      </BelowTheFold>
    </Layout>
  );
}
