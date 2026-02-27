import {
  Layout,
  AboveTheFold,
  BelowTheFold,
  HeroImage,
  StickyGrid,
} from "../../components";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import CardHero from "@site/static/img/blog/mlflow-ai-gateway-thumbnail.png";
import Card1 from "@site/static/img/GenAI_gateway/GenAI_gateway_1.png";
import Card2 from "@site/static/img/GenAI_gateway/GenAI_gateway_2.png";
import Card3 from "@site/static/img/GenAI_gateway/GenAI_gateway_3.png";

export default function AiGateway() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="AI gateway"
        title="Unified access to all AI models"
        body="Standardize how you interact with different LLM providers using one central interface."
        hasGetStartedButton={`${MLFLOW_GENAI_DOCS_URL}governance/ai-gateway/setup/`}
      >
        <HeroImage src={CardHero} alt="" />
      </AboveTheFold>

      <StickyGrid
        cards={[
          {
            title: "Access 50+ Model Providers",
            body: "Define and manage multiple LLM endpoints across providers in a single place, enabling centralized API key management and seamless integration.",
            image: <img src={Card1} alt="Configure endpoints" />,
          },
          {
            title: "Traffic routing and fallbacks",
            body: "Split traffic across multiple models for A/B testing and gradual rollouts. Define fallback chains so requests automatically reroute to a backup provider when the primary is unavailable.",
            image: <img src={Card2} alt="Traffic routing and fallbacks" />,
          },
          {
            title: "Usage tracking",
            body: "Every request is recorded as an MLflow trace. Visualize request volume, latency percentiles, token consumption, and cost breakdowns across all endpoints from a unified dashboard.",
            image: <img src={Card3} alt="Usage tracking dashboard" />,
          },
        ]}
      />

      <BelowTheFold contentType="genai" />
    </Layout>
  );
}
