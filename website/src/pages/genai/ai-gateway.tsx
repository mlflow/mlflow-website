import {
  Layout,
  AboveTheFold,
  BelowTheFold,
  HeroImage,
  StickyGrid,
} from "../../components";
import CardHero from "@site/static/img/GenAI_gateway/GenAI_gateway_hero.png";
import Card1 from "@site/static/img/GenAI_gateway/GenAI_gateway_1.png";
import Card2 from "@site/static/img/GenAI_gateway/GenAI_gateway_2.png";

export default function AiGateway() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="AI gateway"
        title="Unified access to all AI models"
        body="Standardize how you interact with different LLM providers using one central interface."
        hasGetStartedButton="https://mlflow.org/docs/latest/genai/governance/ai-gateway/setup/"
      >
        <HeroImage src={CardHero} alt="" />
      </AboveTheFold>

      <StickyGrid
        cards={[
          {
            title: "Configure endpoints",
            body: "Define and manage multiple LLM endpoints across providers in a single YAML file, enabling centralized API key management and seamless integration.",
            image: <img src={Card1} alt="Improved Model Accuracy" />,
          },
          {
            title: "Rate limiting",
            body: "Rate limits can be set per endpoint, such as 10 calls per minute, by specifying the limit and time period in the configuration.",
            image: <img src={Card2} alt="Spending Oversight" />,
          },
        ]}
      />

      <BelowTheFold contentType="genai" />
    </Layout>
  );
}
