import {
  Layout,
  AboveTheFold,
  BelowTheFold,
  HeroImage,
  StickyGrid,
  EcosystemList,
  ProcessSection,
} from "../../components";
import Link from "@docusaurus/Link";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import type { EcosystemItem } from "../../components/EcosystemList/EcosystemList";
import CardHero from "@site/static/img/GenAI_gateway/GenAI_gateway_hero.png";
import Card1 from "@site/static/img/GenAI_gateway/GenAI_gateway_1.png";
import Card2 from "@site/static/img/GenAI_gateway/GenAI_gateway_2.png";
import Card3 from "@site/static/img/GenAI_gateway/GenAI_gateway_3.png";

const gatewayProviders: EcosystemItem[] = [
  {
    title: "OpenAI",
    src: "/img/openai.svg",
    href: `${MLFLOW_GENAI_DOCS_URL}governance/ai-gateway/setup/`,
  },
  {
    title: "Anthropic",
    src: "/img/anthropic.svg",
    href: `${MLFLOW_GENAI_DOCS_URL}governance/ai-gateway/setup/`,
  },
  {
    title: "Gemini",
    src: "/img/google-gemini.svg",
    href: `${MLFLOW_GENAI_DOCS_URL}governance/ai-gateway/setup/`,
  },
  {
    title: "Amazon Bedrock",
    src: "/img/bedrock.png",
    href: `${MLFLOW_GENAI_DOCS_URL}governance/ai-gateway/setup/`,
  },
  {
    title: "Databricks",
    src: "/img/databricks.svg",
    href: `${MLFLOW_GENAI_DOCS_URL}governance/ai-gateway/setup/`,
  },
  {
    title: "Mistral",
    src: "/img/mistral.png",
    href: `${MLFLOW_GENAI_DOCS_URL}governance/ai-gateway/setup/`,
  },
  {
    title: "Cohere",
    src: "/img/cohere-logo.png",
    href: `${MLFLOW_GENAI_DOCS_URL}governance/ai-gateway/setup/`,
  },
  {
    title: "DeepSeek",
    src: "/img/deepseek.png",
    href: `${MLFLOW_GENAI_DOCS_URL}governance/ai-gateway/setup/`,
  },
  {
    title: "Groq",
    src: "/img/groq.svg",
    href: `${MLFLOW_GENAI_DOCS_URL}governance/ai-gateway/setup/`,
  },
  {
    title: "Together AI",
    src: "/img/together-ai-logo.png",
    href: `${MLFLOW_GENAI_DOCS_URL}governance/ai-gateway/setup/`,
  },
  {
    title: "Fireworks AI",
    src: "/img/fireworks-ai-logo.png",
    href: `${MLFLOW_GENAI_DOCS_URL}governance/ai-gateway/setup/`,
  },
  {
    title: "Ollama",
    src: "/img/ollama.png",
    href: `${MLFLOW_GENAI_DOCS_URL}governance/ai-gateway/setup/`,
  },
];

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

      <EcosystemList
        title="Supported providers"
        body={
          <>
            Route requests to any major LLM provider through a single, unified
            interface. The gateway handles credentials, usage tracking, and
            failover so your application code stays provider-agnostic. Learn
            more in our{" "}
            <Link
              href="/faq/ai-gateway"
              style={{
                color: "inherit",
                textDecoration: "underline",
                opacity: 0.9,
              }}
            >
              AI Gateway FAQ
            </Link>
            .
          </>
        }
        items={gatewayProviders}
      />

      <ProcessSection
        subtitle="Set up governed LLM access in minutes. No additional infrastructure required."
        getStartedLink={`${MLFLOW_GENAI_DOCS_URL}governance/ai-gateway/quickstart/`}
        steps={[
          {
            number: "1",
            title: "Start MLflow Server",
            description:
              "Launch the tracking server. The AI Gateway is included out of the box.",
            time: "~30 seconds",
            code: `pip install 'mlflow[genai]'\nuvx mlflow server`,
            language: "bash",
          },
          {
            number: "2",
            title: "Create an Endpoint",
            description:
              "Add API keys and configure endpoints from the UI, no server restart needed.",
            time: "~1 minute",
            code: `# Open the Gateway UI\n# http://localhost:5000/#/gateway\n#\n# 1. Create an API key with your\n#    provider credentials\n# 2. Create an endpoint and link\n#    it to your API key`,
            language: "bash",
          },
          {
            number: "3",
            title: "Query Through the Gateway",
            description:
              "Use any OpenAI-compatible SDK. Point the base URL at the gateway and use your endpoint name as the model.",
            time: "~30 seconds",
            code: `from openai import OpenAI\n\nclient = OpenAI(\n    base_url="http://localhost:5000"\n           "/gateway/mlflow/v1",\n    api_key="unused",\n)\nresponse = client.chat.completions.create(\n    model="my-endpoint",\n    messages=[{"role": "user",\n               "content": "Hello!"}],\n)`,
            language: "python",
          },
        ]}
      />

      <BelowTheFold contentType="genai" hideGetStarted />
    </Layout>
  );
}
