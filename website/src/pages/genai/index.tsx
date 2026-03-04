import Link from "@docusaurus/Link";
import Head from "@docusaurus/Head";
import {
  Layout,
  LogosCarousel,
  LatestNews,
  AboveTheFold,
  BelowTheFold,
  EcosystemList,
  BenefitsSection,
  Button,
  ProcessSection,
  StatsBand,
  HighlightedKeyword,
} from "../../components";
import { StickyFeaturesGrid } from "../../components/ProductTabs/StickyFeaturesGrid";
import { categories } from "../../components/ProductTabs/features";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import { TrustPills } from "../../components/TrustPills/TrustPills";
import type { EcosystemItem } from "../../components/EcosystemList/EcosystemList";

const SEO_TITLE =
  "Open Source Agent Engineering Platform | MLflow AI Platform";
const SEO_DESCRIPTION =
  "Ship AI agents and LLM apps to production with MLflow's AI Engineering Platform. Built-in observability, evaluation, prompt management, and monitoring. 100+ integrations.";

export const genaiIntegrations: EcosystemItem[] = [
  // --- Major LLM Providers ---
  {
    title: "OpenAI",
    src: "img/openai.svg",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/openai.html",
  },
  {
    title: "Anthropic",
    src: "img/anthropic.svg",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/anthropic.html",
  },
  {
    title: "Gemini",
    src: "img/google-gemini.svg",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/gemini.html",
  },
  {
    title: "Amazon Bedrock",
    src: "img/bedrock.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/bedrock.html",
  },
  {
    title: "Databricks",
    src: "img/databricks.svg",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/databricks.html",
  },
  {
    title: "Qwen",
    src: "img/qwen-logo.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/qwen.html",
  },
  {
    title: "Kimi",
    src: "img/kimi-logo.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/litellm.html",
  },
  {
    title: "GLM",
    src: "img/glm-logo.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/litellm.html",
  },
  {
    title: "DeepSeek",
    src: "img/deepseek.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/deepseek.html",
  },
  {
    title: "Mistral",
    src: "img/mistral.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/mistral.html",
  },
  {
    title: "Cohere",
    src: "img/cohere-logo.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/cohere.html",
  },
  {
    title: "Groq",
    src: "img/groq.svg",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/groq.html",
  },
  {
    title: "Ollama",
    src: "img/ollama.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/ollama.html",
  },
  {
    title: "Together AI",
    src: "img/together-ai-logo.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/togetherai.html",
  },
  {
    title: "Fireworks AI",
    src: "img/fireworks-ai-logo.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/fireworksai.html",
  },
  {
    title: "HuggingFace",
    src: "img/huggingface.svg",
    href: "https://mlflow.org/docs/latest/ml/deep-learning/transformers/index.html",
  },
  {
    title: "Claude Code",
    src: "img/claude.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/claude_code.html",
  },
  {
    title: "LiteLLM",
    src: "img/litellm.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/litellm.html",
  },
  // --- Major Agent Frameworks (Python) ---
  {
    title: "LangChain / LangGraph",
    src: "img/langchain.svg",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langchain.html",
  },
  {
    title: "OpenAI Agents",
    src: "img/openai-logo-only.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/openai-agent.html",
  },
  {
    title: "Vercel AI",
    src: "img/vercel.svg",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/vercelai.html",
  },
  {
    title: "CrewAI",
    src: "img/crewai.svg",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/crewai.html",
  },
  {
    title: "AutoGen",
    src: "img/autogen.jpeg",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/autogen.html",
  },
  {
    title: "LlamaIndex",
    src: "img/llamaindex.svg",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/llamaindex.html",
  },
  {
    title: "DSPy",
    src: "img/dspy.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/dspy.html",
  },
  {
    title: "PydanticAI",
    src: "img/pydantic-ai.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/pydantic_ai.html",
  },
  {
    title: "Google ADK",
    src: "img/google-adk.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/google-adk.html",
  },
  {
    title: "Semantic Kernel",
    src: "img/semantic-kernel.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/semantic_kernel.html",
  },
  {
    title: "Strands Agent",
    src: "img/strands-agents.svg",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/strands.html",
  },
  {
    title: "Agno",
    src: "img/agno.jpeg",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/agno.html",
  },
  {
    title: "AgentCore",
    src: "img/agentcore-logo.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/bedrock-agentcore.html",
  },
  {
    title: "Agent Framework",
    src: "img/microsoft-agent-framework-logo.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/microsoft-agent-framework.html",
  },
  {
    title: "Smolagents",
    src: "img/smolagents.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/smolagents.html",
  },
  {
    title: "Haystack",
    src: "img/haystack.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/haystack.html",
  },
  {
    title: "AG2",
    src: "img/ag2.svg",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/ag2.html",
  },
  {
    title: "LiveKit Agents",
    src: "img/livekit-logo.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/livekit.html",
  },
  {
    title: "Pipecat",
    src: "img/pipecat.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/pipecat.html",
  },
  {
    title: "Koog",
    src: "img/koog.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/koog.html",
  },
  {
    title: "txtai",
    src: "img/txtai-logo.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/txtai.html",
  },
  // --- Agent Frameworks (TypeScript) ---
  {
    title: "Mastra",
    src: "img/mastra-logo.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/mastra.html",
  },
  {
    title: "VoltAgent",
    src: "img/voltagent-logo.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/voltagent.html",
  },
  // --- Agent Frameworks (Java) ---
  {
    title: "Spring AI",
    src: "img/spring-ai-logo.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/spring-ai.html",
  },
  // --- Tools ---
  {
    title: "Instructor",
    src: "img/instructor-logo.svg",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/instructor.html",
  },
  {
    title: "Prompt flow",
    src: "img/promptflow.svg",
    href: "https://mlflow.org/docs/latest/ml/model/index.html#promptflow-promptflow-experimental",
  },
  {
    title: "Langflow",
    src: "img/langflow.svg",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langflow.html",
  },
  // --- Gateways ---
  {
    title: "OpenRouter",
    src: "img/openrouter-logo.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/openrouter.html",
  },
  {
    title: "Portkey",
    src: "img/portkey-logo.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/portkey.html",
  },
  {
    title: "Helicone",
    src: "img/helicone-logo.png",
    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/helicone.html",
  },
];

export default function GenAi(): JSX.Element {
  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://mlflow.org/genai" />
        <link rel="canonical" href="https://mlflow.org/genai" />
      </Head>

      <Layout>
      <AboveTheFold
        title={
          <span className="text-[48px] xxs:text-[64px] xs:text-[80px] leading-[110%]">
            Open Source Agent Engineering Platform
          </span>
        }
        body={[
          <>
            Confidently ship agents and LLM applications to production with
            built-in{" "}
            <HighlightedKeyword href="/genai/observability">
              observability
            </HighlightedKeyword>
            ,{" "}
            <HighlightedKeyword href="/genai/evaluations">
              evaluation
            </HighlightedKeyword>
            ,{" "}
            <HighlightedKeyword href="/genai/prompt-registry">
              prompt management
            </HighlightedKeyword>
            ,{" "}
            <HighlightedKeyword href="/genai/observability">
              monitoring
            </HighlightedKeyword>
            ,{" "}
            <HighlightedKeyword href="/genai/ai-gateway">
              cost controls
            </HighlightedKeyword>
            , and much more.
          </>,
        ]}
        bodyColor="white"
        bodySize="xl"
        actions={
          <div className="flex flex-col items-center gap-4">
            <div className="flex flex-wrap justify-center items-center gap-4">
              <Link to="#get-started">
                <Button variant="primary" size="medium">
                  Get Started
                </Button>
              </Link>
              <Link to={MLFLOW_GENAI_DOCS_URL}>
                <Button variant="outline" size="medium">
                  View Docs
                </Button>
              </Link>
            </div>
            <TrustPills />
          </div>
        }
      />

      <LogosCarousel />

      <div className="w-full px-4 md:px-8 lg:px-16 pb-36">
        <div className="max-w-7xl mx-auto">
          <StickyFeaturesGrid
            features={
              categories.find((c) => c.id === "llm-agents")?.features ?? []
            }
            colorTheme="red"
          />
        </div>
      </div>

      <StatsBand
        title="Most Adopted Open Source AI Platform"
        body={
          <>
            Backed by Linux Foundation, MLflow has been fully committed to
            open source for 5+ years. Now trusted by thousands of organizations
            and research teams worldwide to power their{" "}
            <Link
              href="/llmops"
              style={{ color: "inherit", textDecoration: "underline" }}
            >
              LLMOps
            </Link>{" "}
            workflows.
          </>
        }
      />
      <EcosystemList
        title="Works with Any LLM and Agent Framework"
        body="From LLM providers to agent frameworks — MLflow integrates seamlessly with 100+ tools across the AI ecosystem. Supports any programming language and natively integrates with OpenTelemetry and MCP."
        items={genaiIntegrations}
      />
      <BenefitsSection communityLabel="LLMOps" />
      <ProcessSection
        subtitle="From zero to production-ready agents in minutes. No complex setup or major code changes required."
        colorTheme="red"
      />
      <BelowTheFold contentType="genai" hideGetStarted>
        <LatestNews />
      </BelowTheFold>
    </Layout>
    </>
  );
}
