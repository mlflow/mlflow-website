import Head from "@docusaurus/Head";
import {
  Layout,
  AboveTheFold,
  BelowTheFold,
  HeroImage,
  StickyGrid,
} from "../../components";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import CardHero from "@site/static/img/GenAI_prompts/GenAI_prompts_hero.png";
import Card1 from "@site/static/img/GenAI_prompts/GenAI_prompts_1.png";
import Card2 from "@site/static/img/GenAI_prompts/GenAI_prompts_2.png";
import Card3 from "@site/static/img/GenAI_prompts/GenAI_prompts_3.png";
import Card4 from "@site/static/img/GenAI_prompts/GenAI_prompts_4.png";

const SEO_TITLE = "Prompt Registry | MLflow AI Platform";
const SEO_DESCRIPTION =
  "Create, version, and manage prompt templates with MLflow's AI Engineering Platform. Compare changes, evaluate prompt versions, and automatically optimize prompts.";

export default function PromptRegistryVersioning() {
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
          content="https://mlflow.org/genai/prompt-registry"
        />
        <link rel="canonical" href="https://mlflow.org/genai/prompt-registry" />
      </Head>

      <Layout>
        <AboveTheFold
          sectionLabel="Prompt registry"
          title="The single source of truth for your prompts"
          body="Create, store, and version prompts easily in the Prompt Registry."
          hasGetStartedButton={`${MLFLOW_GENAI_DOCS_URL}prompt-registry/create-and-edit-prompts/`}
        >
          <HeroImage src={CardHero} alt="MLflow prompt registry screenshot" />
        </AboveTheFold>

        <StickyGrid
          cards={[
            {
              title: "Create and edit prompts",
              body: "Define prompt templates with variables, manage versions with commit messages and metadata, and compare changes.",
              image: (
                <img src={Card1} alt="MLflow prompt registry screenshot" />
              ),
            },
            {
              title: "Compare prompts",
              body: "Track changes across prompt versions with a built-in diff view for easier prompt iteration and change management.",
              image: (
                <img src={Card2} alt="MLflow prompt registry screenshot" />
              ),
            },
            {
              title: "Evaluate Prompts",
              body: "Set up evaluation experiments, compare different prompt versions, analyze results, and select the most effective prompts.",
              image: (
                <img src={Card3} alt="MLflow prompt registry screenshot" />
              ),
            },
            {
              title: "Manage prompt lifecycle with aliases",
              body: "Use aliases (e.g., development, staging, production) to manage prompt versions across environments and implement governance.",
              image: (
                <img src={Card4} alt="MLflow prompt registry screenshot" />
              ),
            },
          ]}
        />

        <BelowTheFold contentType="genai" />
      </Layout>
    </>
  );
}
