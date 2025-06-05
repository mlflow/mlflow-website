import { MLFLOW_DOCS_URL } from "@site/src/constants";
import {
  Layout,
  AboveTheFold,
  BelowTheFold,
  HeroImage,
  StickyGrid,
} from "../../components";
import CardHero from "@site/static/img/GenAI_prompts/GenAI_prompts_hero.png";
import Card1 from "@site/static/img/GenAI_prompts/GenAI_prompts_1.png";
import Card2 from "@site/static/img/GenAI_prompts/GenAI_prompts_2.png";
import Card3 from "@site/static/img/GenAI_prompts/GenAI_prompts_3.png";
import Card4 from "@site/static/img/GenAI_prompts/GenAI_prompts_4.png";

export default function PromptRegistryVersioning() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="Prompt registry"
        title="The single source of truth for your prompts"
        body="Create, store, and version prompts easily in the Prompt Registry."
        hasGetStartedButton={MLFLOW_DOCS_URL}
      >
        <HeroImage src={CardHero} alt="" />
      </AboveTheFold>

      <StickyGrid
        cards={[
          {
            title: "Create and edit prompts",
            body: "Define prompt templates with variables, manage versions with commit messages and metadata, and compare changes.",
            image: <img src={Card1} alt="" />,
          },
          {
            title: "Compare prompts",
            body: "Track changes across prompt versions with a built-in diff view for easier prompt iteration and change management.",
            image: <img src={Card2} alt="" />,
          },
          {
            title: "Evaluate Prompts",
            body: "Set up evaluation experiments, compare different prompt versions, analyze results, and select the most effective prompts.",
            image: <img src={Card3} alt="" />,
          },
          {
            title: "Manage prompt lifecycle with aliases",
            body: "Use aliases (e.g., development, staging, production) to manage prompt versions across environments and implement governance.",
            image: <img src={Card4} alt="" />,
          },
        ]}
      />

      <BelowTheFold />
    </Layout>
  );
}
