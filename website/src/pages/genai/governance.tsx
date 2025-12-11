import { MLFLOW_DOCS_URL } from "@site/src/constants";
import {
  Layout,
  AboveTheFold,
  BelowTheFold,
  HeroImage,
  StickyGrid,
} from "../../components";
import CardHero from "@site/static/img/GenAI_governance/GenAI_governance_hero.png";
import Card1 from "@site/static/img/GenAI_governance/GenAI_governance_1.png";
import Card2 from "@site/static/img/GenAI_governance/GenAI_governance_2.png";
import Card3 from "@site/static/img/GenAI_governance/GenAI_governance_3.png";
import Card4 from "@site/static/img/GenAI_governance/GenAI_governance_4.png";
import Card5 from "@site/static/img/GenAI_governance/GenAI_governance_5.png";

export default function Governance() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="Governance"
        title="Enterprise governance with Unity Catalog"
        body="MLflow is integrated with Unity Catalog to track the lifecycle and lineage of your app’s assets - models, prompts, datasets, and metrics - and apply access controls."
        hasGetStartedButton={MLFLOW_DOCS_URL}
      >
        <HeroImage src={CardHero} alt="" />
      </AboveTheFold>

      <StickyGrid
        cards={[
          {
            title: "Unified Data and AI governance",
            body: "Unity Catalog provides central, unified governance over all your data and AI assets - including GenAI and classic/deep learning ML. Enforce access controls and automatically track lineage.",
            image: <img src={Card1} alt="" />,
          },
          {
            title: "Prompt Registry",
            body: "Track every prompt template, its version history, and deployment lifecycle in the Unity Catalog. Each prompt is linked to its associated apps and evaluation results. Integrate prompts into your app’s code base via our SDK to allow non-technical users to edit prompts without access to your code base.",
            image: <img src={Card2} alt="" />,
          },
          {
            title: "App Version Registry",
            body: [
              "Track every application version and its associated prompts and evaluation results in the Unity Catalog.",
              "You can store the app’s code as a deployable asset or link to Git commits to integrate with your existing software development lifecycle.",
            ],
            image: <img src={Card3} alt="" />,
          },
          {
            title: "Collaboration & Sharing",
            body: [
              "Enable cross-organization discovery and sharing of prompts and apps",
              "You can store the app’s code as a deployable asset or link to Git commits to integrate with your existing software development lifecycle.",
            ],
            image: <img src={Card4} alt="" />,
          },
          {
            title: "Evaluation Dataset & Metric Registry",
            body: "Track and manage evaluation datasets and custom metrics as UC assets.",
            image: <img src={Card5} alt="" />,
          },
        ]}
      />

      <BelowTheFold contentType="genai" />
    </Layout>
  );
}
