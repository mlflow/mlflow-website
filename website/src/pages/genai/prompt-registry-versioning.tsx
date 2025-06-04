import {
  Layout,
  Grid,
  GridItem,
  AboveTheFold,
  BelowTheFold,
  Card,
  Section,
} from "../../components";
import CardHero from "@site/static/img/GenAI_prompts&versions/GenAI_prompt&versioning_hero.png";
import Card1 from "@site/static/img/GenAI_prompts&versions/GenAI_prompt&versioning_1.png";
import Card2 from "@site/static/img/GenAI_prompts&versions/GenAI_prompt&versioning_2.png";
import Card3 from "@site/static/img/GenAI_prompts&versions/GenAI_prompt&versioning_3.png";
import Card4 from "@site/static/img/GenAI_prompts&versions/GenAI_prompt&versioning_4.png";

export default function PromptRegistryVersioning() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="Prompt registry and versioning"
        title="Prompt registry & versioning"
        body="Manage prompts and track versions of GenAI applications. Create, store, and version prompts in the Prompt Registry, and track and compare different versions of GenAI applications to ensure quality and maintainability."
        hasGetStartedButton
      >
        <div className="w-full max-w-[800px] rounded-lg overflow-hidden mx-auto">
          <img src={CardHero} alt="" />
        </div>
      </AboveTheFold>

      <Section title="Prompt Registry">
        <Grid columns={2}>
          <GridItem width="wide">
            <Card
              title="Prompt Registry"
              body="Centrally manage your prompts with robust version control, aliasing for deployments, and lineage tracking. Create, edit, and evaluate prompts, use them in applications and deployed environments, and manage their lifecycles effectively."
              image={<img src={Card1} alt="" />}
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Create and Edit Prompts"
              body="Define prompt templates with variables, manage versions with commit messages and metadata, and compare changes."
              image={<img src={Card2} alt="" />}
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Use Prompts in Apps"
              body="Load prompts from the registry using URIs, bind variables, and integrate with frameworks like LangChain or LlamaIndex. Log prompt versions with MLflow Models for lineage."
              image={<img src={Card3} alt="" />}
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Evaluate Prompts"
              body="Set up evaluation experiments, compare different prompt versions, analyze results, and select the most effective prompts."
              image={<img src={Card4} alt="" />}
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Manage Prompt Lifecycles with Aliases"
              body="Use aliases (e.g., development, staging, production) to manage prompt versions across environments and implement governance."
              image={<img src={Card4} alt="" />}
            />
          </GridItem>
        </Grid>
      </Section>

      <Section title="Versioning">
        <Grid columns={2}>
          <GridItem width="wide">
            <Card
              title="Version Tracking"
              body="Track different versions of your GenAI applications using LoggedModels. Link evaluation results, traces, and prompt versions to specific application versions. Optionally package application code for deployment and compare versions to understand performance impacts."
              image={<img src={Card4} alt="" />}
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Track Application Versions with MLflow"
              body="Use LoggedModel as a central metadata record linking to external code (e.g., Git commits), prompt versions, and configurations. Set active models for associating evaluations and traces."
              image={<img src={Card4} alt="" />}
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Optionally Package App Code & Files"
              body="Bundle application code, dependencies, and artifacts into a LoggedModel for deployment, especially for environments like Databricks Model Serving."
              image={<img src={Card4} alt="" />}
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Compare App Versions"
              body="Compare different LoggedModel versions using metrics like performance, cost, and quality scores to make data-driven decisions."
              image={<img src={Card4} alt="" />}
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Link Evaluation Results and Traces to App Versions"
              body="Automatically link evaluation metrics, outputs, and traces from `mlflow.genai.evaluate()` and autologging back to the specific LoggedModel version."
              image={<img src={Card4} alt="" />}
            />
          </GridItem>
        </Grid>
      </Section>

      <BelowTheFold />
    </Layout>
  );
}
