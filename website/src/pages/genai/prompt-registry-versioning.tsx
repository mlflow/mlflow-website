import {
  Layout,
  Grid,
  GridItem,
  AboveTheFold,
  BelowTheFold,
  Card,
  Section,
} from "../../components";

const FakeImage = () => (
  <div className="w-full aspect-[3/2] bg-black rounded-lg border border-[rgba(255,255,255,0.08)]"></div>
);

export default function PromptRegistryVersioning() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="Prompt & Version Management"
        title="Prompt & Version Management"
        body="Manage prompts and track versions of GenAI applications. Create, store, and version prompts in the Prompt Registry, and track and compare different versions of GenAI applications to ensure quality and maintainability."
        hasGetStartedButton
      >
        <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
      </AboveTheFold>

      <Section title="Prompt Registry">
        <Grid columns={2}>
          <GridItem width="wide">
            <Card
              title="Prompt Registry"
              body="Centrally manage your prompts with robust version control, aliasing for deployments, and lineage tracking. Create, edit, and evaluate prompts, use them in applications and deployed environments, and manage their lifecycles effectively."
              image={<FakeImage />}
            />
          </GridItem>
          <GridItem direction="reverse">
            <Card
              title="Create and Edit Prompts"
              body="Define prompt templates with variables, manage versions with commit messages and metadata, and compare changes."
              image={<FakeImage />}
            />
          </GridItem>
          <GridItem>
            <Card
              title="Use Prompts in Apps"
              body="Load prompts from the registry using URIs, bind variables, and integrate with frameworks like LangChain or LlamaIndex. Log prompt versions with MLflow Models for lineage."
              image={<FakeImage />}
            />
          </GridItem>
          <GridItem direction="reverse">
            <Card
              title="Evaluate Prompts"
              body="Set up evaluation experiments, compare different prompt versions, analyze results, and select the most effective prompts."
              image={<FakeImage />}
            />
          </GridItem>
          <GridItem>
            <Card
              title="Manage Prompt Lifecycles with Aliases"
              body="Use aliases (e.g., development, staging, production) to manage prompt versions across environments and implement governance."
              image={<FakeImage />}
            />
          </GridItem>
        </Grid>
      </Section>

      <Section title="Version Tracking">
        <Grid columns={2}>
          <GridItem>
            <Card
              title="Version Tracking"
              body="Track different versions of your GenAI applications using LoggedModels. Link evaluation results, traces, and prompt versions to specific application versions. Optionally package application code for deployment and compare versions to understand performance impacts."
              image={<FakeImage />}
            />
          </GridItem>
          <GridItem direction="reverse">
            <Card
              title="Track Application Versions with MLflow"
              body="Use LoggedModel as a central metadata record linking to external code (e.g., Git commits), prompt versions, and configurations. Set active models for associating evaluations and traces."
              image={<FakeImage />}
            />
          </GridItem>
          <GridItem>
            <Card
              title="Optionally Package App Code & Files"
              body="Bundle application code, dependencies, and artifacts into a LoggedModel for deployment, especially for environments like Databricks Model Serving."
              image={<FakeImage />}
            />
          </GridItem>
          <GridItem direction="reverse">
            <Card
              title="Compare App Versions"
              body="Compare different LoggedModel versions using metrics like performance, cost, and quality scores to make data-driven decisions."
              image={<FakeImage />}
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Link Evaluation Results and Traces to App Versions"
              body="Automatically link evaluation metrics, outputs, and traces from `mlflow.genai.evaluate()` and autologging back to the specific LoggedModel version."
              image={<FakeImage />}
            />
          </GridItem>
        </Grid>
      </Section>

      <BelowTheFold />
    </Layout>
  );
}
