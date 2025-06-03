import {
  Layout,
  Grid,
  GridItem,
  AboveTheFold,
  BelowTheFold,
  Card,
} from "../../components";

const FakeImage = () => (
  <div className="w-full aspect-[3/2] bg-black rounded-lg border border-[rgba(255,255,255,0.08)]"></div>
);

export default function Governance() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="Quality metrics"
        title="Enterprise governance with Unity Catalog"
        body="MLflow is integrated with Unity Catalog to track the lifecycle and lineage of your app’s assets - models, prompts, datasets, and metrics - and apply access controls."
        hasGetStartedButton
      >
        <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
      </AboveTheFold>

      <Grid columns={2}>
        <GridItem>
          <Card
            title="Unified Data and AI governance"
            body="Unity Catalog provides central, unified governance over all your data and AI assets - including GenAI and classic/deep learning ML. Enforce access controls and automatically track lineage."
            image={<FakeImage />}
          />
        </GridItem>
        <GridItem direction="reverse">
          <Card
            title="Prompt Registry"
            body="Track every prompt template, its version history, and deployment lifecycle in the Unity Catalog. Each prompt is linked to its associated apps and evaluation results. Integrate prompts into your app’s code base via our SDK to allow non-technical users to edit prompts without access to your code base."
            image={<FakeImage />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="App Version Registry"
            body={[
              "Track every application version and its associated prompts and evaluation results in the Unity Catalog.",
              "You can store the app’s code as a deployable asset or link to Git commits to integrate with your existing software development lifecycle.",
            ]}
            image={<FakeImage />}
          />
        </GridItem>
        <GridItem width="wide" direction="reverse">
          <Card
            title="Evaluation Dataset & Metric Registry"
            body="Track and manage evaluation datasets and custom metrics as UC assets."
            image={<FakeImage />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Collaboration & Sharing"
            body={[
              "Enable cross-organization discovery and sharing of prompts and apps",
              "You can store the app’s code as a deployable asset or link to Git commits to integrate with your existing software development lifecycle.",
            ]}
            image={<FakeImage />}
          />
        </GridItem>
      </Grid>

      <BelowTheFold />
    </Layout>
  );
}
