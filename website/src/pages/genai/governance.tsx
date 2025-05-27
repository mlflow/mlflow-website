import {
  Layout,
  SectionLabel,
  Grid,
  GridItem,
  GetStartedWithMLflow,
  SocialWidget,
  GetStartedButton,
  Heading,
  Body,
} from "../../components";

const FakeImage = () => (
  <div className="w-full aspect-[3/2] bg-black rounded-lg border border-[rgba(255,255,255,0.08)]"></div>
);

export default function Governance() {
  return (
    <Layout variant="red" direction="up">
      <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-container">
        <div className="flex flex-col justify-center items-center gap-6 w-full">
          <SectionLabel color="red" label="QUALITY METRICS" />
          <Heading level={1}>Enterprise governance with Unity Catalog</Heading>
          <div className="max-w-3xl mx-auto text-center">
            <Body size="l">
              MLflow is integrated with Unity Catalog to track the lifecycle and
              lineage of your app’s assets - models, prompts, datasets, and
              metrics - and apply access controls.
            </Body>
          </div>
          <GetStartedButton />
        </div>
        <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
      </div>
      <div className="flex flex-col gap-40 w-full px-6 md:px-20 max-w-container">
        <Grid columns={2}>
          <GridItem>
            <div className="flex flex-col gap-10">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Unified Data and AI governance</h3>
              </div>

              <Body size="l">
                Unity Catalog provides central, unified governance over all your
                data and AI assets - including GenAI and classic/deep learning
                ML. Enforce access controls and automatically track lineage.
              </Body>
            </div>
            <FakeImage />
          </GridItem>
          <GridItem direction="reverse">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Prompt Registry</h3>
              <Body size="l">
                Track every prompt template, its version history, and deployment
                lifecycle in the Unity Catalog. Each prompt is linked to its
                associated apps and evaluation results. Integrate prompts into
                your app’s code base via our SDK to allow non-technical users to
                edit prompts without access to your code base.
              </Body>
            </div>
            <FakeImage />
          </GridItem>
          <GridItem width="wide">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">App Version Registry</h3>
              <Body size="l">
                Track every application version and its associated prompts and
                evaluation results in the Unity Catalog.
              </Body>
              <Body size="l">
                You can store the app’s code as a deployable asset or link to
                Git commits to integrate with your existing software development
                lifecycle.
              </Body>
            </div>
            <FakeImage />
          </GridItem>
          <GridItem width="wide" direction="reverse">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">
                Evaluation Dataset & Metric Registry
              </h3>
              <Body size="l">
                Track and manage evaluation datasets and custom metrics as UC
                assets.
              </Body>
            </div>
            <FakeImage />
          </GridItem>
          <GridItem width="wide">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Collaboration & Sharing</h3>
              <Body size="l">
                Enable cross-organization discovery and sharing of prompts and
                apps
              </Body>
              <Body size="l">
                You can store the app’s code as a deployable asset or link to
                Git commits to integrate with your existing software development
                lifecycle.
              </Body>
            </div>
            <FakeImage />
          </GridItem>
        </Grid>
        <GetStartedWithMLflow />
        <SocialWidget variant="red" />
      </div>
    </Layout>
  );
}
