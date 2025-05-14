import {
  Layout,
  SectionLabel,
  Button,
  Grid,
  GridRow,
  GridItem,
  GetStartedWithMLflow,
  SocialWidget,
} from "../../components";

const FakeImage = () => (
  <div className="w-[600px] h-[400px] bg-black rounded-lg"></div>
);

export default function Governance() {
  return (
    <Layout variant="red">
      <div
        className="flex flex-col bg-[linear-gradient(to_top,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-2.png')]
 bg-center bg-no-repeat bg-cover w-full pt-42 pb-20 py-20 bg-size-[100%_340px]"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <SectionLabel color="red" label="QUALITY METRICS" />
            <h1 className="text-center text-wrap">
              Enterprise governance with Unity Catalog
            </h1>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white/60">
              MLflow is integrated with Unity Catalog to track the lifecycle and
              lineage of your app’s assets - models, prompts, datasets, and
              metrics - and apply access controls.
            </p>
            <Button>Get Started</Button>
          </div>
          <div className="w-[800px] h-[450px] bg-black rounded-lg mx-auto"></div>
        </div>
      </div>
      <div className="flex flex-col gap-40 w-full px-6 md:px-20">
        <Grid>
          <GridRow>
            <GridItem className="flex flex-col md:flex-row gap-6 md:gap-20 py-10 justify-between items-center">
              <div className="flex flex-col gap-10">
                <div className="flex flex-col gap-4">
                  <h3 className="text-white">
                    Unified Data and AI governancer
                  </h3>
                </div>

                <p className="text-white/60">
                  Unity Catalog provides central, unified governance over all
                  your data and AI assets - including GenAI and classic/deep
                  learning ML. Enforce access controls and automatically track
                  lineage.
                </p>
              </div>
              <FakeImage />
            </GridItem>
          </GridRow>
          <GridRow>
            <GridItem className="py-10 pr-0 md:pr-10 gap-10">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Prompt Registry</h3>
                <p className="text-white/60 text-lg">
                  Track every prompt template, its version history, and
                  deployment lifecycle in the Unity Catalog. Each prompt is
                  linked to its associated apps and evaluation results.
                  Integrate prompts into your app’s code base via our SDK to
                  allow non-technical users to edit prompts without access to
                  your code base.
                </p>
              </div>
              <FakeImage />
            </GridItem>
            <GridItem className="py-10 pl-0 md:pl-10 gap-10">
              <FakeImage />
              <div className="flex flex-col gap-4">
                <h3 className="text-white">App Version Registry</h3>
                <p className="text-white/60 text-lg">
                  Track every application version and its associated prompts and
                  evaluation results in the Unity Catalog.
                </p>
                <p className="text-white/60 text-lg">
                  You can store the app’s code as a deployable asset or link to
                  Git commits to integrate with your existing software
                  development lifecycle.
                </p>
              </div>
            </GridItem>
          </GridRow>
          <GridRow>
            <GridItem className="py-10 pr-0 md:pr-10 gap-10">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  Evaluation Dataset & Metric Registry
                </h3>
                <p className="text-white/60 text-lg">
                  Track and manage evaluation datasets and custom metrics as UC
                  assets.
                </p>
              </div>
              <FakeImage />
            </GridItem>
            <GridItem className="py-10 pl-0 md:pl-10 gap-10">
              <FakeImage />
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Collaboration & Sharing</h3>
                <p className="text-white/60 text-lg">
                  Enable cross-organization discovery and sharing of prompts and
                  apps
                </p>
                <p className="text-white/60 text-lg">
                  You can store the app’s code as a deployable asset or link to
                  Git commits to integrate with your existing software
                  development lifecycle.
                </p>
              </div>
            </GridItem>
          </GridRow>
        </Grid>
        <GetStartedWithMLflow />
        <SocialWidget variant="red" />
      </div>
    </Layout>
  );
}
