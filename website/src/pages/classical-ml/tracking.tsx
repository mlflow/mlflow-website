import {
  Layout,
  Grid,
  GridItem,
  Body,
  AboveTheFold,
  BelowTheFold,
} from "../../components";

const FakeImage = () => (
  <div className="w-full aspect-[3/2] bg-black rounded-lg border border-[rgba(255,255,255,0.08)]"></div>
);

export default function Tracking() {
  return (
    <Layout variant="blue" direction="up">
      <AboveTheFold
        sectionLabel="Tracking"
        title="Comprehensive Experiment Tracking for ML Excellence"
        body=" Document, compare, and reproduce your machine learning experiments with MLflow's powerful tracking capabilities"
        hasGetStartedButton
      >
        <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
      </AboveTheFold>

      <Grid columns={2}>
        <GridItem>
          <div className="flex flex-col gap-4">
            <h3 className="text-white">Complete Experiment Lifecycle</h3>
            <Body size="l">
              MLflow Tracking automatically captures parameters, code versions,
              metrics, and model weights for each training iteration. Log
              trained models, visualizations, interface signatures, and data
              samples to ensure complete reproducibility across your entire ML
              workflow
            </Body>
          </div>
          <FakeImage />
        </GridItem>
        <GridItem>
          <div className="flex flex-col gap-4">
            <h3 className="text-white">Visual Comparison</h3>
            <Body size="l">
              Compare results across multiple experiments with MLflow's powerful
              visualization tools. Quickly identify best-performing models and
              retrieve their corresponding code and parameters based on
              different metrics of interest across various projects.
            </Body>
          </div>
          <FakeImage />
        </GridItem>
        <GridItem width="wide">
          <div className="flex flex-col gap-4">
            <h3 className="text-white">Seamless Collaboration</h3>
            <Body size="l">
              Organize models and iterations into experiments for easy team
              collaboration while maintaining traceability. Enable team members
              to share results while maintaining a unified view of all projects
              through a single interface.
            </Body>
          </div>
          <FakeImage />
        </GridItem>
      </Grid>

      <BelowTheFold />
    </Layout>
  );
}
