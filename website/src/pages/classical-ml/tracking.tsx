import { MLFLOW_GET_STARTED_URL } from "@site/src/constants";
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

export default function Tracking() {
  return (
    <Layout variant="blue" direction="up">
      <AboveTheFold
        sectionLabel="Tracking"
        title="Comprehensive Experiment Tracking for ML Excellence"
        body=" Document, compare, and reproduce your machine learning experiments with MLflow's powerful tracking capabilities"
        hasGetStartedButton={MLFLOW_GET_STARTED_URL}
      >
        <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
      </AboveTheFold>

      <Grid columns={2}>
        <GridItem>
          <Card
            title="Complete Experiment Lifecycle"
            body="MLflow Tracking automatically captures parameters, code versions, metrics, and model weights for each training iteration. Log trained models, visualizations, interface signatures, and data samples to ensure complete reproducibility across your entire ML workflow"
            image={<FakeImage />}
          />
        </GridItem>
        <GridItem>
          <Card
            title="Visual Comparison"
            body="Compare results across multiple experiments with MLflow's powerful visualization tools. Quickly identify best-performing models and retrieve their corresponding code and parameters based on different metrics of interest across various projects."
            image={<FakeImage />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Seamless Collaboration"
            body="Organize models and iterations into experiments for easy team collaboration while maintaining traceability. Enable team members to share results while maintaining a unified view of all projects through a single interface."
            image={<FakeImage />}
          />
        </GridItem>
      </Grid>

      <BelowTheFold />
    </Layout>
  );
}
