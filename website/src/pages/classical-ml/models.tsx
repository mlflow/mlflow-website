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

export default function Models() {
  return (
    <Layout variant="blue" direction="up">
      <AboveTheFold
        sectionLabel="Models"
        title="Transform ML Experiments into Production-Ready Models"
        body="Build, deploy, and manage machine learning models with confidence using MLflow's comprehensive model lifecycle management"
        hasGetStartedButton={MLFLOW_GET_STARTED_URL}
      >
        <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
      </AboveTheFold>

      <Grid columns={2}>
        <GridItem>
          <Card
            title="Unified Model Format"
            body="MLflow's MLModel file provides a standardized structure for packaging models from any framework, capturing essential dependencies and input/output specifications. This consistent packaging approach eliminates integration friction while ensuring models can be reliably deployed across any environment."
            image={<FakeImage />}
          />
        </GridItem>
        <GridItem>
          <Card
            title="Comprehensive Model Metadata"
            body="Track crucial model requirements and artifacts including data schemas, preprocessing steps, and environment dependencies automatically with MLflow's metadata system. Create fully reproducible model packages that document the complete model context for simplified governance and troubleshooting."
            image={<FakeImage />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Flexible Deployment Options"
            body="Deploy models as Docker containers, Python functions, REST endpoints, or directly to various serving platforms with MLflow's versatile deployment capabilities. Streamline the transition from development to production with consistent model behavior across any target environment, from local testing to cloud-based serving."
            image={<FakeImage />}
          />
        </GridItem>
      </Grid>

      <BelowTheFold />
    </Layout>
  );
}
