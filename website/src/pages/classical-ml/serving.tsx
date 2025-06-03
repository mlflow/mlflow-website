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

export default function Serving() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="Serving"
        title="Flexible Model Deployment for Any Production Environment"
        body="Deploy your ML and DL models with confidence using MLflow's versatile serving options for real-time and batch inference"
        hasGetStartedButton={MLFLOW_GET_STARTED_URL}
      >
        <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
      </AboveTheFold>

      <Grid columns={2}>
        <GridItem>
          <Card
            title="Scalable Real-Time Serving"
            body="MLflow provides a unified, scalable interface for deploying models as REST APIs that automatically adjust to meet demand fluctuations. With managed deployment on Databricks, your endpoints can intelligently scale up or down based on traffic patterns, optimizing both performance and infrastructure costs with no manual intervention required."
            image={<FakeImage />}
          />
        </GridItem>
        <GridItem>
          <Card
            title="High-Performance Batch Inference"
            body="Deploy production models for batch inference directly on Apache Spark, enabling efficient processing of billions of predictions on massive datasets"
            image={<FakeImage />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Comprehensive Deployment Options"
            body="Deploy models across multiple environments including Docker containers, cloud services like Databricks, Azure ML and AWS SageMaker, or Kubernetes clusters with consistent behavior."
            image={<FakeImage />}
          />
        </GridItem>
      </Grid>

      <BelowTheFold />
    </Layout>
  );
}
