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

export default function UnifiedRegistry() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="Unified registry"
        title="Centralized Model Governance and Discovery"
        body="Streamline your ML workflows with MLflow's comprehensive model registry for version control, approvals, and deployment management"
        hasGetStartedButton={MLFLOW_GET_STARTED_URL}
      >
        <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
      </AboveTheFold>

      <Grid columns={2}>
        <GridItem>
          <Card
            title="Seamless Unity Catalog Integration"
            body="MLflow Model Registry integrates directly with Unity Catalog to provide enterprise-grade governance across your entire ML asset portfolio. Apply consistent security policies, lineage tracking, and access controls to both data and models through a unified permission system."
            image={<FakeImage />}
          />
        </GridItem>
        <GridItem>
          <Card
            title="Stage-Based Model Lifecycle"
            body="Move models through customizable staging environments (Development, Staging, Production, or any stage alias you choose) with built-in approval workflow capabilities and automated notifications. Maintain complete audit trails of model transitions with detailed metadata about who approved changes and when they occurred."
            image={<FakeImage />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Model Deployment Flexibility"
            body="Deploy models as containers, batch jobs, or REST endpoints with MLflow's streamlined deployment capabilities that eliminate boilerplate code. Use model aliases to create named references that enable seamless model updates in production without changing your application code."
            image={<FakeImage />}
          />
        </GridItem>
      </Grid>

      <BelowTheFold />
    </Layout>
  );
}
