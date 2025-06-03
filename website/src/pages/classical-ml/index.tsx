import { MLFLOW_GET_STARTED_URL } from "@site/src/constants";
import {
  Layout,
  VerticalTabs,
  VerticalTabsList,
  VerticalTabsTrigger,
  VerticalTabsContent,
  LogosCarousel,
  Grid,
  GridItem,
  LatestNews,
  AboveTheFold,
  BelowTheFold,
  Section,
  Card,
} from "../../components";

export default function GenAi(): JSX.Element {
  return (
    <Layout variant="blue">
      <AboveTheFold
        title="Mastering the ML lifecycle"
        body="From experiment to production, MLflow streamlines your complete machine learning journey with enterprise-grade tracking, model management, and deployment."
        hasGetStartedButton={MLFLOW_GET_STARTED_URL}
      />

      <Section
        label="Core features"
        title="Build confidently, deploy seamlessly"
        body="Cover experimentation, reproducibility, deployment, and a central model registry"
      >
        <VerticalTabs
          defaultValue="unified-workflow"
          className="w-full my-12 px-10"
        >
          <VerticalTabsList>
            <VerticalTabsTrigger
              value="unified-workflow"
              label="Unified Workflow"
              description="MLflow streamlines your entire ML process with tracking, packaging, and deployment capabilities, eliminating tool fragmentation so you can focus on model development rather than infrastructure"
            />
            <VerticalTabsTrigger
              value="reliable-reproducability"
              label="Reliable Reproducability"
              description="Automatically logs parameters, weights, artifacts, code, metrics, and dependencies to ensure experiments can be restored accurately, enabling confident governance for enterprise deployments."
            />
            <VerticalTabsTrigger
              value="framework-neutral"
              label="Framework Neutral"
              description="Works seamlessly with popular tools like scikit-learn, PyTorch, TensorFlow, and XGBoost without vendor lock-in, providing flexibility with a common interface."
            />
            <VerticalTabsTrigger
              value="deployment-ready"
              label="Deployment Ready"
              description="Simplifies the path from experimentation to production with a built-in registry that gives you complete control over model states, whether sharing new approaches or deploying solutions."
            />
            <VerticalTabsTrigger
              value="enterprise-ready"
              label="Enterprise Ready"
              description="Databricks-managed MLflow adds robust security, automated scaling, and high availability for mission-critical workloads while reducing operational overhead and delivering exceptional performance."
            />
          </VerticalTabsList>
          <VerticalTabsContent value="unified-workflow">
            <img src="/img/demo-image.png" />
          </VerticalTabsContent>
          <VerticalTabsContent value="reliable-reproducability">
            <img src="/img/demo-image.png" />
          </VerticalTabsContent>
          <VerticalTabsContent value="framework-neutral">
            <img src="/img/demo-image.png" />
          </VerticalTabsContent>
          <VerticalTabsContent value="deployment-ready">
            <img src="/img/demo-image.png" />
          </VerticalTabsContent>
          <VerticalTabsContent value="enterprise-ready">
            <img src="/img/demo-image.png" />
          </VerticalTabsContent>
        </VerticalTabs>
      </Section>

      <LogosCarousel />

      <Section label="Why us?" title="Why MLflow is unique">
        <Grid columns={2}>
          <GridItem>
            <Card
              title="Industry pioneer"
              bodySize="m"
              body="MLflow has established itself as a pioneering open-source platform for managing the end-to-end machine learning lifecycle. Created by Databricks, it has become one of the most widely adopted MLOps tools in the industry, with integration support from major cloud providers."
            />
          </GridItem>
          <GridItem>
            <Card
              title="Framework neutrality"
              bodySize="m"
              body="MLflow's framework-agnostic design is one of its strongest differentiators. Unlike proprietary solutions that lock you into specific ecosystems, MLflow works seamlessly with all popular ML frameworks including scikit-learn, PyTorch, TensorFlow, and XGBoost."
            />
          </GridItem>
          <GridItem>
            <Card
              title="Comprehensive Lifecycle Management"
              bodySize="m"
              body="MLflow uniquely addresses the complete machine learning lifecycle through four integrated components: - MLflow Tracking for logging parameters, metrics, and artifacts - MLflow Projects for reproducible code packaging - MLflow Models for standardized deployment - MLflow Model Registry for centralized version management"
            />
          </GridItem>
          <GridItem>
            <Card
              title="Enterprise Adoption"
              bodySize="m"
              body="MLflow's impact extends beyond its technical capabilities. It has gained significant traction among enterprise teams requiring robust experiment tracking and model lifecycle management. Databricks offers a managed MLflow service with enhanced security and scalability."
            />
          </GridItem>
        </Grid>
      </Section>

      <BelowTheFold>
        <LatestNews />
      </BelowTheFold>
    </Layout>
  );
}
