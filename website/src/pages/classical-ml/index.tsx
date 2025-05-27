import {
  Layout,
  CopyCommand,
  VerticalTabs,
  VerticalTabsList,
  VerticalTabsTrigger,
  VerticalTabsContent,
  SectionLabel,
  LogosCarousel,
  Grid,
  GridItem,
  LatestNews,
  Heading,
  Body,
  AboveTheFold,
  BelowTheFold,
  Section,
} from "../../components";

export default function GenAi(): JSX.Element {
  return (
    <Layout variant="blue">
      <AboveTheFold
        title="Mastering the ML lifecycle"
        body="From experiment to production, MLflow streamlines your complete machine learning journey with enterprise-grade tracking, model management, and deployment."
      >
        <CopyCommand code="pip install mlflow" />
      </AboveTheFold>

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
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Industry pioneer</h3>
              <Body size="m">
                MLflow has established itself as a pioneering open-source
                platform for managing the end-to-end machine learning lifecycle.
                Created by Databricks, it has become one of the most widely
                adopted MLOps tools in the industry, with integration support
                from major cloud providers.
              </Body>
            </div>
          </GridItem>
          <GridItem>
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Framework neutrality</h3>
              <Body size="m">
                MLflow's framework-agnostic design is one of its strongest
                differentiators. Unlike proprietary solutions that lock you into
                specific ecosystems, MLflow works seamlessly with all popular ML
                frameworks including scikit-learn, PyTorch, TensorFlow, and
                XGBoost.
              </Body>
            </div>
          </GridItem>
          <GridItem>
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Comprehensive Lifecycle Management</h3>
              <Body size="m">
                MLflow uniquely addresses the complete machine learning
                lifecycle through four integrated components: - MLflow Tracking
                for logging parameters, metrics, and artifacts - MLflow Projects
                for reproducible code packaging - MLflow Models for standardized
                deployment - MLflow Model Registry for centralized version
                management
              </Body>
            </div>
          </GridItem>
          <GridItem>
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Enterprise Adoption</h3>
              <Body size="m">
                MLflow's impact extends beyond its technical capabilities. It
                has gained significant traction among enterprise teams requiring
                robust experiment tracking and model lifecycle management.
                Databricks offers a managed MLflow service with enhanced
                security and scalability.
              </Body>
            </div>
          </GridItem>
        </Grid>
      </Section>

      <BelowTheFold>
        <LatestNews />
      </BelowTheFold>
    </Layout>
  );
}
