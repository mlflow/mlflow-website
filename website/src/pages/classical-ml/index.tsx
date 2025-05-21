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
  GetStartedWithMLflow,
  LatestNews,
  SocialWidget,
} from "../../components";

export default function GenAi(): JSX.Element {
  return (
    <Layout variant="blue">
      <div
        className="flex flex-col bg-[linear-gradient(to_bottom,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-3.png')]
 bg-top bg-no-repeat bg-cover w-full pt-42 pb-20 py-20"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-container">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <h1 className="text-center text-wrap">
              Mastering the ML lifecycle
            </h1>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white">
              From experiment to production, MLflow streamlines your complete
              machine learning journey with enterprise-grade tracking, model
              management, and deployment.
            </p>
          </div>
          <div className="flex flex-col md:flex-row gap-10">
            <CopyCommand code="pip install mlflow" />
          </div>
        </div>
      </div>
      <div className="flex flex-col gap-40 w-full px-6 md:px-20 max-w-container">
        <div className="flex flex-col w-full items-center justify-center gap-16">
          <div className="flex flex-col w-full items-center justify-center gap-6">
            <SectionLabel color="green" label="CORE FEATURES" />
            <h1>Build confidently, deploy seamlessly</h1>
            <p className="text-white">
              Cover experimentation, reproducibility, deployment, and a central
              model registry
            </p>
          </div>
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
        </div>
        <LogosCarousel />
        <div className="flex flex-col items-center justify-center gap-16">
          <div className="flex flex-col gap-6">
            <SectionLabel color="green" label="WHY US?" />
            <h1>Why MLflow is unique</h1>
          </div>
          <Grid columns={2}>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Industry pioneer</h3>
                <p className="text-white/60">
                  MLflow has established itself as a pioneering open-source
                  platform for managing the end-to-end machine learning
                  lifecycle. Created by Databricks, it has become one of the
                  most widely adopted MLOps tools in the industry, with
                  integration support from major cloud providers.
                </p>
              </div>
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Framework neutrality</h3>
                <p className="text-white/60">
                  MLflow's framework-agnostic design is one of its strongest
                  differentiators. Unlike proprietary solutions that lock you
                  into specific ecosystems, MLflow works seamlessly with all
                  popular ML frameworks including scikit-learn, PyTorch,
                  TensorFlow, and XGBoost.
                </p>
              </div>
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  Comprehensive Lifecycle Management
                </h3>
                <p className="text-white/60">
                  MLflow uniquely addresses the complete machine learning
                  lifecycle through four integrated components: - MLflow
                  Tracking for logging parameters, metrics, and artifacts -
                  MLflow Projects for reproducible code packaging - MLflow
                  Models for standardized deployment - MLflow Model Registry for
                  centralized version management
                </p>
              </div>
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Enterprise Adoption</h3>
                <p className="text-white/60">
                  MLflow's impact extends beyond its technical capabilities. It
                  has gained significant traction among enterprise teams
                  requiring robust experiment tracking and model lifecycle
                  management. Databricks offers a managed MLflow service with
                  enhanced security and scalability.
                </p>
              </div>
            </GridItem>
          </Grid>
        </div>
        <GetStartedWithMLflow />
        <LatestNews variant="green" />
        <SocialWidget variant="green" />
      </div>
    </Layout>
  );
}
