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
  Button,
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
          <Grid columns={2}>
            <GridItem lg-width="wide">
              <div className="flex flex-col gap-4">
                <div
                  role="heading"
                  aria-level={3}
                  className="text-white text-xl"
                >
                  Unified Workflow
                </div>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  MLflow streamlines your entire ML process with tracking,
                  packaging, and deployment capabilities, eliminating tool
                  fragmentation so you can focus on model development rather
                  than infrastructure
                </p>
                <a href="#" className="hidden md:block">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </a>
              </div>
              <div>
                <img src="/img/demo-image.png" alt="" />
              </div>
            </GridItem>
            <GridItem direction="reverse" lg-width="wide" lg-direction="normal">
              <div className="flex flex-col gap-4">
                <div
                  role="heading"
                  aria-level={3}
                  className="text-white text-xl"
                >
                  Reliable Reproducability
                </div>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Automatically logs parameters, weights, artifacts, code,
                  metrics, and dependencies to ensure experiments can be
                  restored accurately, enabling confident governance for
                  enterprise deployments.
                </p>
                <a href="#" className="hidden md:block">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </a>
              </div>
              <div>
                <img src="/img/demo-image.png" alt="" />
              </div>
            </GridItem>
            <GridItem width="wide">
              <div className="flex flex-col gap-4">
                <div
                  role="heading"
                  aria-level={3}
                  className="text-white text-xl"
                >
                  Framework Neutral
                </div>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Works seamlessly with popular tools like scikit-learn,
                  PyTorch, TensorFlow, and XGBoost without vendor lock-in,
                  providing flexibility with a common interface.
                </p>
                <a href="#" className="hidden md:block">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </a>
              </div>
              <div>
                <img src="/img/demo-image.png" alt="" />
              </div>
            </GridItem>
            <GridItem direction="reverse" lg-width="wide" lg-direction="normal">
              <div className="flex flex-col gap-4">
                <div
                  role="heading"
                  aria-level={3}
                  className="text-white text-xl"
                >
                  Deployment Ready
                </div>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Simplifies the path from experimentation to production with a
                  built-in registry that gives you complete control over model
                  states, whether sharing new approaches or deploying solutions.
                </p>
                <a href="#" className="hidden md:block">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </a>
              </div>
              <div>
                <img src="/img/demo-image.png" alt="" />
              </div>
            </GridItem>
            <GridItem lg-width="wide">
              <div className="flex flex-col gap-4">
                <div
                  role="heading"
                  aria-level={3}
                  className="text-white text-xl"
                >
                  Enterprise Ready
                </div>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Databricks-managed MLflow adds robust security, automated
                  scaling, and high availability for mission-critical workloads
                  while reducing operational overhead and delivering exceptional
                  performance.
                </p>
                <a href="#" className="hidden md:block">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </a>
              </div>
              <div>
                <img src="/img/demo-image.png" alt="" />
              </div>
            </GridItem>
          </Grid>
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
