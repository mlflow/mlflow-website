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
  GridRow,
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
        <div className="flex flex-col gap-16 w-full px-6 md:px-20">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <h1 className="text-center text-wrap">
              Mastering the ML lifecycle
            </h1>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto">
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
      <div className="flex flex-col gap-40 w-full px-6 md:px-20">
        <div className="flex flex-col w-full items-center justify-center gap-16">
          <div className="flex flex-col w-full items-center justify-center gap-6">
            <SectionLabel color="green" label="CORE FEATURES" />
            <h1>Build confidently, deploy seamlessly</h1>
            <p>
              Cover experimentation, reproducibility, deployment, and a central
              model registry
            </p>
          </div>
          <VerticalTabs defaultValue="tab1" className="w-full my-12 px-10">
            <VerticalTabsList>
              <VerticalTabsTrigger
                value="tab1"
                label="Unified Workflow"
                description="MLflow streamlines your entire ML process with tracking, packaging, and deployment capabilities, eliminating tool fragmentation so you can focus on model development rather than infrastructure"
              />
              <VerticalTabsTrigger
                value="tab2"
                label="Reliable Reproducability"
                description="Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat ex non sit lorem eu proident."
              />
              <VerticalTabsTrigger
                value="tab3"
                label="Framework Neutral"
                description="Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat ex non sit lorem eu proident."
              />
              <VerticalTabsTrigger
                value="tab4"
                label="Deployment Ready"
                description="Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat ex non sit lorem eu proident."
              />
              <VerticalTabsTrigger
                value="tab5"
                label="Enterprise Ready"
                description="Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat ex non sit lorem eu proident."
              />
            </VerticalTabsList>
            <VerticalTabsContent value="tab1">
              <img src="/img/demo-image.png" />
            </VerticalTabsContent>
            <VerticalTabsContent value="tab2">
              <img src="/img/demo-image.png" />
            </VerticalTabsContent>
            <VerticalTabsContent value="tab3">
              <img src="/img/demo-image.png" />
            </VerticalTabsContent>

            <VerticalTabsContent value="tab4">
              <img src="/img/demo-image.png" />
            </VerticalTabsContent>

            <VerticalTabsContent value="tab5">
              <img src="/img/demo-image.png" />
            </VerticalTabsContent>
          </VerticalTabs>
        </div>
        <LogosCarousel
          images={[
            "img/companies/databricks.svg",
            "img/companies/microsoft.svg",
            "img/companies/meta.svg",
            "img/companies/mosaicml.svg",
            "img/companies/zillow.svg",
            "img/companies/toyota.svg",
            "img/companies/booking.svg",
            "img/companies/wix.svg",
            "img/companies/accenture.svg",
            "img/companies/asml.svg",
          ]}
        />
        <div className="flex flex-col items-center justify-center gap-16">
          <div className="flex flex-col gap-6">
            <SectionLabel color="green" label="WHY US?" />
            <h1>Why MLflow is unique</h1>
          </div>
          <Grid>
            <GridRow>
              <GridItem className="py-10 pr-0 md:pr-10 gap-10">
                <div className="flex flex-col gap-4">
                  <h3>Industry pioneer</h3>
                  <p className="text-white/60">
                    MLflow has established itself as a pioneering open-source
                    platform for managing the end-to-end machine learning
                    lifecycle. Created by Databricks, it has become one of the
                    most widely adopted MLOps tools in the industry, with
                    integration support from major cloud providers.
                  </p>
                </div>
              </GridItem>
              <GridItem className="py-10 pl-0 md:pl-10 gap-10">
                <div className="flex flex-col gap-4">
                  <h3>Framework neutrality</h3>
                  <p className="text-white/60">
                    MLflow's framework-agnostic design is one of its strongest
                    differentiators. Unlike proprietary solutions that lock you
                    into specific ecosystems, MLflow works seamlessly with all
                    popular ML frameworks including scikit-learn, PyTorch,
                    TensorFlow, and XGBoost.
                  </p>
                </div>
              </GridItem>
            </GridRow>
            <GridRow>
              <GridItem className="py-10 pr-0 md:pr-10 gap-10">
                <div className="flex flex-col gap-4">
                  <h3>Comprehensive Lifecycle Management</h3>
                  <p className="text-white/60">
                    MLflow uniquely addresses the complete machine learning
                    lifecycle through four integrated components: - MLflow
                    Tracking for logging parameters, metrics, and artifacts -
                    MLflow Projects for reproducible code packaging - MLflow
                    Models for standardized deployment - MLflow Model Registry
                    for centralized version management
                  </p>
                </div>
              </GridItem>
              <GridItem className="py-10 pl-0 md:pl-10 gap-10">
                <div className="flex flex-col gap-4">
                  <h3>Enterprise Adoption</h3>
                  <p className="text-white/60">
                    MLflow's impact extends beyond its technical capabilities.
                    It has gained significant traction among enterprise teams
                    requiring robust experiment tracking and model lifecycle
                    management. Databricks offers a managed MLflow service with
                    enhanced security and scalability.
                  </p>
                </div>
              </GridItem>
            </GridRow>
          </Grid>
        </div>
        <GetStartedWithMLflow />
        <LatestNews variant="green" />
        <SocialWidget variant="green" />
      </div>
    </Layout>
  );
}
