import {
  Layout,
  Grid,
  GridItem,
  GetStartedWithMLflow,
  SocialWidget,
  GetStartedButton,
  Body,
  AboveTheFold,
} from "../../components";

const FakeImage = () => (
  <div className="w-full aspect-[3/2] bg-black rounded-lg border border-[rgba(255,255,255,0.08)]"></div>
);

export default function Serving() {
  return (
    <Layout variant="blue" direction="up">
      <AboveTheFold
        sectionLabel="Serving"
        title="Flexible Model Deployment for Any Production Environment"
        body="Deploy your ML and DL models with confidence using MLflow's versatile serving options for real-time and batch inference"
      >
        <div className="flex flex-col gap-16 items-center -mt-10">
          <GetStartedButton />
          <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
        </div>
      </AboveTheFold>
      <div className="flex flex-col gap-40 w-full px-6 md:px-20 max-w-container">
        <Grid columns={2}>
          <GridItem>
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Scalable Real-Time Serving</h3>
              <Body size="l">
                MLflow provides a unified, scalable interface for deploying
                models as REST APIs that automatically adjust to meet demand
                fluctuations. With managed deployment on Databricks, your
                endpoints can intelligently scale up or down based on traffic
                patterns, optimizing both performance and infrastructure costs
                with no manual intervention required.
              </Body>
            </div>
            <FakeImage />
          </GridItem>
          <GridItem>
            <div className="flex flex-col gap-4">
              <h3 className="text-white">High-Performance Batch Inference</h3>
              <Body size="l">
                Deploy production models for batch inference directly on Apache
                Spark, enabling efficient processing of billions of predictions
                on massive datasets
              </Body>
            </div>
            <FakeImage />
          </GridItem>
          <GridItem width="wide">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Comprehensive Deployment Options</h3>
              <Body size="l">
                Deploy models across multiple environments including Docker
                containers, cloud services like Databricks, Azure ML and AWS
                SageMaker, or Kubernetes clusters with consistent behavior.
              </Body>
            </div>
            <FakeImage />
          </GridItem>
        </Grid>
        <GetStartedWithMLflow />
        <SocialWidget />
      </div>
    </Layout>
  );
}
