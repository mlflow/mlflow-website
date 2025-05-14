import {
  Layout,
  SectionLabel,
  Button,
  Grid,
  GridRow,
  GridItem,
  GetStartedWithMLflow,
  SocialWidget,
} from "../../components";

const FakeImage = () => (
  <div className="w-[600px] h-[400px] bg-black rounded-lg"></div>
);

export default function Serving() {
  return (
    <Layout variant="blue">
      <div
        className="flex flex-col bg-[linear-gradient(to_top,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-3.png')]
 bg-center bg-no-repeat bg-cover w-full pt-42 pb-20 py-20 bg-size-[100%_340px]"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-7xl mx-auto">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <SectionLabel color="green" label="SERVING" />
            <h1 className="text-center text-wrap">
              Flexible Model Deployment for Any Production Environment
            </h1>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white/60">
              Deploy your ML and DL models with confidence using MLflow's
              versatile serving options for real-time and batch inference
            </p>
            <Button>Get Started</Button>
          </div>
          <div className="w-[800px] h-[450px] bg-black rounded-lg mx-auto"></div>
        </div>
      </div>
      <div className="flex flex-col gap-40 w-full px-6 md:px-20 max-w-7xl mx-auto">
        <Grid>
          <GridRow>
            <GridItem className="py-10 pr-0 md:pr-10 gap-10">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Scalable Real-Time Serving</h3>
                <p className="text-white/60 text-lg">
                  MLflow provides a unified, scalable interface for deploying
                  models as REST APIs that automatically adjust to meet demand
                  fluctuations. With managed deployment on Databricks, your
                  endpoints can intelligently scale up or down based on traffic
                  patterns, optimizing both performance and infrastructure costs
                  with no manual intervention required.
                </p>
              </div>
              <FakeImage />
            </GridItem>
            <GridItem className="py-10 pl-0 md:pl-10 gap-10">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">High-Performance Batch Inference</h3>
                <p className="text-white/60 text-lg">
                  Deploy production models for batch inference directly on
                  Apache Spark, enabling efficient processing of billions of
                  predictions on massive datasets
                </p>
              </div>
              <FakeImage />
            </GridItem>
          </GridRow>
          <GridRow>
            <GridItem className="flex flex-col md:flex-row gap-6 md:gap-20 py-10 justify-between items-center">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Comprehensive Deployment Options</h3>
                <p className="text-white/60 text-lg">
                  Deploy models across multiple environments including Docker
                  containers, cloud services like Databricks, Azure ML and AWS
                  SageMaker, or Kubernetes clusters with consistent behavior.
                </p>
              </div>
              <FakeImage />
            </GridItem>
          </GridRow>
        </Grid>
        <GetStartedWithMLflow variant="blue" />
        <SocialWidget variant="red" />
      </div>
    </Layout>
  );
}
