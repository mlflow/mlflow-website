import {
  Layout,
  SectionLabel,
  Button,
  Grid,
  GridItem,
  GetStartedWithMLflow,
  SocialWidget,
  GetStartedButton,
} from "../../components";

const FakeImage = () => (
  <div className="w-full aspect-[3/2] bg-black rounded-lg border border-[rgba(255,255,255,0.08)]"></div>
);

export default function Models() {
  return (
    <Layout variant="blue">
      <div
        className="flex flex-col bg-[linear-gradient(to_top,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-3.png')]
 bg-center bg-no-repeat w-full pt-42 pb-20 py-20 bg-size-[100%_340px]"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-container">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <SectionLabel color="green" label="MODELS" />
            <h1 className="text-center text-wrap">
              Transform ML Experiments into Production-Ready Models
            </h1>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-gray-600">
              Build, deploy, and manage machine learning models with confidence
              using MLflow's comprehensive model lifecycle management
            </p>
            <GetStartedButton />
          </div>
          <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
        </div>
      </div>
      <div className="flex flex-col gap-40 w-full px-6 md:px-20 max-w-container">
        <Grid columns={2}>
          <GridItem>
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Unified Model Format</h3>
              <p className="text-gray-600 text-lg">
                MLflow's MLModel file provides a standardized structure for
                packaging models from any framework, capturing essential
                dependencies and input/output specifications. This consistent
                packaging approach eliminates integration friction while
                ensuring models can be reliably deployed across any environment.
              </p>
            </div>
            <FakeImage />
          </GridItem>
          <GridItem>
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Comprehensive Model Metadata</h3>
              <p className="text-gray-600 text-lg">
                Track crucial model requirements and artifacts including data
                schemas, preprocessing steps, and environment dependencies
                automatically with MLflow's metadata system. Create fully
                reproducible model packages that document the complete model
                context for simplified governance and troubleshooting.
              </p>
            </div>
            <FakeImage />
          </GridItem>
          <GridItem width="wide">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Flexible Deployment Options</h3>
              <p className="text-gray-600 text-lg">
                Deploy models as Docker containers, Python functions, REST
                endpoints, or directly to various serving platforms with
                MLflow's versatile deployment capabilities. Streamline the
                transition from development to production with consistent model
                behavior across any target environment, from local testing to
                cloud-based serving.
              </p>
            </div>
            <FakeImage />
          </GridItem>
        </Grid>
        <GetStartedWithMLflow variant="blue" />
        <SocialWidget variant="green" />
      </div>
    </Layout>
  );
}
