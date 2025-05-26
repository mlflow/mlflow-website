import {
  Layout,
  SectionLabel,
  Grid,
  GridItem,
  GetStartedWithMLflow,
  SocialWidget,
  GetStartedButton,
  Heading,
  Body,
} from "../../components";

const FakeImage = () => (
  <div className="w-full aspect-[3/2] bg-black rounded-lg border border-[rgba(255,255,255,0.08)]"></div>
);

export default function Tracking() {
  return (
    <Layout variant="blue">
      <div
        className="flex flex-col bg-[linear-gradient(to_top,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-3.png')]
 bg-center bg-no-repeat w-full pt-42 pb-20 py-20 bg-size-[100%_340px]"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-container">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <SectionLabel color="green" label="TRACKING" />
            <Heading level={1}>
              Comprehensive Experiment Tracking for ML Excellence
            </Heading>
            <div className="max-w-3xl mx-auto text-center">
              <Body size="l">
                Document, compare, and reproduce your machine learning
                experiments with MLflow's powerful tracking capabilities
              </Body>
            </div>
            <GetStartedButton />
          </div>
          <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
        </div>
      </div>
      <div className="flex flex-col gap-40 w-full px-6 md:px-20 max-w-container">
        <Grid columns={2}>
          <GridItem>
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Complete Experiment Lifecycle</h3>
              <Body size="l">
                MLflow Tracking automatically captures parameters, code
                versions, metrics, and model weights for each training
                iteration. Log trained models, visualizations, interface
                signatures, and data samples to ensure complete reproducibility
                across your entire ML workflow
              </Body>
            </div>
            <FakeImage />
          </GridItem>
          <GridItem>
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Visual Comparison</h3>
              <Body size="l">
                Compare results across multiple experiments with MLflow's
                powerful visualization tools. Quickly identify best-performing
                models and retrieve their corresponding code and parameters
                based on different metrics of interest across various projects.
              </Body>
            </div>
            <FakeImage />
          </GridItem>
          <GridItem width="wide">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Seamless Collaboration</h3>
              <Body size="l">
                Organize models and iterations into experiments for easy team
                collaboration while maintaining traceability. Enable team
                members to share results while maintaining a unified view of all
                projects through a single interface.
              </Body>
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
