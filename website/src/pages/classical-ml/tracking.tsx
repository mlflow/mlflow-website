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

export default function Tracking() {
  return (
    <Layout variant="blue">
      <div
        className="flex flex-col bg-[linear-gradient(to_top,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-3.png')]
 bg-center bg-no-repeat bg-cover w-full pt-42 pb-20 py-20 bg-size-[100%_340px]"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-7xl mx-auto">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <SectionLabel color="green" label="TRACKING" />
            <h1 className="text-center text-wrap">
              Comprehensive Experiment Tracking for ML Excellence
            </h1>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white/60">
              Document, compare, and reproduce your machine learning experiments
              with MLflow's powerful tracking capabilities
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
                <h3 className="text-white">Complete Experiment Lifecycle</h3>
                <p className="text-white/60 text-lg">
                  MLflow Tracking automatically captures parameters, code
                  versions, metrics, and model weights for each training
                  iteration. Log trained models, visualizations, interface
                  signatures, and data samples to ensure complete
                  reproducibility across your entire ML workflow
                </p>
              </div>
              <FakeImage />
            </GridItem>
            <GridItem className="py-10 pl-0 md:pl-10 gap-10">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Visual Comparison</h3>
                <p className="text-white/60 text-lg">
                  Compare results across multiple experiments with MLflow's
                  powerful visualization tools. Quickly identify best-performing
                  models and retrieve their corresponding code and parameters
                  based on different metrics of interest across various
                  projects.
                </p>
              </div>
              <FakeImage />
            </GridItem>
          </GridRow>
          <GridRow>
            <GridItem className="flex flex-col md:flex-row gap-6 md:gap-20 py-10 justify-between items-center">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Seamless Collaboration</h3>
                <p className="text-white/60 text-lg">
                  Organize models and iterations into experiments for easy team
                  collaboration while maintaining traceability. Enable team
                  members to share results while maintaining a unified view of
                  all projects through a single interface.
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
