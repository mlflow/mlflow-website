import {
  Layout,
  SectionLabel,
  Button,
  Grid,
  GridItem,
  GetStartedWithMLflow,
  SocialWidget,
  GetStartedButton,
  Heading,
} from "../../components";

const FakeImage = () => (
  <div className="w-full aspect-[3/2] bg-black rounded-lg border border-[rgba(255,255,255,0.08)]"></div>
);

export default function HyperparamTuning() {
  return (
    <Layout variant="blue">
      <div
        className="flex flex-col bg-[linear-gradient(to_top,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-3.png')]
 bg-center bg-no-repeat w-full pt-42 pb-20 py-20 bg-size-[100%_340px]"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-container">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <SectionLabel color="green" label="HYPERPARAM TUNING" />
            <Heading level={1}>Simplify your model training workflow</Heading>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-gray-600">
              Use state-of-the-art hyperparameter optimization techniques with
              an intuitive set of APIs
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
              <h3 className="text-white">Scalable HPO</h3>
              <p className="text-gray-600 text-lg">
                Leverage the native integration between MLflow and Optuna to run
                distributed hyperparameter optimization at scale using Spark
                UDFs. The MLflow tracking server provides robust trial data
                storage that persists through node failures, ensuring your
                optimization jobs complete successfully even in complex scalable
                distributed environments.
              </p>
            </div>
            <FakeImage />
          </GridItem>
          <GridItem>
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Native Tracking</h3>
              <p className="text-gray-600 text-lg">
                Every Optuna trial is automatically logged to MLflow, creating a
                comprehensive record of your hyperparameter search space and
                results. MLflow's intuitive UI enables teams to visualize
                parameter importance, correlation between hyperparameters and
                metrics, and identify promising regions in the search space
                without writing additional code.
              </p>
            </div>
            <FakeImage />
          </GridItem>
          <GridItem width="wide">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Enterprise-Ready</h3>
              <p className="text-gray-600 text-lg">
                Scale your hyperparameter optimization from development to
                production with MLflow's project packaging and model registry
                integration. Easily compare models across different optimization
                runs, promote the best performers to production, and maintain
                full lineage tracking from hyperparameter selection to deployed
                model.
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
