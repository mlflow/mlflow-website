import {
  Layout,
  Grid,
  GridItem,
  Body,
  AboveTheFold,
  BelowTheFold,
} from "../../components";

const FakeImage = () => (
  <div className="w-full aspect-[3/2] bg-black rounded-lg border border-[rgba(255,255,255,0.08)]"></div>
);

export default function HyperparamTuning() {
  return (
    <Layout variant="blue" direction="up">
      <AboveTheFold
        sectionLabel="Hyperparam tuning"
        title="Simplify your model training workflow"
        body="Use state-of-the-art hyperparameter optimization techniques with an intuitive set of APIs"
        hasGetStartedButton
      >
        <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
      </AboveTheFold>

      <Grid columns={2}>
        <GridItem>
          <div className="flex flex-col gap-4">
            <h3 className="text-white">Scalable HPO</h3>
            <Body size="l">
              Leverage the native integration between MLflow and Optuna to run
              distributed hyperparameter optimization at scale using Spark UDFs.
              The MLflow tracking server provides robust trial data storage that
              persists through node failures, ensuring your optimization jobs
              complete successfully even in complex scalable distributed
              environments.
            </Body>
          </div>
          <FakeImage />
        </GridItem>
        <GridItem>
          <div className="flex flex-col gap-4">
            <h3 className="text-white">Native Tracking</h3>
            <Body size="l">
              Every Optuna trial is automatically logged to MLflow, creating a
              comprehensive record of your hyperparameter search space and
              results. MLflow's intuitive UI enables teams to visualize
              parameter importance, correlation between hyperparameters and
              metrics, and identify promising regions in the search space
              without writing additional code.
            </Body>
          </div>
          <FakeImage />
        </GridItem>
        <GridItem width="wide">
          <div className="flex flex-col gap-4">
            <h3 className="text-white">Enterprise-Ready</h3>
            <Body size="l">
              Scale your hyperparameter optimization from development to
              production with MLflow's project packaging and model registry
              integration. Easily compare models across different optimization
              runs, promote the best performers to production, and maintain full
              lineage tracking from hyperparameter selection to deployed model.
            </Body>
          </div>
          <FakeImage />
        </GridItem>
      </Grid>

      <BelowTheFold />
    </Layout>
  );
}
