import { MLFLOW_GET_STARTED_URL } from "@site/src/constants";
import {
  Layout,
  Grid,
  GridItem,
  AboveTheFold,
  BelowTheFold,
  Card,
  HeroImage,
} from "../../components";
import CardHero from "@site/static/img/Classical_hyperparam/classical_hyperparam_hero.png";
import Card1 from "@site/static/img/Classical_hyperparam/classical_hyperparam_1.png";
import Card2 from "@site/static/img/Classical_hyperparam/classical_hyperparam_2.png";
import Card3 from "@site/static/img/Classical_hyperparam/classical_hyperparam_3.png";

export default function HyperparamTuning() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="Hyperparam tuning"
        title="Simplify your model training workflow"
        body="Use state-of-the-art hyperparameter optimization techniques with an intuitive set of APIs"
        hasGetStartedButton={MLFLOW_GET_STARTED_URL}
      >
        <HeroImage src={CardHero} alt="" />
      </AboveTheFold>

      <Grid columns={2}>
        <GridItem width="wide">
          <Card
            title="Native ML library integrations with mlflow.autolog"
            body="mlflow.autolog() integrates with popular ML libraries to automatically log hyperparameters, metrics, and artifacts—enabling efficient tracking and comparison of tuning experiments with no manual effort."
            image={<img src={Card1} alt="" />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Scalable Hyper Parameter Tuning"
            body="Leverage the native integration between MLflow and Optuna to run distributed hyperparameter optimization at scale using Spark UDFs. The MLflow tracking server provides robust trial data storage that persists through node failures, ensuring your optimization jobs complete successfully even in complex scalable distributed environments."
            image={<img src={Card2} alt="" />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Identify the best model for production"
            body="By visualizing metrics across runs directly in the MLflow UI, users can quickly evaluate tradeoffs and identify the best model for production."
            image={<img src={Card3} alt="" />}
          />
        </GridItem>
      </Grid>

      <BelowTheFold />
    </Layout>
  );
}
