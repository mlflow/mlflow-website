import { MLFLOW_DOCS_URL } from "@site/src/constants";
import {
  Layout,
  Grid,
  GridItem,
  AboveTheFold,
  BelowTheFold,
  Card,
  HeroImage,
} from "../../components";
import CardHero from "@site/static/img/Classical_tracking/classical_tracking_hero.png";
import Card1 from "@site/static/img/Classical_tracking/classical_tracking_1.png";
import Card2 from "@site/static/img/Classical_tracking/classical_tracking_2.png";
import Card3 from "@site/static/img/Classical_tracking/classical_tracking_3.png";
import Card4 from "@site/static/img/Classical_tracking/classical_tracking_4.png";

export default function Tracking() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="Experiment tracking"
        title="Comprehensive experiment tracking"
        body=" Track, compare, and reproduce your machine learning experiments with MLflow's powerful tracking capabilities."
        hasGetStartedButton={MLFLOW_DOCS_URL}
      >
        <HeroImage src={CardHero} alt="" />
      </AboveTheFold>

      <Grid columns={2}>
        <GridItem width="wide">
          <Card
            title="Visual comparison"
            body="Compare results across multiple experiments with MLflow's powerful visualization tools. Quickly identify best-performing models and retrieve their corresponding code and parameters based on different metrics of interest across various projects."
            image={<img src={Card1} alt="" />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Native ML library integrations with mlflow.autolog"
            body="mlflow.autolog() integrates with popular ML libraries to automatically log hyperparameters, metrics, and artifacts—enabling efficient tracking and comparison of tuning experiments with no manual effort."
            image={<img src={Card1} alt="" />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Reliable reproducibility"
            body="Reliably logs parameters, weights, artifacts, code, metrics, and dependencies to ensure experiments can be restored accurately, enabling confident governance for enterprise deployments."
            image={<img src={Card2} alt="" />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Track hyperparameter tuning runs"
            body="Leverage the native integration between MLflow and Optuna to run distributed hyperparameter optimization at scale using Spark UDFs. The MLflow tracking server provides robust trial data storage that persists through node failures, ensuring your optimization jobs complete successfully even in complex scalable distributed environments."
            image={<img src={Card3} alt="" />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Identify the best model for production"
            body="By visualizing metrics across runs directly in the MLflow UI, users can quickly evaluate tradeoffs and identify the best model for production."
            image={<img src={Card3} alt="" />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Complete experiment lifecycle"
            body="MLflow Tracking automatically captures parameters, code versions, metrics, and model weights for each training iteration. Log trained models, visualizations, interface signatures, and data samples to ensure complete reproducibility across your entire ML workflow."
            image={<img src={Card4} alt="" />}
          />
        </GridItem>
      </Grid>

      <BelowTheFold />
    </Layout>
  );
}
