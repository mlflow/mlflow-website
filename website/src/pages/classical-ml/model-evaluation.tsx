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
import CardHero from "@site/static/img/Classical_serving/classical_serving_hero.png";
import Card1 from "@site/static/img/Classical_serving/classical_serving_1.png";
import Card2 from "@site/static/img/Classical_serving/classical_serving_2.png";
import Card3 from "@site/static/img/Classical_serving/classical_serving_3.png";

export default function ModelEvaluation() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="Model evaluation"
        title="Evaluate models with confidence"
        body="Traditional machine learning techniques like classification and regression remain foundational across industries. MLflow provides automated evaluation tools designed specifically for these classic workflows."
        hasGetStartedButton={MLFLOW_DOCS_URL}
      >
        <HeroImage src={CardHero} alt="" />
      </AboveTheFold>

      <Grid columns={2}>
        <GridItem width="wide">
          <Card
            title="Built-in metrics and visualizations"
            body="MLflow automatically computes standard metrics and visualizations—such as ROC curves, precision-recall curves, confusion matrices, and regression diagnostics. These evaluation results are logged and surfaced directly in the MLflow UI, making it easy to explore, compare, and interpret model performance across runs."
            image={<img src={Card1} alt="" />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Custom evaluators"
            body="You can define your own evaluation logic using the custom evaluator interface. This is useful for model types or domains where standard metrics aren’t enough, such as specialized business KPIs or task-specific scoring."
            image={<img src={Card2} alt="" />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Batch evaluation across multiple models"
            body="Evaluate multiple models or runs in a single mlflow.evaluate() call using consistent datasets and metrics. This batch evaluation makes side-by-side performance comparison fast and efficient."
            image={<img src={Card3} alt="" />}
          />
        </GridItem>
      </Grid>

      <BelowTheFold />
    </Layout>
  );
}
