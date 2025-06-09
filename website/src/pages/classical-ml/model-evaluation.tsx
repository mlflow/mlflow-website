import { MLFLOW_DOCS_URL } from "@site/src/constants";
import {
  Layout,
  AboveTheFold,
  BelowTheFold,
  HeroImage,
  StickyGrid,
} from "../../components";
import CardHero from "@site/static/img/Classical_evaluations/classical_evaluation_hero.png";
import Card1 from "@site/static/img/Classical_evaluations/classical_evaluation_1.png";
import Card2 from "@site/static/img/Classical_evaluations/classical_evaluation_2.png";

export default function ModelEvaluation() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="Model evaluation"
        title="Evaluate models with confidence"
        body="Automated evaluation tools for foundational ML techniques like classification and regression."
        hasGetStartedButton={MLFLOW_DOCS_URL}
      >
        <HeroImage src={CardHero} alt="" />
      </AboveTheFold>

      <StickyGrid
        cards={[
          {
            title: "Built-in metrics and visualizations",
            body: "MLflow automatically computes standard metrics and visualizations—such as ROC curves, precision-recall curves, confusion matrices, and regression diagnostics. These evaluation results are logged and surfaced directly in the MLflow UI, making it easy to explore, compare, and interpret model performance across runs.",
            image: <img src={Card1} alt="" />,
          },
          {
            title: "Custom evaluators",
            body: "You can define your own evaluation logic using the custom evaluator interface. This is useful for model types or domains where standard metrics aren’t enough, such as specialized business KPIs or task-specific scoring.",
            image: <img src={Card2} alt="" />,
          },
        ]}
      />

      <BelowTheFold />
    </Layout>
  );
}
