import Head from "@docusaurus/Head";
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

const SEO_TITLE = "ML Model Evaluation | MLflow AI Platform";
const SEO_DESCRIPTION =
  "Evaluate your ML models with confidence using MLflow. Automated evaluation tools for classification, regression, and other machine learning techniques.";

export default function ModelEvaluation() {
  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta
          property="og:url"
          content="https://mlflow.org/classical-ml/model-evaluation"
        />
        <link
          rel="canonical"
          href="https://mlflow.org/classical-ml/model-evaluation"
        />
        <script type="application/ld+json">
          {JSON.stringify({
            "@context": "https://schema.org",
            "@type": "SoftwareApplication",
            name: "MLflow",
            applicationCategory: "DeveloperApplication",
            operatingSystem: "Cross-platform",
            offers: {
              "@type": "Offer",
              price: "0",
              priceCurrency: "USD",
            },
            description: SEO_DESCRIPTION,
            url: "https://mlflow.org/classical-ml/model-evaluation",
            license: "https://www.apache.org/licenses/LICENSE-2.0",
          })}
        </script>
      </Head>
      <Layout>
        <AboveTheFold
          sectionLabel="Model evaluation"
          title="Evaluate models with confidence"
          body="Automated evaluation tools for foundational ML techniques like classification and regression."
          hasGetStartedButton="#get-started"
        >
          <HeroImage src={CardHero} alt="MLflow model evaluation screenshot" />
        </AboveTheFold>

        <StickyGrid
          cards={[
            {
              title: "Built-in metrics and visualizations",
              body: "MLflow automatically computes standard metrics and visualizations—such as ROC curves, precision-recall curves, confusion matrices, and regression diagnostics. These evaluation results are logged and surfaced directly in the MLflow UI, making it easy to explore, compare, and interpret model performance across runs.",
              image: (
                <img src={Card1} alt="MLflow model evaluation screenshot" />
              ),
            },
            {
              title: "Custom evaluators",
              body: "You can define your own evaluation logic using the custom evaluator interface. This is useful for model types or domains where standard metrics aren’t enough, such as specialized business KPIs or task-specific scoring.",
              image: (
                <img src={Card2} alt="MLflow model evaluation screenshot" />
              ),
            },
          ]}
        />

        <BelowTheFold contentType="classical-ml" />
      </Layout>
    </>
  );
}
