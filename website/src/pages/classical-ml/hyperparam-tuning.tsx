import { MLFLOW_DOCS_URL } from "@site/src/constants";
import Head from "@docusaurus/Head";
import {
  Layout,
  AboveTheFold,
  BelowTheFold,
  HeroImage,
  StickyGrid,
} from "../../components";
import CardHero from "@site/static/img/Classical_hyperparam/classical_hyperparam_hero.png";
import Card1 from "@site/static/img/Classical_hyperparam/classical_hyperparam_1.png";
import Card2 from "@site/static/img/Classical_hyperparam/classical_hyperparam_2.png";
import Card3 from "@site/static/img/Classical_hyperparam/classical_hyperparam_3.png";

const SEO_TITLE = "ML Hyperparameter Tuning | MLflow AI Platform";
const SEO_DESCRIPTION =
  "Optimize your ML models with MLflow's hyperparameter tuning. Use state-of-the-art optimization techniques for machine learning and deep learning model training.";

export default function HyperparamTuning() {
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
          content="https://mlflow.org/classical-ml/hyperparam-tuning"
        />
        <link
          rel="canonical"
          href="https://mlflow.org/classical-ml/hyperparam-tuning"
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
            url: "https://mlflow.org/classical-ml/hyperparam-tuning",
            license: "https://www.apache.org/licenses/LICENSE-2.0",
          })}
        </script>
      </Head>
      <Layout>
        <AboveTheFold
          sectionLabel="Hyperparam tuning"
          title="Simplify your model training workflow"
          body="Use state-of-the-art hyperparameter optimization techniques with an intuitive set of APIs"
          hasGetStartedButton={MLFLOW_DOCS_URL}
        >
          <HeroImage
            src={CardHero}
            alt="MLflow hyperparameter tuning screenshot"
          />
        </AboveTheFold>

        <StickyGrid
          cards={[
            {
              title: "Native ML library integrations with mlflow.autolog",
              body: "mlflow.autolog() integrates with popular ML libraries to automatically log hyperparameters, metrics, and artifacts—enabling efficient tracking and comparison of tuning experiments with no manual effort.",
              image: (
                <img
                  src={Card1}
                  alt="MLflow hyperparameter tuning screenshot"
                />
              ),
            },
            {
              title: "Scalable Hyper Parameter Tuning",
              body: "Leverage the native integration between MLflow and Optuna to run distributed hyperparameter optimization at scale using Spark UDFs. The MLflow tracking server provides robust trial data storage that persists through node failures, ensuring your optimization jobs complete successfully even in complex scalable distributed environments.",
              image: (
                <img
                  src={Card2}
                  alt="MLflow hyperparameter tuning screenshot"
                />
              ),
            },
            {
              title: "Identify the best model for production",
              body: "By visualizing metrics across runs directly in the MLflow UI, users can quickly evaluate tradeoffs and identify the best model for production.",
              image: (
                <img
                  src={Card3}
                  alt="MLflow hyperparameter tuning screenshot"
                />
              ),
            },
          ]}
        />

        <BelowTheFold contentType="classical-ml" />
      </Layout>
    </>
  );
}
