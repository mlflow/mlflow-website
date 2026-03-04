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

export default function HyperparamTuning() {
  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta property="og:title" content={SEO_TITLE} />
      </Head>
      <Layout>
        <AboveTheFold
          sectionLabel="Hyperparam tuning"
          title="Simplify your model training workflow"
          body="Use state-of-the-art hyperparameter optimization techniques with an intuitive set of APIs"
          hasGetStartedButton={MLFLOW_DOCS_URL}
        >
          <HeroImage src={CardHero} alt="" />
        </AboveTheFold>

        <StickyGrid
          cards={[
            {
              title: "Native ML library integrations with mlflow.autolog",
              body: "mlflow.autolog() integrates with popular ML libraries to automatically log hyperparameters, metrics, and artifacts—enabling efficient tracking and comparison of tuning experiments with no manual effort.",
              image: <img src={Card1} alt="" />,
            },
            {
              title: "Scalable Hyper Parameter Tuning",
              body: "Leverage the native integration between MLflow and Optuna to run distributed hyperparameter optimization at scale using Spark UDFs. The MLflow tracking server provides robust trial data storage that persists through node failures, ensuring your optimization jobs complete successfully even in complex scalable distributed environments.",
              image: <img src={Card2} alt="" />,
            },
            {
              title: "Identify the best model for production",
              body: "By visualizing metrics across runs directly in the MLflow UI, users can quickly evaluate tradeoffs and identify the best model for production.",
              image: <img src={Card3} alt="" />,
            },
          ]}
        />

        <BelowTheFold contentType="classical-ml" />
      </Layout>
    </>
  );
}
