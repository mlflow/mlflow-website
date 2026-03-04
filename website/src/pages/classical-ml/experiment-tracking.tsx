import Head from "@docusaurus/Head";
import {
  Layout,
  AboveTheFold,
  BelowTheFold,
  HeroImage,
  StickyGrid,
} from "../../components";
import CardHero from "@site/static/img/Classical_tracking/classical_tracking_hero.png";
import Card1 from "@site/static/img/Classical_tracking/classical_tracking_1.png";
import Card2 from "@site/static/img/Classical_tracking/classical_tracking_2.png";
import Card3 from "@site/static/img/Classical_tracking/classical_tracking_3.png";
import Card4 from "@site/static/img/Classical_tracking/classical_tracking_4.png";
import Card5 from "@site/static/img/Classical_tracking/classical_tracking_5.png";
import Card6 from "@site/static/img/Classical_tracking/classical_tracking_6.png";

const SEO_TITLE = "ML Experiment Tracking | MLflow AI Platform";
const SEO_DESCRIPTION =
  "Track, compare, and reproduce your ML experiments with MLflow. Log parameters, metrics, and artifacts for machine learning and deep learning workflows.";

export default function Tracking() {
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
          content="https://mlflow.org/classical-ml/experiment-tracking"
        />
        <link
          rel="canonical"
          href="https://mlflow.org/classical-ml/experiment-tracking"
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
            url: "https://mlflow.org/classical-ml/experiment-tracking",
            license: "https://www.apache.org/licenses/LICENSE-2.0",
          })}
        </script>
      </Head>
      <Layout>
        <AboveTheFold
          sectionLabel="Experiment tracking"
          title="Comprehensive experiment tracking"
          body=" Track, compare, and reproduce your machine learning experiments with MLflow's powerful tracking capabilities."
          hasGetStartedButton="#get-started"
        >
          <HeroImage
            src={CardHero}
            alt="MLflow experiment tracking screenshot"
          />
        </AboveTheFold>

        <StickyGrid
          cards={[
            {
              title: "Visual comparison",
              body: "Compare results across multiple experiments with MLflow's powerful visualization tools. Quickly identify best-performing models and retrieve their corresponding code and parameters based on different metrics of interest across various projects.",
              image: (
                <img src={Card1} alt="MLflow experiment tracking screenshot" />
              ),
            },
            {
              title: "Native ML library integrations with mlflow.autolog",
              body: "mlflow.autolog() integrates with popular ML libraries to automatically log hyperparameters, metrics, and artifacts—enabling efficient tracking and comparison of tuning experiments with no manual effort.",
              image: (
                <img src={Card2} alt="MLflow experiment tracking screenshot" />
              ),
            },
            {
              title: "Reliable reproducibility",
              body: "Reliably logs parameters, weights, artifacts, code, metrics, and dependencies to ensure experiments can be restored accurately, enabling confident governance for enterprise deployments.",
              image: (
                <img src={Card3} alt="MLflow experiment tracking screenshot" />
              ),
            },
            {
              title: "Track hyperparameter tuning runs",
              body: "Leverage the native integration between MLflow and Optuna to run distributed hyperparameter optimization at scale using Spark UDFs. The MLflow tracking server provides robust trial data storage that persists through node failures, ensuring your optimization jobs complete successfully even in complex scalable distributed environments.",
              image: (
                <img src={Card4} alt="MLflow experiment tracking screenshot" />
              ),
            },
            {
              title: "Identify the best model for production",
              body: "By visualizing metrics across runs directly in the MLflow UI, users can quickly evaluate tradeoffs and identify the best model for production.",
              image: (
                <img src={Card5} alt="MLflow experiment tracking screenshot" />
              ),
            },
            {
              title: "Complete experiment lifecycle",
              body: "MLflow Tracking automatically captures parameters, code versions, metrics, and model weights for each training iteration. Log trained models, visualizations, interface signatures, and data samples to ensure complete reproducibility across your entire ML workflow.",
              image: (
                <img src={Card6} alt="MLflow experiment tracking screenshot" />
              ),
            },
          ]}
        />

        <BelowTheFold contentType="classical-ml" />
      </Layout>
    </>
  );
}
