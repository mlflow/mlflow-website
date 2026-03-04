import { MLFLOW_DOCS_URL } from "@site/src/constants";
import Head from "@docusaurus/Head";
import {
  Layout,
  AboveTheFold,
  BelowTheFold,
  HeroImage,
  StickyGrid,
} from "../../components";
import CardHero from "@site/static/img/Classical_serving/classical_serving_hero.png";
import Card1 from "@site/static/img/Classical_serving/classical_serving_1.png";
import Card2 from "@site/static/img/Classical_serving/classical_serving_2.png";
import Card3 from "@site/static/img/Classical_serving/classical_serving_3.png";

const SEO_TITLE = "ML Model Serving | MLflow AI Platform";
const SEO_DESCRIPTION =
  "Deploy and serve your ML models with MLflow. Flexible serving options for real-time and batch inference across machine learning and deep learning models.";

export default function Serving() {
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
          content="https://mlflow.org/classical-ml/serving"
        />
        <link rel="canonical" href="https://mlflow.org/classical-ml/serving" />
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
            url: "https://mlflow.org/classical-ml/serving",
            license: "https://www.apache.org/licenses/LICENSE-2.0",
          })}
        </script>
      </Head>
      <Layout>
        <AboveTheFold
          sectionLabel="Serving"
          title="Flexible Model Deployment"
          body="Deploy your ML and DL models with confidence using MLflow's versatile serving options for real-time and batch inference"
          hasGetStartedButton={MLFLOW_DOCS_URL}
        >
          <HeroImage src={CardHero} alt="MLflow model serving screenshot" />
        </AboveTheFold>

        <StickyGrid
          cards={[
            {
              title: "Scalable Real-Time Serving",
              body: "MLflow provides a unified, scalable interface for deploying models as REST APIs that automatically adjust to meet demand fluctuations. With managed deployment on Databricks, your endpoints can intelligently scale up or down based on traffic patterns, optimizing both performance and infrastructure costs with no manual intervention required.",
              image: <img src={Card1} alt="MLflow model serving screenshot" />,
            },
            {
              title: "High-Performance Batch Inference",
              body: "Deploy production models for batch inference directly on Apache Spark, enabling efficient processing of billions of predictions on massive datasets",
              image: <img src={Card2} alt="MLflow model serving screenshot" />,
            },
            {
              title: "Comprehensive Deployment Options",
              body: "Deploy models across multiple environments including Docker containers, cloud services like Databricks, Azure ML and AWS SageMaker, or Kubernetes clusters with consistent behavior.",
              image: <img src={Card3} alt="MLflow model serving screenshot" />,
            },
          ]}
        />

        <BelowTheFold contentType="classical-ml" />
      </Layout>
    </>
  );
}
