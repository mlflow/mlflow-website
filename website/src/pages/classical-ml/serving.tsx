import { MLFLOW_DOCS_URL } from "@site/src/constants";
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

export default function Serving() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="Serving"
        title="Flexible Model Deployment"
        body="Deploy your ML and DL models with confidence using MLflow's versatile serving options for real-time and batch inference"
        hasGetStartedButton={MLFLOW_DOCS_URL}
      >
        <HeroImage src={CardHero} alt="" />
      </AboveTheFold>

      <StickyGrid
        cards={[
          {
            title: "Scalable Real-Time Serving",
            body: "MLflow provides a unified, scalable interface for deploying models as REST APIs that automatically adjust to meet demand fluctuations. With managed deployment on Databricks, your endpoints can intelligently scale up or down based on traffic patterns, optimizing both performance and infrastructure costs with no manual intervention required.",
            image: <img src={Card1} alt="" />,
          },
          {
            title: "High-Performance Batch Inference",
            body: "Deploy production models for batch inference directly on Apache Spark, enabling efficient processing of billions of predictions on massive datasets",
            image: <img src={Card2} alt="" />,
          },
          {
            title: "Comprehensive Deployment Options",
            body: "Deploy models across multiple environments including Docker containers, cloud services like Databricks, Azure ML and AWS SageMaker, or Kubernetes clusters with consistent behavior.",
            image: <img src={Card3} alt="" />,
          },
        ]}
      />

      <BelowTheFold />
    </Layout>
  );
}
