import { MLFLOW_GET_STARTED_URL } from "@site/src/constants";
import {
  Layout,
  Grid,
  GridItem,
  AboveTheFold,
  BelowTheFold,
  Card,
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
        sectionLabel="Tracking"
        title="Comprehensive Experiment Tracking"
        body=" Document, compare, and reproduce your machine learning experiments with MLflow's powerful tracking capabilities"
        hasGetStartedButton={MLFLOW_GET_STARTED_URL}
      >
        <div className="w-full max-w-[800px] rounded-lg overflow-hidden mx-auto">
          <img src={CardHero} alt="" />
        </div>
      </AboveTheFold>

      <Grid columns={2}>
        <GridItem width="wide">
          <Card
            title="Visual Comparison"
            body="Compare results across multiple experiments with MLflow's powerful visualization tools. Quickly identify best-performing models and retrieve their corresponding code and parameters based on different metrics of interest across various projects."
            cta={{
              text: "Learn more >",
              href: "/",
            }}
            image={<img src={Card1} alt="" />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Reliable Reproducibility"
            body="Automatically logs parameters, weights, artifacts, code, metrics, and dependencies to ensure experiments can be restored accurately, enabling confident governance for enterprise deployments."
            cta={{
              text: "Learn more >",
              href: "/",
            }}
            image={<img src={Card2} alt="" />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Seamless Collaboration"
            body="Organize models and iterations into experiments for easy team collaboration while maintaining traceability. Enable team members to share results while maintaining a unified view of all projects through a single interface."
            cta={{
              text: "Learn more >",
              href: "/",
            }}
            image={<img src={Card3} alt="" />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Complete Experiment Lifecycle"
            body="MLflow Tracking automatically captures parameters, code versions, metrics, and model weights for each training iteration. Log trained models, visualizations, interface signatures, and data samples to ensure complete reproducibility across your entire ML workflow"
            cta={{
              text: "Learn more >",
              href: "/",
            }}
            image={<img src={Card4} alt="" />}
          />
        </GridItem>
      </Grid>

      <BelowTheFold />
    </Layout>
  );
}
