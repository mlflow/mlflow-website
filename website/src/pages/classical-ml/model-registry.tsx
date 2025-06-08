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
import CardHero from "@site/static/img/Classical_registry/classical_registry_hero.png";
import Card1 from "@site/static/img/Classical_registry/classical_registry_1.png";
import Card2 from "@site/static/img/Classical_registry/classical_registry_2.png";
import Card3 from "@site/static/img/Classical_registry/classical_registry_3.png";

export default function ModelRegistryAndDeployment() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="Model registry & deployment"
        title="Deploy and manage models in production"
        body="Streamline your ML workflows with MLflow's comprehensive model registry for version control, approvals, and deployment management"
        hasGetStartedButton={MLFLOW_DOCS_URL}
      >
        <HeroImage src={CardHero} alt="" />
      </AboveTheFold>

      <Grid columns={2}>
        <GridItem width="wide">
          <Card
            title="Stage-based model lifecycle management"
            body="Move models through customizable staging environments (Development, Staging, Production, or any stage alias you choose) with built-in approval workflow capabilities and automated notifications. Maintain complete audit trails of model transitions with detailed metadata about who approved changes and when they occurred."
            image={<img src={Card2} alt="" />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Model deployment flexibility"
            body="Deploy models as Docker containers, Python functions, REST endpoints, or directly to various serving platforms with MLflow's versatile deployment capabilities. Streamline the transition from development to production with consistent model behavior across any target environment, from local testing to cloud-based serving."
            image={<img src={Card3} alt="" />}
          />
        </GridItem>
      </Grid>

      <BelowTheFold />
    </Layout>
  );
}
