import { MLFLOW_GET_STARTED_URL } from "@site/src/constants";
import {
  Layout,
  Grid,
  GridItem,
  AboveTheFold,
  BelowTheFold,
  Card,
} from "../../components";
import CardHero from "@site/static/img/Classical_registry/classical_registry_hero.png";
import Card1 from "@site/static/img/Classical_registry/classical_registry_1.png";
import Card2 from "@site/static/img/Classical_registry/classical_registry_2.png";
import Card3 from "@site/static/img/Classical_registry/classical_registry_3.png";

export default function UnifiedRegistry() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="Model registry"
        title="Centralized Model Governance and Discovery"
        body="Streamline your ML workflows with MLflow's comprehensive model registry for version control, approvals, and deployment management"
        hasGetStartedButton={MLFLOW_GET_STARTED_URL}
      >
        <div className="w-full max-w-[800px] rounded-lg overflow-hidden mx-auto">
          <img src={CardHero} alt="" />
        </div>
      </AboveTheFold>

      <Grid columns={2}>
        <GridItem width="wide">
          <Card
            title="Seamless Unity Catalog Integration"
            body="MLflow Model Registry integrates directly with Unity Catalog to provide enterprise-grade governance across your entire ML asset portfolio. Apply consistent security policies, lineage tracking, and access controls to both data and models through a unified permission system."
            image={<img src={Card1} alt="" />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Stage-Based Model Lifecycle"
            body="Move models through customizable staging environments (Development, Staging, Production, or any stage alias you choose) with built-in approval workflow capabilities and automated notifications. Maintain complete audit trails of model transitions with detailed metadata about who approved changes and when they occurred."
            image={<img src={Card2} alt="" />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Model Deployment Flexibility"
            body="Deploy models as containers, batch jobs, or REST endpoints with MLflow's streamlined deployment capabilities that eliminate boilerplate code. Use model aliases to create named references that enable seamless model updates in production without changing your application code."
            image={<img src={Card3} alt="" />}
          />
        </GridItem>
      </Grid>

      <BelowTheFold />
    </Layout>
  );
}
