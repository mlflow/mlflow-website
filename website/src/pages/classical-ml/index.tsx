import { MLFLOW_GET_STARTED_URL } from "@site/src/constants";
import {
  Layout,
  LogosCarousel,
  Grid,
  GridItem,
  LatestNews,
  AboveTheFold,
  BelowTheFold,
  Section,
  Card,
} from "../../components";
import Card1 from "@site/static/img/Classical_home/Classical_home_1.png";
import Card2 from "@site/static/img/Classical_home/Classical_home_2.png";
import Card3 from "@site/static/img/Classical_home/Classical_home_3.png";
import Card4 from "@site/static/img/Classical_home/Classical_home_4.png";
import Card5 from "@site/static/img/Classical_home/Classical_home_5.png";
import Card6 from "@site/static/img/Classical_home/Classical_home_6.png";

export default function GenAi(): JSX.Element {
  return (
    <Layout>
      <AboveTheFold
        title="Mastering the ML lifecycle"
        body="From experiment to production, MLflow streamlines your complete machine learning journey with enterprise-grade tracking, model management, and deployment."
        hasGetStartedButton={MLFLOW_GET_STARTED_URL}
        bodyColor="white"
      >
        <div className="md:h-20 lg:h-40" />
      </AboveTheFold>

      <Section
        label="Core features"
        title="Build confidently, deploy seamlessly"
        body="Cover experimentation, reproducibility, deployment, and a central model registry"
      >
        <Grid columns={2}>
          <GridItem width="wide">
            <Card
              title="Build production quality models"
              body="MLflow makes it easy to iterate toward production-ready models by organizing and comparing runs, helping teams refine training pipelines based on real performance insights."
              cta={{
                text: "Learn more >",
                href: "/",
              }}
              image={<img src={Card1} alt="" />}
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Framework neutral"
              body="Works seamlessly with popular tools like scikit-learn, PyTorch, TensorFlow, and XGBoost without vendor lock-in, providing flexibility with a common interface."
              cta={{
                text: "Learn more >",
                href: "/",
              }}
              image={<img src={Card2} alt="" />}
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
              image={<img src={Card3} alt="" />}
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Deployment Ready"
              body="Simplifies the path from experimentation to production with a built-in registry that gives you complete control over model states, whether sharing new approaches or deploying solutions."
              cta={{
                text: "Learn more >",
                href: "/",
              }}
              image={<img src={Card4} alt="" />}
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Unified Workflow"
              body="MLflow streamlines your entire ML process with tracking, packaging, and deployment capabilities, eliminating tool fragmentation so you can focus on model development rather than infrastructure"
              cta={{
                text: "Learn more >",
                href: "/",
              }}
              image={<img src={Card5} alt="" />}
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Enterprise Grade"
              body="Databricks-managed MLflow adds robust security, automated scaling, and high availability for mission-critical workloads while reducing operational overhead and delivering exceptional performance."
              cta={{
                text: "Learn more >",
                href: "/",
              }}
              image={<img src={Card6} alt="" />}
            />
          </GridItem>
        </Grid>
      </Section>

      <LogosCarousel />

      <Section label="Why us?" title="Why MLflow is unique">
        <Grid columns={2}>
          <GridItem>
            <Card
              title="Industry pioneer"
              bodySize="m"
              body="MLflow has established itself as a pioneering open-source platform for managing the end-to-end machine learning lifecycle. Created by Databricks, it has become one of the most widely adopted MLOps tools in the industry, with integration support from major cloud providers."
            />
          </GridItem>
          <GridItem>
            <Card
              title="Framework neutrality"
              bodySize="m"
              body="MLflow's framework-agnostic design is one of its strongest differentiators. Unlike proprietary solutions that lock you into specific ecosystems, MLflow works seamlessly with all popular ML frameworks including scikit-learn, PyTorch, TensorFlow, and XGBoost."
            />
          </GridItem>
          <GridItem>
            <Card
              title="Comprehensive Lifecycle Management"
              bodySize="m"
              body="MLflow uniquely addresses the complete machine learning lifecycle through four integrated components: - MLflow Tracking for logging parameters, metrics, and artifacts - MLflow Projects for reproducible code packaging - MLflow Models for standardized deployment - MLflow Model Registry for centralized version management"
            />
          </GridItem>
          <GridItem>
            <Card
              title="Enterprise Adoption"
              bodySize="m"
              body="MLflow's impact extends beyond its technical capabilities. It has gained significant traction among enterprise teams requiring robust experiment tracking and model lifecycle management. Databricks offers a managed MLflow service with enhanced security and scalability."
            />
          </GridItem>
        </Grid>
      </Section>

      <BelowTheFold>
        <LatestNews />
      </BelowTheFold>
    </Layout>
  );
}
