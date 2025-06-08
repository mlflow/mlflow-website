import { MLFLOW_DOCS_URL } from "@site/src/constants";
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
  ValuePropWidget,
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
        body="From experiment to production, MLflow streamlines your complete machine learning journey with end-to-end tracking, model management, and deployment."
        hasGetStartedButton={MLFLOW_DOCS_URL}
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
              title="Reliable reproducibility"
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
              title="Deployment ready"
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
              title="Unified workflow"
              body="MLflow streamlines your entire ML process with tracking, packaging, and deployment capabilities, eliminating tool fragmentation so you can focus on model development rather than infrastructure"
              cta={{
                text: "Learn more >",
                href: "/",
              }}
              image={<img src={Card5} alt="" />}
            />
          </GridItem>
        </Grid>
      </Section>

      <LogosCarousel />
      <ValuePropWidget />

      <BelowTheFold>
        <LatestNews />
      </BelowTheFold>
    </Layout>
  );
}
