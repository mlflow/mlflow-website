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
import { MLFLOW_DOCS_URL } from "@site/src/constants";
import Card1 from "@site/static/img/GenAI_home/GenAI_home_1.png";
import Card2 from "@site/static/img/GenAI_home/GenAI_home_2.png";
import Card3 from "@site/static/img/GenAI_home/GenAI_home_3.png";
import Card4 from "@site/static/img/GenAI_home/GenAI_home_4.png";
import Card5 from "@site/static/img/GenAI_home/GenAI_home_5.png";
import Card6 from "@site/static/img/GenAI_home/GenAI_home_6.png";
import Card7 from "@site/static/img/GenAI_home/GenAI_home_7.png";

export default function GenAi(): JSX.Element {
  return (
    <Layout>
      <AboveTheFold
        title="Ship high-quality GenAI, fast"
        body={[
          "Traditional software and ML tests aren’t built for GenAI’s free-form language, making it difficult for teams to measure and improve quality. "
          + "MLflow enhance your GenAI applications with end-to-end tracking, observability, evaluations and monitoring, all in one integrated platform.",
        ]}
        hasGetStartedButton={MLFLOW_DOCS_URL}
        bodyColor="white"
      >
        <div className="md:h-20 lg:h-40" />
      </AboveTheFold>

      <Section label="Core features" title="Observability to debug and monitor">
        <Grid columns={2}>
          <GridItem width="wide">
            <Card
              title="Debug with tracing"
              body={[
                "Debug and iterate on GenAI applications using MLflow's tracing, which captures your app's entire execution, including prompts, retrievals, tool calls.",
                "MLflow's open-source, OpenTelemetry-compatible tracing SDK helps avoid vendor lock-in.",
              ]}
              cta={{
                href: "/genai/observability",
                text: "Learn more >",
              }}
              image={<img src={Card1} alt="MLflow tracing" />}
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Monitor in production"
              body={[
                "Maintain production quality with continuous monitoring of quality, latency, and cost. Gain real-time visibility via MLflow's dashboards and trace explorers.",
                "Configure automated online evaluations with alerts to quickly address issues.",
              ]}
              cta={{
                href: "/genai/observability#production-monitoring",
                text: "Learn more >",
              }}
              image={<img src={Card2} alt="MLflow Monitoring" />}
            />
          </GridItem>
        </Grid>
      </Section>

      <Section
        label="Core features"
        title="Evaluation to measure and improve quality"
      >
        <Grid columns={2}>
          <GridItem width="wide">
            <Card
              title="Accurately measure free-form language with LLM judges"
              body="Utilize LLM-as-a-judge metrics, mimicking human expertise, to assess and enhance GenAI quality. Access pre-built judges for common metrics like hallucination or relevance, or develop custom judges tailored to your business needs and expert insights."
              cta={{
                href: "/genai/evaluations#quality-metrics",
                text: "Learn more >",
              }}
              image={<img src={Card3} alt="MLflow LLM judges" />}
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Use production traffic to drive offline improvements"
              body="Adapt to user behavior by creating evaluation datasets and regression tests from production logs. Replay these to assess new prompts or app versions in development, ensuring optimal variants reach production."
              cta={{
                href: "/genai/evaluations",
                text: "Learn more >",
              }}
              image={<img src={Card4} alt="MLflow evaluations" />}
            />
          </GridItem>
        </Grid>
      </Section>

      <Section
        label="Core features"
        title="Lifecycle management to track and version"
      >
        <Grid columns={2}>
          <GridItem width="wide">
            <Card
              title="Prompt registry"
              body={[
                "Version, compare, iterate on, and discover prompt templates directly through the MLflow UI. Reuse prompts across multiple versions of your agent or application code, and view rich lineage identifying which versions are using each prompt. "
                + "Evaluate and monitor prompt quality and performance across multiple versions.",
              ]}
              cta={{
                href: "/genai/governance",
                text: "Learn more >",
              }}
              image={<img src={Card6} alt="MLflow LLM judges" />}
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Agent and application versioning"
              body={[
                "Version your agents, capturing their associated code, parameters, and evalation metrics for each iteration. MLflow's centralized management of agents complements Git, providing full lifecycle capabilities for all your generative AI assets. "
                + "Evaluation and observability data are linked to specific agent/application versions, offering end-to-end versioning and lineage for your entire GenAI application.",
              ]}
              cta={{
                href: "/genai/governance",
                text: "Learn more >",
              }}
              image={<img src={Card7} alt="MLflow LLM judges" />}
            />
          </GridItem>
        </Grid>
      </Section>

      <LogosCarousel />

      <Section label="Why us?" title="Why MLflow is unique">
        <Grid columns={2}>
          <GridItem>
            <Card
              title="Unified, End-to-End MLOps and AI Observability"
              body="MLflow offers a unified platform for the entire GenAI and ML model lifecycle, simplifying the experience and boosting collaboration by reducing tool integration friction."
            />
          </GridItem>
          <GridItem>
            <Card
              title="Open, Flexible, and Extensible"
              body="Open-source and extensible, MLflow prevents vendor lock-in by integrating with the GenAI/ML ecosystem and using open protocols for data ownership, adapting to your existing and future stacks."
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
