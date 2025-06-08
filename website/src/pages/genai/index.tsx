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
          "Enhance your GenAI application with end-to-end tracking, observability, evaluations, all in one integrated platform.",
        ]}
        hasGetStartedButton={MLFLOW_DOCS_URL}
        bodyColor="white"
      >
        <div className="md:h-40 lg:h-80" />
      </AboveTheFold>

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
            title="Accurately measure free-form language with LLM judges"
            body="Utilize LLM-as-a-judge metrics, mimicking human expertise, to assess and enhance GenAI quality. Access pre-built judges for common metrics like hallucination or relevance, or develop custom judges tailored to your business needs and expert insights."
            cta={{
              href: "/genai/evaluations",
              text: "Learn more >",
            }}
            image={<img src={Card3} alt="MLflow LLM judges" />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Prompt Registry"
            body={[
              "Version, compare, iterate on, and discover prompt templates directly through the MLflow UI. Reuse prompts across multiple versions of your agent or application code, and view rich lineage identifying which versions are using each prompt.",
            ]}
            cta={{
              href: "/genai/prompt-registry",
              text: "Learn more >",
            }}
            image={<img src={Card6} alt="MLflow LLM judges" />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Agent and application versioning"
            body={[
              "Version your agents, capturing their associated code, parameters, and evalation metrics for each iteration. MLflow's centralized management of agents complements Git, providing full lifecycle capabilities for all your generative AI assets.",
            ]}
            cta={{
              href: "/genai/app-versioning",
              text: "Learn more >",
            }}
            image={<img src={Card7} alt="MLflow LLM judges" />}
          />
        </GridItem>
      </Grid>

      <LogosCarousel />
      <ValuePropWidget />
      <BelowTheFold>
        <LatestNews />
      </BelowTheFold>
    </Layout>
  );
}
