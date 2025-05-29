import TraceTab from "@site/static/img/trace-tab.jpg";
import EvalsTab from "@site/static/img/evals-tab.jpg";
import AssessmentsTab from "@site/static/img/assessments.jpg";

import {
  Layout,
  CopyCommand,
  LogosCarousel,
  Grid,
  GridItem,
  LatestNews,
  AboveTheFold,
  BelowTheFold,
  Section,
  Card,
} from "../../components";

export default function GenAi(): JSX.Element {
  return (
    <Layout variant="red">
      <AboveTheFold
        title="Ship high-quality AI, fast"
        body={[
          "Traditional software and ML tests aren't built for GenAI's free-form language, making it difficult for teams to measure and improve quality.",
          "MLflow combines metrics that reliably measure GenAI quality with trace observability so you can measure, improve, and monitor quality, cost, and latency.",
        ]}
      >
        <CopyCommand code="pip install mlflow" />
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
              image={
                <img
                  className="rounded-xl"
                  src={TraceTab}
                  alt="MLflow tracing"
                />
              }
              // Product GIF of the tracing UI for a complex trace
            />
          </GridItem>
          <GridItem width="wide" direction="reverse">
            <Card
              title="Monitor in production"
              body={[
                "Maintain production quality with continuous monitoring of quality, latency, and cost. Gain real-time visibility via MLflow's dashboards and trace explorers.",
                "Configure automated online evaluations with alerts to quickly address issues.",
              ]}
              cta={{
                href: "/genai",
                text: "Learn more >",
              }}
              image={
                <img
                  className="rounded-xl mt-10 mb-4"
                  src={AssessmentsTab}
                  alt="MLflow Monitoring"
                />
              }
              // Product GIF of the Trace UI view + charts
            />
          </GridItem>
        </Grid>
      </Section>

      <Section
        label="Core features"
        title="Evaluation to measure and improve quality"
        body="MLflow simplifies GenAI evaluation, enabling easy collection and recording of LLM judge and human feedback directly on traces."
      >
        <Grid columns={2}>
          <GridItem width="wide">
            <Card
              title="Accurately measure free-form language with LLM judges"
              body="Utilize LLM-as-a-judge metrics, mimicking human expertise, to assess and enhance GenAI quality. Access pre-built judges for common metrics like hallucination or relevance, or develop custom judges tailored to your business needs and expert insights."
              cta={{
                href: "/genai",
                text: "Learn more >",
              }}
              image={
                <img
                  className="rounded-xl mt-10 mb-4"
                  src={AssessmentsTab}
                  alt="MLflow LLM judges"
                />
              }
              // Animation showing humans providing feedback and it being synthesized into an llm judge
            />
          </GridItem>
          <GridItem width="wide" direction="reverse">
            <Card
              title="Use production traffic to drive offline improvements"
              body="Adapt to user behavior by creating evaluation datasets and regression tests from production logs. Replay these to assess new prompts or app versions in development, ensuring optimal variants reach production."
              cta={{
                href: "/genai/evaluations",
                text: "Learn more >",
              }}
              image={
                <img
                  className="rounded-xl"
                  src={EvalsTab}
                  alt="MLflow evaluations"
                />
              }
              // Product GIF of running mlflow.evaluate and then seeing the eval results list UI
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Use human feedback to improve quality"
              body="Collect expert feedback through web UIs and end-user ratings from your app via APIs. Use this feedback to understand how your app should behave and align your custom LLM-judge metrics with expert judgement."
              cta={{
                href: "/genai/human-feedback",
                text: "Learn more >",
              }}
              image={
                <img
                  className="rounded-xl mt-10 mb-4"
                  src={AssessmentsTab}
                  alt="MLflow LLM judges"
                />
              }
              // Product GIF of the review app adding a custom feedback label and then it showing on the mlflow trace UI
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
                "Version, compare, iterate on, and discover prompt templates directly through the MLflow UI. Reuse prompts across multiple versions of your agent or application code, and view rich lineage identifying which versions are using each prompt.",
                "Evaluate and monitor prompt quality and performance across multiple versions.",
              ]}
              cta={{
                href: "/genai/governance",
                text: "Learn more >",
              }}
              image={
                <img
                  className="rounded-xl mt-10 mb-4"
                  src={AssessmentsTab}
                  alt="MLflow LLM judges"
                />
              }
              // Figma of the prompt registry, showing adding a prompt and then comparing 2 versions
            />
          </GridItem>
          <GridItem width="wide" direction="reverse">
            <Card
              title="Agent and application versioning"
              body={[
                "Version your agents, capturing their associated code, parameters, and evalation metrics for each iteration. MLflow's centralized management of agents complements Git, providing full lifecycle capabilities for all your generative AI assets.",
                "Evaluation and observability data are linked to specific agent/application versions, offering end-to-end versioning and lineage for your entire GenAI application.",
              ]}
              cta={{
                href: "/genai/governance",
                text: "Learn more >",
              }}
              image={
                <img
                  className="rounded-xl mt-10 mb-4"
                  src={AssessmentsTab}
                  alt="MLflow LLM judges"
                />
              }
              // Product GIF of the versions tab showing multiple models and then zooming into one to see evals + params like a prompt
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
          <GridItem>
            <Card
              title="Enterprise-Grade Security & Governance on a Unified Data & AI Platform"
              body="Managed MLflow on Databricks offers enterprise-grade security and deep Mosaic AI integrations for enhanced datasets, development, RAG, serving, and gateways. Unity Catalog ensures centralized governance over all AI assets."
            />
          </GridItem>
          <GridItem>
            <Card
              title="Unlock Downstream Value with Databricks AI/BI"
              body="Leverage your GenAI and ML data for downstream business processes by building rich performance dashboards, reports, and queries with Databricks AI/BI and Databricks SQL."
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
