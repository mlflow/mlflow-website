import TraceTab from "@site/static/img/trace-tab.jpg";
import EvalsTab from "@site/static/img/evals-tab.jpg";
import AssessmentsTab from "@site/static/img/assessments.jpg";

import {
  Layout,
  CopyCommand,
  SectionLabel,
  LogosCarousel,
  Grid,
  GridItem,
  GetStartedWithMLflow,
  LatestNews,
  SocialWidget,
  Button,
  Heading,
  Body,
} from "../../components";
import Link from "@docusaurus/Link";

export default function GenAi(): JSX.Element {
  return (
    <Layout variant="red">
      <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-container">
        <div className="flex flex-col justify-center items-center gap-6 w-full">
          <Heading level={1}>Ship high-quality AI, fast</Heading>
          <div className="max-w-3xl mx-auto text-center">
            <Body size="l">
              Traditional software and ML tests aren't built for GenAI's
              free-form language, making it difficult for teams to measure and
              improve quality.
            </Body>
            <Body size="l">
              MLflow combines metrics that reliably measure GenAI quality with
              trace observability so you can measure, improve, and monitor
              quality, cost, and latency.
            </Body>
          </div>
        </div>
        <div className="flex flex-col md:flex-row gap-10">
          <CopyCommand code="pip install mlflow" />
        </div>
      </div>
      <div className="flex flex-col gap-40 w-full px-6 md:px-20 max-w-container">
        <div className="flex flex-col w-full items-center justify-center gap-16">
          <div className="flex flex-col w-full items-center justify-center gap-6">
            <SectionLabel color="red" label="CORE FEATURES" />
            <Heading level={2}>Observability to debug and monitor</Heading>
          </div>
          <Grid columns={2}>
            <GridItem width="wide">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Debug with tracing</h3>
                <Body size="l">
                  Debug and iterate on GenAI applications using MLflow's
                  tracing, which captures your app's entire execution, including
                  prompts, retrievals, tool calls.
                </Body>
                <Body size="l">
                  MLflow's open-source, OpenTelemetry-compatible tracing SDK
                  helps avoid vendor lock-in.
                </Body>

                <Link href="/genai/observability">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </Link>
              </div>
              <div>
                <img
                  className="rounded-xl"
                  src={TraceTab}
                  alt="MLflow tracing"
                />
                {/* Product GIF of the tracing UI for a complex trace */}
              </div>
            </GridItem>
            <GridItem width="wide" direction="reverse">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Monitor in production</h3>
                <Body size="l">
                  Maintain production quality with continuous monitoring of
                  quality, latency, and cost. Gain real-time visibility via
                  MLflow's dashboards and trace explorers.
                </Body>
                <Body size="l">
                  Configure automated online evaluations with alerts to quickly
                  address issues.
                </Body>
                <Link href="/genai/monitoring">
                  {" "}
                  {/* Placeholder link */}
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </Link>
              </div>
              <div>
                <img
                  className="rounded-xl mt-10 mb-4"
                  src={AssessmentsTab} /* Placeholder image */
                  alt="MLflow Monitoring"
                />
              </div>
              {/* Product GIF of the Trace UI view + charts */}
            </GridItem>
          </Grid>
        </div>
        <div className="flex flex-col w-full items-center justify-center gap-16">
          <div className="flex flex-col w-full items-center justify-center gap-6">
            <SectionLabel color="red" label="CORE FEATURES" />
            <Heading level={2}>
              Evaluation to measure and improve quality
            </Heading>
            <Body size="l">
              MLflow simplifies GenAI evaluation, enabling easy collection and
              recording of LLM judge and human feedback directly on traces.
            </Body>
            {/* <p className="text-white">
              Tackle the challenges of building GenAI head on
            </p> */}
          </div>
          <Grid columns={2}>
            <GridItem width="wide">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  Accurately measure free-form language with LLM judges
                </h3>
                <Body size="l">
                  Utilize LLM-as-a-judge metrics, mimicking human expertise, to
                  assess and enhance GenAI quality. Access pre-built judges for
                  common metrics like hallucination or relevance, or develop
                  custom judges tailored to your business needs and expert
                  insights.
                </Body>
                <Link href="/genai/quality-metrics">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </Link>
              </div>
              <div>
                <img
                  className="rounded-xl mt-10 mb-4"
                  src={AssessmentsTab}
                  alt="MLflow LLM judges"
                />
              </div>
              {/* Animation showing humans providing feedback and it being synthesized into an llm judge */}
            </GridItem>

            <GridItem width="wide" direction="reverse">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  Use production traffic to drive offline improvements
                </h3>
                <Body size="l">
                  Adapt to user behavior by creating evaluation datasets and
                  regression tests from production logs. Replay these to assess
                  new prompts or app versions in development, ensuring optimal
                  variants reach production.
                </Body>
                <Link href="/genai/evaluations">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </Link>
              </div>
              <div>
                <img
                  className="rounded-xl"
                  src={EvalsTab}
                  alt="MLflow evaluations"
                />
              </div>
              {/* Product GIF of running mlflow.evaluate and then seeing the eval results list UI */}
            </GridItem>

            <GridItem width="wide">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  Use human feedback to improve quality
                </h3>
                <Body size="l">
                  Collect expert feedback through web UIs and end-user ratings
                  from your app via APIs. Use this feedback to understand how
                  your app should behave and align your custom LLM-judge metrics
                  with expert judgement.
                </Body>
                <Link href="/genai/human-feedback">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </Link>
              </div>
              <div>
                <img
                  className="rounded-xl mt-10 mb-4"
                  src={AssessmentsTab}
                  alt="MLflow LLM judges"
                />
              </div>
              {/* Product GIF of the review app adding a custom feedback label and then it showing on the mlflow trace UI */}
            </GridItem>
          </Grid>
        </div>
        <div className="flex flex-col w-full items-center justify-center gap-16">
          <div className="flex flex-col w-full items-center justify-center gap-6">
            <SectionLabel color="red" label="CORE FEATURES" />
            <Heading level={2}>
              Lifecycle management to track and version
            </Heading>
            {/* <p className="text-white">
              Tackle the challenges of building GenAI head on
            </p> */}
          </div>
          <Grid columns={2}>
            <GridItem width="wide">
              <div className="flex flex-col gap-4 justify-center">
                <h3 className="text-white">Prompt registry</h3>
                <Body size="l">
                  Version, compare, iterate on, and discover prompt templates
                  directly through the MLflow UI. Reuse prompts across multiple
                  versions of your agent or application code, and view rich
                  lineage identifying which versions are using each prompt.
                </Body>
                <Body size="l">
                  Evaluate and monitor prompt quality and performance across
                  multiple versions.
                </Body>
                <Link href="/genai/governance">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </Link>
              </div>
              <div>
                <img
                  className="rounded-xl mt-10 mb-4"
                  src={AssessmentsTab}
                  alt="MLflow LLM judges"
                />
              </div>
              {/* Figma of the prompt registry, showing adding a prompt and then comparing 2 versions */}
            </GridItem>

            <GridItem width="wide" direction="reverse">
              <div className="flex flex-col gap-4 justify-center">
                <h3 className="text-white">Agent and application versioning</h3>
                <Body size="l">
                  Version your agents, capturing their associated code,
                  parameters, and evalation metrics for each iteration. MLflow's
                  centralized management of agents complements Git, providing
                  full lifecycle capabilities for all your generative AI assets.
                </Body>
                <Body size="l">
                  Evaluation and observability data are linked to specific
                  agent/application versions, offering end-to-end versioning and
                  lineage for your entire GenAI application.
                </Body>
                <Link href="/genai/governance">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </Link>
              </div>
              <div>
                <img
                  className="rounded-xl mt-10 mb-4"
                  src={AssessmentsTab}
                  alt="MLflow LLM judges"
                />
                {/* Product GIF of the versions tab showing multiple models and then zooming into one to see evals + params like a prompt  */}
              </div>
            </GridItem>
          </Grid>
        </div>

        <LogosCarousel />
        <div className="flex flex-col items-center justify-center gap-16">
          <div className="flex flex-col gap-6">
            <SectionLabel color="red" label="WHY US?" />
            <Heading level={2}>Why MLflow is unique</Heading>
          </div>
          <Grid columns={2}>
            <GridItem>
              <div className="flex flex-col gap-4 h-full">
                <h2 className="text-white">
                  Unified, End-to-End MLOps and AI Observability
                </h2>
                <Body size="l">
                  MLflow offers a unified platform for the entire GenAI and ML
                  model lifecycle, simplifying the experience and boosting
                  collaboration by reducing tool integration friction.
                </Body>
              </div>
            </GridItem>

            <GridItem>
              <div className="flex flex-col gap-4 h-full">
                <h2 className="text-white">Open, Flexible, and Extensible</h2>
                <Body size="l">
                  Open-source and extensible, MLflow prevents vendor lock-in by
                  integrating with the GenAI/ML ecosystem and using open
                  protocols for data ownership, adapting to your existing and
                  future stacks.
                </Body>
              </div>
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4 h-full">
                <h2 className="text-white">
                  Enterprise-Grade Security &amp; Governance on a Unified Data
                  &amp; AI Platform
                </h2>
                <Body size="l">
                  Managed MLflow on Databricks offers enterprise-grade security
                  and deep Mosaic AI integrations for enhanced datasets,
                  development, RAG, serving, and gateways. Unity Catalog ensures
                  centralized governance over all AI assets.
                </Body>
              </div>
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4 h-full">
                <h2 className="text-white">
                  Unlock Downstream Value with Databricks AI/BI
                </h2>
                <Body size="l">
                  Leverage your GenAI and ML data for downstream business
                  processes by building rich performance dashboards, reports,
                  and queries with Databricks AI/BI and Databricks SQL.
                </Body>
              </div>
            </GridItem>
          </Grid>
        </div>
        <GetStartedWithMLflow />
        <LatestNews variant="red" />
        <SocialWidget variant="red" />
      </div>
    </Layout>
  );
}
