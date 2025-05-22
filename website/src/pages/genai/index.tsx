import ReviewApp from "@site/static/img/review-app.jpg";
import TraceTab from "@site/static/img/trace-tab.jpg";
import EvalsTab from "@site/static/img/evals-tab.jpg";
import AssessmentsTab from "@site/static/img/assessments.jpg";

import {
  Layout,
  CopyCommand,
  VerticalTabs,
  VerticalTabsList,
  VerticalTabsTrigger,
  VerticalTabsContent,
  SectionLabel,
  LogosCarousel,
  Grid,
  GridItem,
  GetStartedWithMLflow,
  LatestNews,
  SocialWidget,
  Button,
} from "../../components";

export default function GenAi(): JSX.Element {
  return (
    <Layout variant="red">
      <div
        className="flex flex-col bg-[linear-gradient(to_bottom,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-2.png')]
 bg-top bg-no-repeat bg-cover w-full pt-42 pb-20 py-20"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-container">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <h1 className="text-center text-wrap">
              Ship high-quality AI, fast
            </h1>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white">
              Traditional software and ML tests aren't built for GenAI's
              free-form language, making it difficult for teams to measure and
              improve quality.
            </p>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white">
              MLflow combines metrics that reliably measure GenAI quality with
              trace observability so you can measure, improve, and monitor
              quality, cost, and latency.
            </p>
          </div>
          <div className="flex flex-col md:flex-row gap-10">
            <CopyCommand code="pip install mlflow" />
          </div>
        </div>
      </div>
      <div className="flex flex-col gap-40 w-full px-6 md:px-20 max-w-container">
        <div className="flex flex-col w-full items-center justify-center gap-16">
          <div className="flex flex-col w-full items-center justify-center gap-6">
            <SectionLabel color="red" label="CORE FEATURES" />
            <h1>Observability to debug and monitor</h1>
          </div>
          <Grid columns={2}>
            <GridItem width="wide">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Debug with tracing</h3>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Debug and iterate on GenAI applications using MLflow's
                  tracing, which captures your app's entire execution, including
                  prompts, retrievals, tool calls.
                </p>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  MLflow's open-source, OpenTelemetry-compatible tracing SDK
                  helps avoid vendor lock-in.
                </p>

                <a href="/genai/observability">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </a>
              </div>
              <div>
                <img
                  className="rounded-xl"
                  src={TraceTab}
                  alt="MLflow tracing"
                />
              </div>
            </GridItem>
            <GridItem width="wide" direction="reverse">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Monitor in production</h3>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Maintain production quality with continuous monitoring of
                  quality, latency, and cost. Gain real-time visibility via
                  MLflow's dashboards and trace explorers.
                </p>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Configure automated online evaluations with alerts to quickly
                  address issues.
                </p>
                <a href="/genai/monitoring">
                  {" "}
                  {/* Placeholder link */}
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </a>
              </div>
              <div>
                <img
                  className="rounded-xl mt-10 mb-4"
                  src={AssessmentsTab} /* Placeholder image */
                  alt="MLflow Monitoring"
                />
              </div>
            </GridItem>
          </Grid>
        </div>
        <div className="flex flex-col w-full items-center justify-center gap-16">
          <div className="flex flex-col w-full items-center justify-center gap-6">
            <SectionLabel color="red" label="CORE FEATURES" />
            <h1>Evaluation to measure and improve quality</h1>
            <p className="text-lg text-center text-wrap max-w-2xl mx-auto text-white/80">
              MLflow simplifies GenAI evaluation, enabling easy collection and
              recording of LLM judge and human feedback directly on traces.
            </p>
            {/* <p className="text-white">
              Tackle the challenges of building GenAI head on
            </p> */}
          </div>
          <Grid columns={2}>
            <GridItem width="wide">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  LLM Judges for Quality Measurement
                </h3>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Utilize LLM-as-a-judge metrics, mimicking human expertise, to
                  assess and enhance GenAI quality. Access pre-built judges for
                  common metrics like hallucination or relevance, or develop
                  custom judges tailored to your business needs and expert
                  insights.
                </p>
                <a href="/genai/quality-metrics">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </a>
              </div>
              <div>
                <img
                  className="rounded-xl mt-10 mb-4"
                  src={AssessmentsTab}
                  alt="MLflow LLM judges"
                />
              </div>
            </GridItem>

            <GridItem width="wide" direction="reverse">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Evaluate with production logs</h3>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Adapt to user behavior by creating evaluation datasets and
                  regression tests from production logs. Replay these to assess
                  new prompts or app versions in development, ensuring optimal
                  variants reach production.
                </p>
                <a href="/genai/evaluations">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </a>
              </div>
              <div>
                <img
                  className="rounded-xl"
                  src={EvalsTab}
                  alt="MLflow evaluations"
                />
              </div>
            </GridItem>

            <GridItem width="wide">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  Capture and incorporate human feedback
                </h3>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Collect expert feedback through web UIs and end-user ratings
                  from your app via APIs. Use this feedback to understand how
                  your app should behave and align your custom LLM-judge metrics
                  with expert judgement.
                </p>
                <a href="/genai/human-feedback">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </a>
              </div>
              <div>
                <img
                  className="rounded-xl mt-10 mb-4"
                  src={AssessmentsTab}
                  alt="MLflow LLM judges"
                />
              </div>
            </GridItem>
          </Grid>
        </div>
        <div className="flex flex-col w-full items-center justify-center gap-16">
          <div className="flex flex-col w-full items-center justify-center gap-6">
            <SectionLabel color="red" label="CORE FEATURES" />
            <h1>Lifecycle management to track and version</h1>
            {/* <p className="text-white">
              Tackle the challenges of building GenAI head on
            </p> */}
          </div>
          <Grid columns={2}>
            <GridItem width="wide">
              <div className="flex flex-col gap-4 justify-center">
                <h3 className="text-white">Prompt registry</h3>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Version, compare, iterate on, and discover prompt templates
                  directly through the MLflow UI. Reuse prompts across multiple
                  versions of your agent or application code, and view rich
                  lineage identifying which versions are using each prompt.
                </p>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Evaluate and monitor prompt quality and performance across
                  multiple versions.
                </p>
                <a href="/genai/governance">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </a>
              </div>
              <div>
                <img
                  className="rounded-xl mt-10 mb-4"
                  src={AssessmentsTab}
                  alt="MLflow LLM judges"
                />
              </div>
            </GridItem>

            <GridItem width="wide" direction="reverse">
              <div className="flex flex-col gap-4 justify-center">
                <h3 className="text-white">Agent and application versioning</h3>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Version your agents, capturing their associated code,
                  parameters, and evalation metrics for each iteration. MLflow's
                  centralized management of agents complements Git, providing
                  full lifecycle capabilities for all your generative AI assets.
                </p>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Evaluation and observability data are linked to specific
                  agent/application versions, offering end-to-end versioning and
                  lineage for your entire GenAI application.
                </p>
                <a href="/genai/governance">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </a>
              </div>
              <div>
                <img
                  className="rounded-xl mt-10 mb-4"
                  src={AssessmentsTab}
                  alt="MLflow LLM judges"
                />
              </div>
            </GridItem>
          </Grid>
        </div>

        <LogosCarousel />
        <div className="flex flex-col items-center justify-center gap-16">
          <div className="flex flex-col gap-6">
            <SectionLabel color="red" label="WHY US?" />
            <h1>Why MLflow is unique</h1>
          </div>
          <Grid columns={2}>
            <GridItem>
              <div className="flex flex-col gap-4 h-full">
                <h2 className="text-white">
                  Unified, End-to-End MLOps and AI Observability
                </h2>
                <p className="text-white/60">
                  MLflow offers a unified platform for the entire GenAI and ML
                  model lifecycle, simplifying the experience and boosting
                  collaboration by reducing tool integration friction.
                </p>
              </div>
            </GridItem>

            <GridItem>
              <div className="flex flex-col gap-4 h-full">
                <h2 className="text-white">Open, Flexible, and Extensible</h2>
                <p className="text-white/60">
                  Open-source and extensible, MLflow prevents vendor lock-in by
                  integrating with the GenAI/ML ecosystem and using open
                  protocols for data ownership, adapting to your existing and
                  future stacks.
                </p>
              </div>
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4 h-full">
                <h2 className="text-white">
                  Enterprise-Grade Security &amp; Governance on a Unified Data
                  &amp; AI Platform
                </h2>
                <p className="text-white/60">
                  Managed MLflow on Databricks offers enterprise-grade security
                  and deep Mosaic AI integrations for enhanced datasets,
                  development, RAG, serving, and gateways. Unity Catalog ensures
                  centralized governance over all AI assets.
                </p>
              </div>
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4 h-full">
                <h2 className="text-white">
                  Unlock Downstream Value with Databricks AI/BI
                </h2>
                <p className="text-white/60">
                  Leverage your GenAI and ML data for downstream business
                  processes by building rich performance dashboards, reports,
                  and queries with Databricks AI/BI and Databricks SQL.
                </p>
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
