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
  GridRow,
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
            <h1>Build confidently, deploy seamlessly</h1>
            <p className="text-white">
              Tackle the challenges of building GenAI head on
            </p>
          </div>
          <VerticalTabs defaultValue="tracking" className="w-full my-12 px-10">
            <VerticalTabsList>
              <VerticalTabsTrigger
                value="tracking"
                label="Capture and debug application logs with end-to-end observability (tracing)"
                description="Trace your app with OpenTelemetry-compatible SDKs that capture every invocation’s inputs, outputs, and step-by-step execution -  prompts, retrievals, tool calls, and more - alongside cost, latency, and errors. Use these traces to quickly debug errors, improve business logic, and optimize cost and latency."
              />
              <VerticalTabsTrigger
                value="llm-judges"
                label="Measure and improve quality with human-aligned, automated metrics (LLM judges)"
                description="Capture and convert expert feedback into metrics (LLM judges) that understand your business requirements and can measure the nuances of plain-language GenAI outputs. Use these metrics to evaluate, monitor, and improve quality in development and production at scale, without waiting for human review."
              />
              <VerticalTabsTrigger
                value="evaluation-datasets"
                label="Adapt to evolving user behavior with production log evaluation (evaluation datasets)"
                description="Use production logs to understand user behavior, turning low-quality responses into evaluation datasets and high-quality responses into regression tests. Replay these datasets to evaluate new prompts or app variants in development so you can ship the best variants to ensure your application continues to deliver high-quality responses as user behavior evolves."
              />
              <VerticalTabsTrigger
                value="human-insight"
                label="Incorporate human insight with an intuitive labeling and review experience "
                description="Capture domain expert feedback via web-based UIs and end-user ratings from your app via APIs. Use this feedback to enrich your understanding of how the app should behave and improve your custom LLM-judge metrics. "
              />
              <VerticalTabsTrigger
                value="enterprise-governance"
                label="Enterprise governance with Unity Catalog"
                description="MLflow is integrated with Unity Catalog to track the lifecycle and lineage of your app’s assets - models, prompts, datasets, and metrics - and apply access controls."
              />
            </VerticalTabsList>
            <VerticalTabsContent value="tracking">
              <img src="/img/demo-image.png" />
            </VerticalTabsContent>
            <VerticalTabsContent value="llm-judges">
              <img src="/img/demo-image.png" />
            </VerticalTabsContent>
            <VerticalTabsContent value="evaluation-datasets">
              <img src="/img/demo-image.png" />
            </VerticalTabsContent>

            <VerticalTabsContent value="human-insight">
              <img src="/img/demo-image.png" />
            </VerticalTabsContent>

            <VerticalTabsContent value="tab5">
              <img src="/img/demo-image.png" />
            </VerticalTabsContent>
          </VerticalTabs>
        </div>
        <LogosCarousel
          images={[
            "/img/companies/databricks.svg",
            "/img/companies/microsoft.svg",
            "/img/companies/meta.svg",
            "/img/companies/mosaicml.svg",
            "/img/companies/zillow.svg",
            "/img/companies/toyota.svg",
            "/img/companies/booking.svg",
            "/img/companies/wix.svg",
            "/img/companies/accenture.svg",
            "/img/companies/asml.svg",
          ]}
        />
        <div className="flex flex-col items-center justify-center gap-16">
          <div className="flex flex-col gap-6">
            <SectionLabel color="red" label="WHY US?" />
            <h1>Why MLflow is unique</h1>
          </div>
          <Grid>
            <GridRow>
              <GridItem className="py-10 px-10 gap-10">
                <div className="flex flex-col gap-4">
                  <h2 className="text-white">Tracing</h2>
                  <h3 className="text-white">
                    Capture and debug application logs with end-to-end
                    observability
                  </h3>
                  <p style={{ marginBottom: 0 }} className="text-white/60">
                    Trace your app with OpenTelemetry-compatible SDKs that
                    capture every invocation's inputs, outputs, and step-by-step
                    execution - prompts, retrievals, tool calls, and more -
                    alongside cost, latency, and errors. g{" "}
                  </p>
                  <p style={{ marginBottom: 0 }} className="text-white/60">
                    Use these traces to quickly debug errors, improve business
                    logic, and optimize cost and latency.
                  </p>
                  <a href="/genai/observability">
                    <Button variant="outline" size="small">
                      Learn more &gt;
                    </Button>
                  </a>
                </div>
              </GridItem>
              <GridItem className="justify-center px-10 py-10">
                <img
                  className="rounded-xl"
                  src={TraceTab}
                  alt="MLflow tracing"
                />
              </GridItem>
            </GridRow>
            <GridRow>
              <GridItem className="justify-center px-10 py-10">
                <img
                  className="rounded-xl"
                  src={EvalsTab}
                  alt="MLflow evaluations"
                />
              </GridItem>
              <GridItem className="py-10 px-10 gap-10">
                <div className="flex flex-col gap-4">
                  <h3 className="text-white">
                    Adapt to evolving user behavior with production log
                    evaluation
                  </h3>
                  <p style={{ marginBottom: 0 }} className="text-white/60">
                    Use production logs to understand user behavior, turning
                    low-quality responses into evaluation datasets and
                    high-quality responses into regression tests.
                  </p>
                  <p style={{ marginBottom: 0 }} className="text-white/60">
                    Replay these datasets to evaluate new prompts or app
                    variants in development so you can ship the best variants to
                    ensure your application continues to deliver high-quality
                    responses as user behavior evolves.
                  </p>
                  <a href="/genai/evaluations">
                    <Button variant="outline" size="small">
                      Learn more &gt;
                    </Button>
                  </a>
                </div>
              </GridItem>
            </GridRow>
            <GridRow>
              <GridItem className="px-10 py-10">
                <div className="flex flex-col gap-4">
                  <img
                    className="rounded-xl mt-10 mb-4"
                    src={AssessmentsTab}
                    alt="MLflow LLM judges"
                  />
                  <h3 className="text-white">
                    Measure and improve quality with human-aligned, automated
                    metrics
                  </h3>
                  <p style={{ marginBottom: 0 }} className="text-white/60">
                    Capture and convert expert feedback into metrics (LLM
                    judges) that understand your business requirements and can
                    measure the nuances of plain-language GenAI outputs.
                  </p>
                  <p style={{ marginBottom: 0 }} className="text-white/60">
                    Use these metrics to evaluate, monitor, and improve quality
                    in development and production at scale, without waiting for
                    human review.
                  </p>
                  <a href="/genai/quality-metrics">
                    <Button variant="outline" size="small">
                      Learn more &gt;
                    </Button>
                  </a>
                </div>
              </GridItem>
              <GridItem className="px-10 py-10">
                <div className="flex flex-col gap-4">
                  <img
                    className="rounded-xl mt-10 mb-4"
                    src={AssessmentsTab}
                    alt="MLflow LLM judges"
                  />
                  <h3 className="text-white">
                    Incorporate human insight with an intuitive labeling and
                    review experience
                  </h3>
                  <p style={{ marginBottom: 0 }} className="text-white/60">
                    Capture domain expert feedback via web-based UIs and
                    end-user ratings from your app via APIs.
                  </p>
                  <p style={{ marginBottom: 0 }} className="text-white/60">
                    Use this feedback to enrich your understanding of how the
                    app should behave and improve your custom LLM-judge metrics.
                  </p>
                  <a href="/genai/human-feedback">
                    <Button variant="outline" size="small">
                      Learn more &gt;
                    </Button>
                  </a>
                </div>
              </GridItem>
            </GridRow>
            <GridRow>
              <GridItem className="px-10 py-10" />
              <GridItem className="px-10 py-10">
                <div className="flex flex-col gap-4 justify-center">
                  <h3 className="text-white">
                    Enterprise governance with Unity Catalog
                  </h3>
                  <p style={{ marginBottom: 0 }} className="text-white/60">
                    MLflow is integrated with Unity Catalog to track the
                    lifecycle and lineage of your app’s assets - models,
                    prompts, datasets, and metrics - and apply access controls
                  </p>
                  <a href="/genai/governance">
                    <Button variant="outline" size="small">
                      Learn more &gt;
                    </Button>
                  </a>
                </div>
              </GridItem>
            </GridRow>
          </Grid>
        </div>
        <GetStartedWithMLflow />
        <LatestNews variant="red" />
        <SocialWidget variant="red" />
      </div>
    </Layout>
  );
}
