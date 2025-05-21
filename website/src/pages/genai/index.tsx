import ReviewApp from "@site/static/img/review-app.jpg";
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
          <Grid columns={2}>
            <GridItem width="wide">
              <div className="flex flex-col gap-4">
                <h2 className="text-white">Tracing</h2>
                <div
                  role="heading"
                  aria-level={3}
                  className="text-white text-lg"
                >
                  Capture and debug application logs with end-to-end
                  observability
                </div>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Trace your app with OpenTelemetry-compatible SDKs that capture
                  every invocation's inputs, outputs, and step-by-step execution
                  - prompts, retrievals, tool calls, and more - alongside cost,
                  latency, and errors. g{" "}
                </p>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Use these traces to quickly debug errors, improve business
                  logic, and optimize cost and latency.
                </p>
                <a href="/genai/observability" className="hidden md:block">
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
            <GridItem
              width="wide"
              direction="reverse"
              lg-width="wide"
              lg-direction="normal"
            >
              <div className="flex flex-col gap-4">
                <div
                  role="heading"
                  aria-level={3}
                  className="text-white text-xl"
                >
                  Adapt to evolving user behavior with production log evaluation
                </div>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Use production logs to understand user behavior, turning
                  low-quality responses into evaluation datasets and
                  high-quality responses into regression tests.
                </p>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Replay these datasets to evaluate new prompts or app variants
                  in development so you can ship the best variants to ensure
                  your application continues to deliver high-quality responses
                  as user behavior evolves.
                </p>
                <a href="/genai/evaluations" className="hidden md:block">
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
            <GridItem direction="reverse" lg-width="wide" lg-direction="normal">
              <div className="flex flex-col gap-4">
                <div
                  role="heading"
                  aria-level={3}
                  className="text-white text-xl"
                >
                  Measure and improve quality with human-aligned, automated
                  metrics
                </div>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Capture and convert expert feedback into metrics (LLM judges)
                  that understand your business requirements and can measure the
                  nuances of plain-language GenAI outputs.
                </p>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Use these metrics to evaluate, monitor, and improve quality in
                  development and production at scale, without waiting for human
                  review.
                </p>
                <a href="/genai/quality-metrics" className="hidden md:block">
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
            <GridItem direction="reverse" lg-width="wide" lg-direction="normal">
              <div className="flex flex-col gap-4">
                <div
                  role="heading"
                  aria-level={3}
                  className="text-white text-xl"
                >
                  Incorporate human insight with an intuitive labeling and
                  review experience
                </div>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Capture domain expert feedback via web-based UIs and end-user
                  ratings from your app via APIs.
                </p>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Use this feedback to enrich your understanding of how the app
                  should behave and improve your custom LLM-judge metrics.
                </p>
                <a href="/genai/human-feedback" className="hidden md:block">
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
            <GridItem
              width="wide"
              direction="reverse"
              lg-width="wide"
              lg-direction="normal"
            >
              <div className="flex flex-col gap-4 justify-center">
                <div
                  role="heading"
                  aria-level={3}
                  className="text-white text-xl"
                >
                  Enterprise governance with Unity Catalog
                </div>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  MLflow is integrated with Unity Catalog to track the lifecycle
                  and lineage of your app’s assets - models, prompts, datasets,
                  and metrics - and apply access controls
                </p>
                <a href="/genai/governance" className="hidden md:block">
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
          <Grid columns={2} lg-columns={4}>
            <GridItem>
              <article className="flex flex-col gap-4">
                <div
                  role="heading"
                  aria-level={2}
                  className="text-xl font-bold"
                >
                  Built on top of a data platform
                </div>
                <p className="text-white/60 text-sm">
                  Evaluation and monitoring are workflows that generate data.
                  Easily use the data from your evaluation & monitoring
                  workflows to build dashboards & apps using the features of the
                  Databricks platform.
                </p>
              </article>
            </GridItem>
            <GridItem>
              <article className="flex flex-col gap-4">
                <div
                  role="heading"
                  aria-level={2}
                  className="text-xl font-bold"
                >
                  Integrated and unified governance
                </div>
                <p className="text-white/60 text-sm">
                  Tightly integrated with Unity Catalog, which offers unified,
                  enterprise-grade governance across all your data and ai assets
                  - including all assets created by MLflow.
                </p>
              </article>
            </GridItem>
            <GridItem>
              <article className="flex flex-col gap-4">
                <div
                  role="heading"
                  aria-level={2}
                  className="text-xl font-bold"
                >
                  Data intelligence
                </div>
                <p className="text-white/60 text-sm">
                  Data Intelligence that helps make the developer workflow for
                  improving quality faster/more efficient - LLM judges that are
                  tuned to understand your business data; topic detection &
                  classification that help understand your users.
                </p>
              </article>
            </GridItem>
            <GridItem>
              <article className="flex flex-col gap-4">
                <div
                  role="heading"
                  aria-level={2}
                  className="text-xl font-bold"
                >
                  Secure and scalable
                </div>
                <p className="text-white/60 text-sm">
                  Databricks is a trusted vendor and we host the managed version
                  - no need to self host since we are a trusted vendor.
                </p>
              </article>
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
