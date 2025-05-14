import {
  Layout,
  SectionLabel,
  Button,
  Grid,
  GridRow,
  GridItem,
  GetStartedWithMLflow,
  SocialWidget,
} from "../../components";

const FakeImage = () => (
  <div className="w-[600px] h-[400px] bg-black rounded-lg"></div>
);

export default function Observability() {
  return (
    <Layout variant="red">
      <div
        className="flex flex-col bg-[linear-gradient(to_top,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-2.png')]
 bg-center bg-no-repeat bg-cover w-full pt-42 pb-20 py-20 bg-size-[100%_340px]"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-7xl mx-auto">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <SectionLabel color="red" label="OBSERVABILITY" />
            <h1 className="text-center text-wrap">
              Capture and debug application logs with end-to-end observability
              (tracing)
            </h1>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white/60">
              Trace your app with OpenTelemetry-compatible SDKs that capture
              every invocation’s inputs, outputs, and step-by-step execution -
              prompts, retrievals, tool calls, and more - alongside cost,
              latency, and errors.
            </p>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white/60">
              Use these traces to quickly debug errors, improve business logic,
              and optimize cost and latency.
            </p>
            <Button>Get Started</Button>
          </div>
          <div className="w-[800px] h-[450px] bg-black rounded-lg mx-auto"></div>
        </div>
      </div>
      <div className="flex flex-col gap-40 w-full px-6 md:px-20 max-w-7xl mx-auto">
        <Grid>
          <GridRow>
            <GridItem className="flex flex-col md:flex-row gap-6 md:gap-20 py-10 justify-between items-center">
              <div className="flex flex-col gap-10">
                <div className="flex flex-col gap-4">
                  <h3 className="text-white">
                    Automatic Instrumentation, yet fully customizable
                  </h3>
                </div>

                <p className="text-white/60">
                  Use our 19+ authoring framework / LLM SDK integrations to
                  instrument your application with one line of code. Use our
                  intuitive APIs to customize the integrations or to instrument
                  tracing in any Python or Typescript application.
                </p>
              </div>
              <FakeImage />
            </GridItem>
          </GridRow>
          <GridRow>
            <GridItem className="py-10 pr-0 md:pr-10 gap-10">
              <FakeImage />
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  Visualize Your app’s Execution Flow
                </h3>
                <p className="text-white/60 text-lg">
                  Visualize, understand, and rapidly debug your application's
                  logic flow, capturing the complete request-response cycle,
                  from user queries to application responses, including each
                  intermediate step (e.g., retrieval, tool calls, LLM
                  interactions, and more).
                </p>
              </div>
            </GridItem>
            <GridItem className="py-10 pl-0 md:pl-10 gap-10">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  Instrument once, use in development and production
                </h3>
                <p className="text-white/60 text-lg">
                  he same trace instrumentation works for production and
                  development - so you can instrument once and get the same
                  insight whether you are debugging in dev or observing in
                  production.
                </p>
              </div>
              <FakeImage />
            </GridItem>
          </GridRow>
          <GridRow>
            <GridItem className="py-10 pr-0 md:pr-10 gap-10">
              <FakeImage />
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Attach Quality Feedback</h3>
                <p className="text-white/60 text-lg">
                  Attach quality assessments from users, domain experts, or LLM
                  judges/metrics directly on each trace so you can quickly
                  pinpoint and debug quality issues
                </p>
              </div>
            </GridItem>
            <GridItem className="py-10 pl-0 md:pl-10 gap-10">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Attach Operational Metrics</h3>
                <p className="text-white/60 text-lg">
                  Attach operational metrics like latency, cost, and resource
                  utilization to measure and improve your application’s
                  operational performance and cost. Errors/exceptions and their
                  associated stack traces are automatically captured.
                </p>
              </div>
              <FakeImage />
            </GridItem>
          </GridRow>
          <GridRow>
            <GridItem className="py-10 pr-0 md:pr-10 gap-10">
              <FakeImage />
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Monitoring and Alerting</h3>
                <p className="text-white/60 text-lg">
                  Visualize operational metrics with real-time dashboards and
                  set alerts that flag quality issues, latency spikes, and
                  errors. Drill into dashboards and alerts to see the exact
                  trace and step so you can rapidly debug issues.
                </p>
              </div>
            </GridItem>
            <GridItem className="py-10 pl-0 md:pl-10 gap-10">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Comprehensive Audit Trail</h3>
                <p className="text-white/60 text-lg">
                  Traces can be used as an audit trail that captures every
                  execution of your app for full transparency and accountability
                  across your GenAI apps.
                </p>
              </div>
              <FakeImage />
            </GridItem>
          </GridRow>
        </Grid>
        <GetStartedWithMLflow />
        <SocialWidget variant="red" />
      </div>
    </Layout>
  );
}
