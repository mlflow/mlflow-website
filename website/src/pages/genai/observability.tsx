import {
  Layout,
  SectionLabel,
  Button,
  Grid,
  GridItem,
  GetStartedWithMLflow,
  SocialWidget,
  GetStartedButton,
} from "../../components";

const FakeImage = () => (
  <div className="w-full aspect-[3/2] bg-black rounded-lg border border-[rgba(255,255,255,0.08)]"></div>
);

export default function Observability() {
  return (
    <Layout variant="red">
      <div
        className="flex flex-col bg-[linear-gradient(to_top,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-2.png')]
 bg-center bg-no-repeat w-full pt-42 pb-20 py-20 bg-size-[100%_340px]"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-container">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <SectionLabel color="red" label="OBSERVABILITY" />
            <h1 className="text-center text-wrap">
              Observability to debug and monitor
            </h1>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white/60">
              Gain visibility into your app's logic to debug issues and improve
              latency. Attach quality feedback and metadata to help you
              understand user behavior and improve quality.
            </p>

            <GetStartedButton />
          </div>
          <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
        </div>
      </div>
      <div className="flex flex-col gap-40 pt-20 w-full px-6 md:px-20 max-w-container">
        <div className="flex flex-col gap-10">
          <div className="flex flex-col w-full items-center justify-center gap-1">
            <div role="heading" aria-level={2} className="text-4xl">
              Best-in-class tracing
            </div>
          </div>
          <Grid columns={2}>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h3 className="text-white">End to end observability</h3>
                <p className="text-white/60">
                  Capture your app's inputs, outputs, and step-by-step
                  execution: prompts, retrievals, tool calls, and more.
                </p>
              </div>
              <FakeImage />
              {/* Animation of an app going from user request to retriever to LLM to tool call to response*/}
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Cost & latency tracking</h3>
                <p className="text-white/60">
                  Track cost and latency for each step of your app's execution.
                </p>
              </div>
              <FakeImage />
              {/* GIF screenshot of summary view paging through several traces */}
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  Visualize your app's execution flow
                </h3>
                <p className="text-white/60">
                  Deep dive into your app's logic and latency with an intuitive
                  UI for effective debugging.
                </p>
              </div>
              <FakeImage />
              {/* GIF screenshot of detailed view + switching to see the timeline view */}
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Quickly understand many traces</h3>
                <p className="text-white/60">
                  Zoom out with a simplified summary UI to quickly review many
                  traces at once to understand how your app processes user
                  requests.
                </p>
              </div>
              <FakeImage />
              {/* GIF screenshot of summary view paging through several traces */}
            </GridItem>
          </Grid>
        </div>
        <div className="flex flex-col gap-10">
          <div className="flex flex-col w-full items-center justify-center gap-1">
            <div role="heading" aria-level={2} className="text-4xl">
              Simple, customizable instrumentation
            </div>
          </div>
          <Grid columns={2}>
            <GridItem width="wide" direction="reverse">
              <div className="flex flex-col gap-10">
                <div className="flex flex-col gap-4">
                  <h3 className="text-white">
                    Automatic instrumentation that's fully customizable
                  </h3>
                </div>
                <p className="text-white/60">
                  Instrument your app with 1-line-of-code integrations for over
                  20 popular LLM SDKs and generative AI frameworks. Optionally,
                  use our intuitive APIs to customize the integrations.
                </p>
              </div>
              <FakeImage />
              {/* Rendering of all the logos we support + a 1-liner of "mlflow.<flavor>.autolog()" in the center of the logos */}
            </GridItem>

            <GridItem width="wide">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">SDKs for custom instrumentation</h3>
                <p className="text-white/60 text-lg">
                  Use our intuitive SDK - decorators, context managers, and
                  low-level APIs - to trace custom code or customize the
                  integrations.
                </p>
              </div>
              <FakeImage />
              {/* Code example of using custom instrumentation that maps the code to a sample trace from the code (example in go/genai/quality) */}
            </GridItem>
            <GridItem width="wide" direction="reverse">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  Instrument once, use in development and production
                </h3>
                <p className="text-white/60 text-lg">
                  The same trace instrumentation works for production and
                  development - so you can instrument once and get the same
                  insight whether you are debugging in dev or observing in
                  production.
                </p>
              </div>
              <FakeImage />
              {/* Animation showing the same trace instrumentation working for production and development ?? */}
            </GridItem>
            <GridItem width="wide">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">OpenTelemetry compatible</h3>
                <p className="text-white/60 text-lg">
                  Fully compatible with OpenTelemetry, so you can export traces
                  to any OpenTelemetry compatible tool, providing you total
                  ownership and portability of your generative AI data.
                </p>
              </div>
              <FakeImage />
            </GridItem>
          </Grid>
        </div>
        <div className="flex flex-col gap-10">
          <div className="flex flex-col w-full items-center justify-center gap-1">
            <div role="heading" aria-level={2} className="text-4xl">
              Annotation capabilities
            </div>
          </div>

          <Grid columns={2}>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Attach Quality Feedback</h3>
                <p className="text-white/60 text-lg">
                  Attach quality assessments from users, domain experts, or LLM
                  judges/metrics directly on each trace so you can quickly
                  understand and debug quality issues.
                </p>
              </div>
              <FakeImage />
              {/* Product GIF of the feedback annotation view - show judge + review app + then show the feedback showing up on the trace in the trace UI */}
            </GridItem>
            <GridItem direction="reverse">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Attach Metadata</h3>
                <p className="text-white/60 text-lg">
                  Attach metadata, such as users, converastions sessions, and
                  custom tags to traces to help you slice and dice based on user
                  behavior.
                </p>
              </div>
              <FakeImage />
              {/* Animation of grouping traces by user and conversation session */}
            </GridItem>
            <GridItem width="wide">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Comments and notes</h3>
                <p className="text-white/60 text-lg">
                  Add notes and context directly to traces for collaborative
                  analysis.
                </p>
              </div>
              <FakeImage />
              {/* Product GIF of adding feedback in the trace UI */}
            </GridItem>
          </Grid>
        </div>
        <div className="flex flex-col gap-10">
          <div className="flex flex-col w-full items-center justify-center gap-1">
            <div role="heading" aria-level={2} className="text-4xl">
              Production monitoring
            </div>
          </div>
          <Grid columns={2}>
            <GridItem width="wide" direction="reverse">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Dashboards</h3>
                <p className="text-white/60 text-lg">
                  Visualize operational and quality metrics with real-time
                  dashboards that flag quality issues, latency spikes, and
                  errors. Drill into dashboards to see the exact trace and step
                  so you can rapidly debug issues.
                </p>
              </div>
              <FakeImage />
              {/* Product GIF of the monitoring dashboard and clicking into a trace */}
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Advanced Search & Filtering</h3>
                <p className="text-white/60 text-lg">
                  Quickly find relevant traces with powerful search and
                  filtering options, allowing you to sift through large volumes
                  of data efficiently.
                </p>
              </div>
              <FakeImage />
              {/* Product GIF searching and filtering for tags + feedback */}
            </GridItem>
            <GridItem direction="reverse">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Conversation Grouping</h3>
                <p className="text-white/60 text-lg">
                  Easily group traces by chat conversation or user for
                  streamlined analysis and a clearer understanding of
                  interaction histories.
                </p>
              </div>
              <FakeImage />
              {/* Figma of what this design will be! */}
            </GridItem>
          </Grid>
        </div>

        <GetStartedWithMLflow />
        <SocialWidget variant="red" />
      </div>
    </Layout>
  );
}
