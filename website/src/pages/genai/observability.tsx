import {
  Layout,
  Grid,
  GridItem,
  Body,
  AboveTheFold,
  BelowTheFold,
} from "../../components";

const FakeImage = () => (
  <div className="w-full aspect-[3/2] bg-black rounded-lg border border-[rgba(255,255,255,0.08)]"></div>
);

export default function Observability() {
  return (
    <Layout variant="red" direction="up">
      <AboveTheFold
        sectionLabel="Observability"
        title="Observability to debug and monitor"
        body="Gain visibility into your app's logic to debug issues and improve latency. Attach quality feedback and metadata to help you understand user behavior and improve quality."
        hasGetStartedButton
      >
        <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
      </AboveTheFold>

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
              <Body size="l">
                Capture your app's inputs, outputs, and step-by-step execution:
                prompts, retrievals, tool calls, and more.
              </Body>
            </div>
            <FakeImage />
            {/* Animation of an app going from user request to retriever to LLM to tool call to response*/}
          </GridItem>
          <GridItem>
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Cost & latency tracking</h3>
              <Body size="l">
                Track cost and latency for each step of your app's execution.
              </Body>
            </div>
            <FakeImage />
            {/* GIF screenshot of summary view paging through several traces */}
          </GridItem>
          <GridItem>
            <div className="flex flex-col gap-4">
              <h3 className="text-white">
                Visualize your app's execution flow
              </h3>
              <Body size="l">
                Deep dive into your app's logic and latency with an intuitive UI
                for effective debugging.
              </Body>
            </div>
            <FakeImage />
            {/* GIF screenshot of detailed view + switching to see the timeline view */}
          </GridItem>
          <GridItem>
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Quickly understand many traces</h3>
              <Body size="l">
                Zoom out with a simplified summary UI to quickly review many
                traces at once to understand how your app processes user
                requests.
              </Body>
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
              <Body size="l">
                Instrument your app with 1-line-of-code integrations for over 20
                popular LLM SDKs and generative AI frameworks. Optionally, use
                our intuitive APIs to customize the integrations.
              </Body>
            </div>
            <FakeImage />
            {/* Rendering of all the logos we support + a 1-liner of "mlflow.<flavor>.autolog()" in the center of the logos */}
          </GridItem>

          <GridItem width="wide">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">SDKs for custom instrumentation</h3>
              <Body size="l">
                Use our intuitive SDK - decorators, context managers, and
                low-level APIs - to trace custom code or customize the
                integrations.
              </Body>
            </div>
            <FakeImage />
            {/* Code example of using custom instrumentation that maps the code to a sample trace from the code (example in go/genai/quality) */}
          </GridItem>
          <GridItem width="wide" direction="reverse">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">
                Instrument once, use in development and production
              </h3>
              <Body size="l">
                The same trace instrumentation works for production and
                development - so you can instrument once and get the same
                insight whether you are debugging in dev or observing in
                production.
              </Body>
            </div>
            <FakeImage />
            {/* Animation showing the same trace instrumentation working for production and development ?? */}
          </GridItem>
          <GridItem width="wide">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">OpenTelemetry compatible</h3>
              <Body size="l">
                Fully compatible with OpenTelemetry, so you can export traces to
                any OpenTelemetry compatible tool, providing you total ownership
                and portability of your generative AI data.
              </Body>
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
              <Body size="l">
                Attach quality assessments from users, domain experts, or LLM
                judges/metrics directly on each trace so you can quickly
                understand and debug quality issues.
              </Body>
            </div>
            <FakeImage />
            {/* Product GIF of the feedback annotation view - show judge + review app + then show the feedback showing up on the trace in the trace UI */}
          </GridItem>
          <GridItem direction="reverse">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Attach Metadata</h3>
              <Body size="l">
                Attach metadata, such as users, converastions sessions, and
                custom tags to traces to help you slice and dice based on user
                behavior.
              </Body>
            </div>
            <FakeImage />
            {/* Animation of grouping traces by user and conversation session */}
          </GridItem>
          <GridItem width="wide">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Comments and notes</h3>
              <Body size="l">
                Add notes and context directly to traces for collaborative
                analysis.
              </Body>
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
              <Body size="l">
                Visualize operational and quality metrics with real-time
                dashboards that flag quality issues, latency spikes, and errors.
                Drill into dashboards to see the exact trace and step so you can
                rapidly debug issues.
              </Body>
            </div>
            <FakeImage />
            {/* Product GIF of the monitoring dashboard and clicking into a trace */}
          </GridItem>
          <GridItem>
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Advanced Search & Filtering</h3>
              <Body size="l">
                Quickly find relevant traces with powerful search and filtering
                options, allowing you to sift through large volumes of data
                efficiently.
              </Body>
            </div>
            <FakeImage />
            {/* Product GIF searching and filtering for tags + feedback */}
          </GridItem>
          <GridItem direction="reverse">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Conversation Grouping</h3>
              <Body size="l">
                Easily group traces by chat conversation or user for streamlined
                analysis and a clearer understanding of interaction histories.
              </Body>
            </div>
            <FakeImage />
            {/* Figma of what this design will be! */}
          </GridItem>
        </Grid>
      </div>

      <BelowTheFold />
    </Layout>
  );
}
