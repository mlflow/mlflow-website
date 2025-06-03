import {
  Layout,
  Grid,
  GridItem,
  AboveTheFold,
  BelowTheFold,
  Section,
  Card,
} from "../../components";

const FakeImage = () => (
  <div className="w-full aspect-[3/2] bg-black rounded-lg border border-[rgba(255,255,255,0.08)]"></div>
);

export default function Observability() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="Observability"
        title="Observability to debug and monitor"
        body="Gain visibility into your app's logic to debug issues and improve latency. Attach quality feedback and metadata to help you understand user behavior and improve quality."
        hasGetStartedButton
      >
        <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
      </AboveTheFold>

      <Section title="Best-in-class tracing">
        <Grid columns={2}>
          <GridItem>
            <Card
              title="End to end observability"
              body="Capture your app's inputs, outputs, and step-by-step execution: prompts, retrievals, tool calls, and more."
              // Animation of an app going from user request to retriever to LLM to tool call to response
              image={<FakeImage />}
            />
          </GridItem>
          <GridItem>
            <Card
              title="Cost & latency tracking"
              body="Track cost and latency for each step of your app's execution."
              image={<FakeImage />}
              // GIF screenshot of summary view paging through several traces
            />
          </GridItem>
          <GridItem>
            <Card
              title="Visualize your app's execution flow"
              body="Deep dive into your app's logic and latency with an intuitive UI for effective debugging."
              image={<FakeImage />}
              // GIF screenshot of detailed view + switching to see the timeline view
            />
          </GridItem>
          <GridItem>
            <Card
              title="Quickly understand many traces"
              body="Zoom out with a simplified summary UI to quickly review many traces at once to understand how your app processes user requests."
              image={<FakeImage />}
              // GIF screenshot of summary view paging through several traces
            />
          </GridItem>
        </Grid>
      </Section>

      <Section title="Simple, customizable instrumentation">
        <Grid columns={2}>
          <GridItem width="wide" direction="reverse">
            <Card
              title="Automatic instrumentation that's fully customizable"
              body="Instrument your app with 1-line-of-code integrations for over 20 popular LLM SDKs and generative AI frameworks. Optionally, use our intuitive APIs to customize the integrations."
              image={<FakeImage />}
              // Rendering of all the logos we support + a 1-liner of "mlflow.<flavor>.autolog()" in the center of the logos
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="SDKs for custom instrumentation"
              body="Use our intuitive SDK - decorators, context managers, and low-level APIs - to trace custom code or customize the integrations."
              image={<FakeImage />}
              // Code example of using custom instrumentation that maps the code to a sample trace from the code (example in go/genai/quality)
            />
          </GridItem>
          <GridItem width="wide" direction="reverse">
            <Card
              title="Instrument once, use in development and production"
              body="The same trace instrumentation works for production and development - so you can instrument once and get the same insight whether you are debugging in dev or observing in production."
              image={<FakeImage />}
              // Animation showing the same trace instrumentation working for production and development ??
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="OpenTelemetry compatible"
              body="Fully compatible with OpenTelemetry, so you can export traces to any OpenTelemetry compatible tool, providing you total ownership and portability of your generative AI data."
              image={<FakeImage />}
            />
          </GridItem>
        </Grid>
      </Section>

      <Section title="Annotation capabilities">
        <Grid columns={2}>
          <GridItem>
            <Card
              title="Attach Quality Feedback"
              body="Attach quality assessments from users, domain experts, or LLM judges/metrics directly on each trace so you can quickly understand and debug quality issues."
              image={<FakeImage />}
              // Product GIF of the feedback annotation view - show judge + review app + then show the feedback showing up on the trace in the trace UI
            />
          </GridItem>
          <GridItem direction="reverse">
            <Card
              title="Attach Metadata"
              body="Attach metadata, such as users, converastions sessions, and custom tags to traces to help you slice and dice based on user behavior."
              image={<FakeImage />}
              // Animation of grouping traces by user and conversation session
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Comments and notes"
              body="Add notes and context directly to traces for collaborative analysis."
              image={<FakeImage />}
              // Product GIF of adding feedback in the trace UI
            />
          </GridItem>
        </Grid>
      </Section>

      <Section id="production-monitoring" title="Production monitoring">
        <Grid columns={2}>
          <GridItem width="wide" direction="reverse">
            <Card
              title="Dashboards"
              body="Visualize operational and quality metrics with real-time dashboards that flag quality issues, latency spikes, and errors. Drill into dashboards to see the exact trace and step so you can rapidly debug issues."
              image={<FakeImage />}
              // Product GIF of the monitoring dashboard and clicking into a trace
            />
          </GridItem>
          <GridItem>
            <Card
              title="Advanced Search & Filtering"
              body="Quickly find relevant traces with powerful search and filtering options, allowing you to sift through large volumes of data efficiently."
              image={<FakeImage />}
              // Product GIF searching and filtering for tags + feedback
            />
          </GridItem>
          <GridItem direction="reverse">
            <Card
              title="Conversation Grouping"
              body="Easily group traces by chat conversation or user for streamlined analysis and a clearer understanding of interaction histories."
              image={<FakeImage />}
              // Figma of what this design will be!
            />
          </GridItem>
        </Grid>
      </Section>

      <BelowTheFold />
    </Layout>
  );
}
