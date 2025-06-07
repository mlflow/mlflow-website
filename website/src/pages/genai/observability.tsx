import { MLFLOW_DOCS_URL } from "@site/src/constants";
import {
  Layout,
  Grid,
  GridItem,
  AboveTheFold,
  BelowTheFold,
  Section,
  Card,
  HeroImage,
} from "../../components";
import CardHero from "@site/static/img/GenAI_observability/GenAI_observability_hero.png";
import Card1 from "@site/static/img/GenAI_observability/GenAI_observability_1.png";
import Card2 from "@site/static/img/GenAI_observability/GenAI_observability_2.png";
import Card3 from "@site/static/img/GenAI_observability/GenAI_observability_3.png";
import Card4 from "@site/static/img/GenAI_observability/GenAI_observability_4.png";
import Card5 from "@site/static/img/GenAI_observability/GenAI_observability_5.png";
import Card6 from "@site/static/img/GenAI_observability/GenAI_observability_6.png";
import Card7 from "@site/static/img/GenAI_observability/GenAI_observability_7.png";
import Card8 from "@site/static/img/GenAI_observability/GenAI_observability_8.png";
import Card9 from "@site/static/img/GenAI_observability/GenAI_observability_9.png";
import Card10 from "@site/static/img/GenAI_observability/GenAI_observability_10.png";
import Card11 from "@site/static/img/GenAI_observability/GenAI_observability_11.png";
import Card12 from "@site/static/img/GenAI_observability/GenAI_observability_12.png";
import Card13 from "@site/static/img/GenAI_observability/GenAI_observability_13.png";
import Card14 from "@site/static/img/GenAI_observability/GenAI_observability_14.png";

export default function Observability() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="Observability"
        title="Observability to debug and monitor"
        body="Gain visibility into your app's logic to debug issues and improve latency. Attach quality feedback and metadata to help you understand user behavior and improve quality."
        hasGetStartedButton={MLFLOW_DOCS_URL}
      >
        <HeroImage src={CardHero} alt="" />
      </AboveTheFold>

      <Section title="Best-in-class tracing">
        <Grid columns={2}>
          <GridItem width="wide">
            <Card
              title="End to end observability"
              body="Capture your app's inputs, outputs, and step-by-step execution: prompts, retrievals, tool calls, and more."
              image={<img src={Card1} alt="" />}
              // Animation of an app going from user request to retriever to LLM to tool call to response
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Cost & latency tracking"
              body="Track cost and latency for each step of your app's execution."
              image={<img src={Card2} alt="" />}
              // GIF screenshot of summary view paging through several traces
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Visualize your app's execution flow"
              body="Deep dive into your app's logic and latency with an intuitive UI for effective debugging."
              image={<img src={Card3} alt="" />}
              // GIF screenshot of detailed view + switching to see the timeline view
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Quickly understand many traces"
              body="Zoom out with a simplified summary UI to quickly review many traces at once to understand how your app processes user requests."
              image={<img src={Card4} alt="" />}
              // GIF screenshot of summary view paging through several traces
            />
          </GridItem>
        </Grid>
      </Section>

      <Section title="Simple, customizable instrumentation">
        <Grid columns={2}>
          <GridItem width="wide">
            <Card
              title="Automatic instrumentation that's fully customizable"
              body="Instrument your app with 1-line-of-code integrations for over 20 popular LLM SDKs and generative AI frameworks. Optionally, use our intuitive APIs to customize the integrations."
              image={<img src={Card5} alt="" />}
              // Rendering of all the logos we support + a 1-liner of "mlflow.<flavor>.autolog()" in the center of the logos
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="SDKs for custom instrumentation"
              body="Use our intuitive SDK - decorators, context managers, and low-level APIs - to trace custom code or customize the integrations."
              image={<img src={Card6} alt="" />}
              // Code example of using custom instrumentation that maps the code to a sample trace from the code (example in go/genai/quality)
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Instrument once, use in development and production"
              body="The same trace instrumentation works for production and development - so you can instrument once and get the same insight whether you are debugging in dev or observing in production."
              image={<img src={Card7} alt="" />}
              // Animation showing the same trace instrumentation working for production and development ??
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="OpenTelemetry compatible"
              body="Fully compatible with OpenTelemetry, so you can export traces to any OpenTelemetry compatible tool, providing you total ownership and portability of your generative AI data."
              image={<img src={Card8} alt="" />}
            />
          </GridItem>
        </Grid>
      </Section>

      <Section title="Annotation capabilities">
        <Grid columns={2}>
          <GridItem width="wide">
            <Card
              title="Attach Quality Feedback"
              body="Attach quality assessments from users, domain experts, or LLM judges/metrics directly on each trace so you can quickly understand and debug quality issues."
              image={<img src={Card9} alt="" />}
              // Product GIF of the feedback annotation view - show judge + review app + then show the feedback showing up on the trace in the trace UI
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Attach Metadata"
              body="Attach metadata, such as users, converastions sessions, and custom tags to traces to help you slice and dice based on user behavior."
              image={<img src={Card10} alt="" />}
              // Animation of grouping traces by user and conversation session
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Comments and notes"
              body="Add notes and context directly to traces for collaborative analysis."
              image={<img src={Card11} alt="" />}
              // Product GIF of adding feedback in the trace UI
            />
          </GridItem>
        </Grid>
      </Section>

      <Section id="production-monitoring" title="Production monitoring">
        <Grid columns={2}>
          <GridItem width="wide">
            <Card
              title="Dashboards"
              body="Visualize operational and quality metrics with real-time dashboards that flag quality issues, latency spikes, and errors. Drill into dashboards to see the exact trace and step so you can rapidly debug issues."
              image={<img src={Card12} alt="" />}
              // Product GIF of the monitoring dashboard and clicking into a trace
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Advanced Search & Filtering"
              body="Quickly find relevant traces with powerful search and filtering options, allowing you to sift through large volumes of data efficiently."
              image={<img src={Card13} alt="" />}
              // Product GIF searching and filtering for tags + feedback
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Conversation Grouping"
              body="Easily group traces by chat conversation or user for streamlined analysis and a clearer understanding of interaction histories."
              image={<img src={Card14} alt="" />}
              // Figma of what this design will be!
            />
          </GridItem>
        </Grid>
      </Section>

      <BelowTheFold />
    </Layout>
  );
}
