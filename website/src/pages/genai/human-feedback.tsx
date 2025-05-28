import {
  Layout,
  Grid,
  GridItem,
  AboveTheFold,
  BelowTheFold,
  Card,
} from "../../components";

const FakeImage = () => (
  <div className="w-full aspect-[3/2] bg-black rounded-lg border border-[rgba(255,255,255,0.08)]"></div>
);

export default function HumanFeedback() {
  return (
    <Layout variant="red" direction="up">
      <AboveTheFold
        sectionLabel="Human feedback"
        title="Incorporate human insight to understand and improve quality"
        body={[
          "Capture domain expert feedback to understand how your app should behave and align your custom LLM-judge metrics with those expert's judgement.",
          "Capture end user feedback to quickly pinpoint quality issues in production.",
        ]}
        hasGetStartedButton
      >
        <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
      </AboveTheFold>

      <Grid columns={2}>
        <GridItem>
          <Card
            title="Intuitive labeling UIs for business users"
            body="MLflow's Review App enables busy domain experts to quickly provide feedback on production logs. Share logs for review and use predefined or custom questions."
            // Hybrid animation / product GIF of showing a trace in the trace UI, then animating it to go to the review app, and then seeing the review app in action (provide feedback clicked) and then animation to see it on the trace UI
            image={<FakeImage />}
          />
        </GridItem>
        <GridItem direction="reverse">
          <Card
            title="Track and visualize feedback"
            body="MLflow replaces spreadsheets by attaching expert/user feedback to traces as versioned labels. Visualize this data in MLflow Trace UIs and dashboards to swiftly identify quality issues."
            image={<FakeImage />}
            // Animation of an app executing, producing a trace, having feedback attached to it, and then seeing the feedback in the trace UI
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Capture end-user feedback"
            body="MLflow scalable feedback APIs allow you to attach end-user feedback from your deployed app to the source MLflow Trace, so you debug negative feedback with access to the step-by-step execution."
            // Product GIF of a fake production app and then seeing the feedback in the trace UI
            image={<FakeImage />}
          />
        </GridItem>
        <GridItem width="wide" direction="reverse">
          <Card
            title="Integrated Chat App"
            body="Deploy new app versions to the Review App's chat UI. Domain experts can interact, give instant feedback, and help rapidly assess quality and pinpoint issues."
            // Product GIF of the review app chat mode and then seeing the feedback in the trace UI
            image={<FakeImage />}
          />
        </GridItem>
      </Grid>

      <BelowTheFold />
    </Layout>
  );
}
