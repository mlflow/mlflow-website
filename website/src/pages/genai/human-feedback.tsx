import {
  Layout,
  SectionLabel,
  Grid,
  GridItem,
  GetStartedWithMLflow,
  SocialWidget,
  GetStartedButton,
  Heading,
  Body,
} from "../../components";

const FakeImage = () => (
  <div className="w-full aspect-[3/2] bg-black rounded-lg border border-[rgba(255,255,255,0.08)]"></div>
);

export default function HumanFeedback() {
  return (
    <Layout variant="red" direction="up">
      <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-container">
        <div className="flex flex-col justify-center items-center gap-6 w-full">
          <SectionLabel color="red" label="HUMAN FEEDBACK" />
          <Heading level={1}>
            Incorporate human insight to understand and improve quality
          </Heading>
          <div className="max-w-3xl mx-auto text-center">
            <Body size="l">
              Capture domain expert feedback to understand how your app should
              behave and align your custom LLM-judge metrics with those expert's
              judgement.
            </Body>
            <Body size="l">
              Capture end user feedback to quickly pinpoint quality issues in
              production.
            </Body>
          </div>
          <GetStartedButton />
        </div>
        <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
      </div>
      <div className="flex flex-col gap-40 w-full px-6 md:px-20 max-w-container">
        <Grid columns={2}>
          <GridItem>
            <div className="flex flex-col gap-10">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  Intuitive labeling UIs for business users
                </h3>
              </div>

              <Body size="l">
                MLflow's Review App enables busy domain experts to quickly
                provide feedback on production logs. Share logs for review and
                use predefined or custom questions.
              </Body>
            </div>
            <FakeImage />
            {/* Hybrid animation / product GIF of showing a trace in the trace UI, then animating it to go to the review app, and then seeing the review app in action (provide feedback clicked) and then animation to see it on the trace UI */}
          </GridItem>
          <GridItem direction="reverse">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Track and visualize feedback</h3>
              <Body size="l">
                MLflow replaces spreadsheets by attaching expert/user feedback
                to traces as versioned labels. Visualize this data in MLflow
                Trace UIs and dashboards to swiftly identify quality issues.
              </Body>
            </div>
            <FakeImage />
            {/* Animation of an app executing, producing a trace, having feedback attached to it, and then seeing the feedback in the trace UI */}
          </GridItem>
          <GridItem width="wide">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Capture end-user feedback</h3>
              <Body size="l">
                MLflow scalable feedback APIs allow you to attach end-user
                feedback from your deployed app to the source MLflow Trace, so
                you debug negative feedback with access to the step-by-step
                execution.
              </Body>
            </div>
            <FakeImage />
            {/* Product GIF of a fake production app and then seeing the feedback in the trace UI */}
          </GridItem>
          <GridItem width="wide" direction="reverse">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Integrated Chat App</h3>
              <Body size="l">
                Deploy new app versions to the Review App's chat UI. Domain
                experts can interact, give instant feedback, and help rapidly
                assess quality and pinpoint issues.
              </Body>
            </div>
            <FakeImage />
            {/* Product GIF of the review app chat mode and then seeing the feedback in the trace UI */}
          </GridItem>
        </Grid>
        <GetStartedWithMLflow />
        <SocialWidget />
      </div>
    </Layout>
  );
}
