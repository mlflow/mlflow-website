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

export default function HumanFeedback() {
  return (
    <Layout variant="red">
      <div
        className="flex flex-col bg-[linear-gradient(to_top,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-2.png')]
 bg-center bg-no-repeat w-full pt-42 pb-20 py-20 bg-size-[100%_340px]"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-container">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <SectionLabel color="red" label="HUMAN FEEDBACK" />
            <h1 className="text-center text-wrap">
              Incorporate human insight to understand and improve quality
            </h1>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white/60">
              Capture domain expert feedback to understand how your app should
              behave and align your custom LLM-judge metrics with those expert's
              judgement.
            </p>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white/60">
              Capture end user feedback to quickly pinpoint quality issues in
              production.
            </p>
            <GetStartedButton />
          </div>
          <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
        </div>
      </div>
      <div className="flex flex-col gap-40 w-full px-6 md:px-20 max-w-container">
        <Grid columns={2}>
          <GridItem>
            <div className="flex flex-col gap-10">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  Initiative labeling UIs, built for business user
                </h3>
              </div>

              <p className="text-white/60">
                MLflow's Review App is built from the ground up to quickly and
                easily collect feedback from busy domain experts. Share any
                production log to experts so they can review the interaction and
                provide feedback. Use pre-defined questions or create custom
                questions.
              </p>
            </div>
            <FakeImage />
          </GridItem>
          <GridItem direction="reverse">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Track and visualize feedback</h3>
              <p className="text-white/60 text-lg">
                Replace spreadsheet-based labeling workflow with MLflow's
                feedback data model - end user and domain expert feedback is
                directly attached to the traces as version-controlled labels.
                Visualize feedback in the MLflow Trace UIs and dashboards to
                quickly pinpoint quality issues.
              </p>
            </div>
            <FakeImage />
          </GridItem>
          <GridItem width="wide">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Capture end-user feedback</h3>
              <p className="text-white/60 text-lg">
                Use MLflow's APIs to link end user feedback from your deployed
                application to the source MLflow Trace.
              </p>
            </div>
            <FakeImage />
          </GridItem>
          <GridItem width="wide" direction="reverse">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Integrated Chat App</h3>
              <p className="text-white/60 text-lg">
                Quickly deploy new app versions to the Review App's built-in
                chat UI so domain experts can interact and provide instant
                feedback so you can rapidly assess quality and pinpoint issues.
              </p>
            </div>
            <FakeImage />
          </GridItem>
        </Grid>
        <GetStartedWithMLflow />
        <SocialWidget variant="red" />
      </div>
    </Layout>
  );
}
