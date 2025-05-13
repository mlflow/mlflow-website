import {
  Layout,
  SectionLabel,
  Button,
  Grid,
  GridRow,
  GridItem,
  GetStartedWithMLflow,
  SocialWidget,
  GetStartedButton,
} from "../../components";

const FakeImage = () => (
  <div className="w-[600px] h-[400px] bg-black rounded-lg"></div>
);

export default function HumanFeedback() {
  return (
    <Layout variant="red">
      <div
        className="flex flex-col bg-[linear-gradient(to_top,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-2.png')]
 bg-center bg-no-repeat bg-cover w-full pt-42 pb-20 py-20 bg-size-[100%_340px]"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-7xl mx-auto">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <SectionLabel color="red" label="QUALITY METRICS" />
            <h1 className="text-center text-wrap">
              Incorporate human insight with an intuitive labeling and review
              experience
            </h1>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white/60">
              Capture domain expert feedback via web-based UIs and end-user
              ratings from your app via APIs.
            </p>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white/60">
              Use this feedback to enrich your understanding of how the app
              should behave and improve your custom LLM-judge metrics.
            </p>
            <GetStartedButton />
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
                    Initiative labeling UIs, built for business user
                  </h3>
                </div>

                <p className="text-white/60">
                  MLflow’s Review App is built from the ground up to quickly and
                  easily collect feedback from busy domain experts on production
                  logs - experts can view inputs and outputs, alongside
                  intermediate steps such as retrieval and tool calls, so they
                  can provide feedback in seconds. Use our pre-defined feedback
                  questions or add your own custom questions.
                </p>
              </div>
              <FakeImage />
            </GridItem>
          </GridRow>
          <GridRow>
            <GridItem className="py-10 pr-0 md:pr-10 gap-10">
              <FakeImage />
              <div className="flex flex-col gap-4">
                <h3 className="text-white">No more spreadsheets!</h3>
                <p className="text-white/60 text-lg">
                  Replace your spreadsheet-based labeling workflow with MLflow’s
                  data model - feedback from the review app flows is directly
                  attached to the source traces as version-controlled labels.
                  Track labeling progress and visualize results via the UI.
                </p>
              </div>
            </GridItem>
            <GridItem className="py-10 pl-0 md:pl-10 gap-10">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Capture end-user feedback</h3>
                <p className="text-white/60 text-lg">
                  In addition to domain experts, use MLflow’s APIs to capture
                  feedback that end users provide in your deployed application.
                  Similar to the review app, feedback is directly attached to
                  the source traces as version-controlled labels & can be
                  visualized via the UI.
                </p>
              </div>
              <FakeImage />
            </GridItem>
          </GridRow>
          <GridRow>
            <GridItem className="flex flex-col md:flex-row gap-6 md:gap-20 py-10 justify-between items-center">
              <FakeImage />
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Integrated Chat App</h3>
                <p className="text-white/60 text-lg">
                  When you have a new version ready for testing, load it into
                  the Review App’s pre-built chat UI to let domain experts use
                  the version and quickly provide feedback.
                </p>
              </div>
            </GridItem>
          </GridRow>
        </Grid>
        <GetStartedWithMLflow />
        <SocialWidget variant="red" />
      </div>
    </Layout>
  );
}
