import {
  Layout,
  Grid,
  GridItem,
  GetStartedWithMLflow,
  SocialWidget,
  Body,
  AboveTheFold,
} from "../../components";

const FakeImage = () => (
  <div className="w-full aspect-[3/2] bg-black rounded-lg border border-[rgba(255,255,255,0.08)]"></div>
);

export default function Evaluations() {
  return (
    <Layout variant="red" direction="up">
      <AboveTheFold
        sectionLabel="Evaluations"
        title="Evaluation to measure and improve quality"
        body="Confidently evaluate quality in development and production to identify issues and iteratively test improvements."
        hasGetStartedButton
      >
        <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
      </AboveTheFold>
      <div className="flex flex-col gap-40 pt-20 w-full px-6 md:px-20 max-w-container">
        <div className="flex flex-col gap-10">
          <div className="flex flex-col w-full items-center justify-center gap-1">
            <div role="heading" aria-level={2} className="text-4xl">
              Accurately evaluate free-form language outputs with LLM judges
            </div>
          </div>
          <Grid columns={2}>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Pre-built LLM judges</h3>
                <Body size="l">
                  Quickly start with built-in LLM judges for safety,
                  hallucination, retrieval quality, and relevance. Our
                  research-backed judges provide accurate, reliable quality
                  evaluation aligned with human expertise.
                </Body>
              </div>
              <FakeImage />
              {/* Visual rendering of the built in judges with an image of our quality being the best in the middle (grab image from blog)*/}
            </GridItem>
            <GridItem direction="reverse">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Customized LLM judges</h3>
                <Body size="l">
                  Adapt our base model to create custom LLM judges tailored to
                  your business needs, aligning with your human expert's
                  judgment.
                </Body>
              </div>
              <FakeImage />
            </GridItem>
            {/* Animation showing humans providing feedback and it being synthesized into an llm judge */}
          </Grid>
        </div>
        <div className="flex flex-col gap-10">
          <div className="flex flex-col w-full items-center justify-center gap-1">
            <div role="heading" aria-level={2} className="text-4xl">
              Use production traffic to drive offline improvements
            </div>
          </div>
          <Grid columns={2}>
            <GridItem width="wide">
              <div className="flex flex-col gap-10">
                <div className="flex flex-col gap-4">
                  <h3 className="text-white">Evaluation datasets</h3>
                </div>

                <Body size="l">
                  Curate high-scoring traces for regression datasets and
                  low-scoring ones for evaluation datasets to use offline to
                  improve quality.
                </Body>
              </div>
              <FakeImage />
            </GridItem>
            {/* Show selecting traces from monitoring UI and saving into eval set */}
          </Grid>
        </div>
        <div className="flex flex-col gap-10">
          <div className="flex flex-col w-full items-center justify-center gap-1">
            <div role="heading" aria-level={2} className="text-4xl">
              Iteratively improve quality through evaluation
            </div>
          </div>
          <Grid columns={2}>
            <GridItem width="wide" direction="reverse">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Test new app / prompt variants</h3>
                <Body size="l">
                  MLflow's Evaluation SDK lets you test new application variants
                  (prompts, models, code) against evaluation and regression
                  datasets. Each variant is linked to its evaluation results,
                  enabling tracking of improvements over time.
                </Body>
              </div>
              <FakeImage />
              {/* Product GIF of running mlflow.evaluate and then seeing the eval results list UI */}
            </GridItem>
            <GridItem width="wide">
              <div className="flex flex-col gap-10">
                <div className="flex flex-col gap-4">
                  <h3 className="text-white">
                    Customize with code-based metrics
                  </h3>
                  <Body size="l">
                    Customize evaluation to measure any aspect of your app's
                    quality or performance using our custom metrics SDK. Convert
                    any Python function—from regex to custom logic—into a
                    metric.
                  </Body>
                </div>
              </div>
              <FakeImage />
              {/* Code snippet of a custom metric function */}
            </GridItem>
            <GridItem width="wide" direction="reverse">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  Identify root causes with evaluation review UIs
                </h3>
                <Body size="l">
                  Use MLflow's Evaluation UI to visualize a summary of your
                  evals and view results record-by-record to quickly identify
                  root causes and further improvement opportunities.
                </Body>
              </div>
              <FakeImage />
              {/* Product GIF of the evaluation UI and using it to filter and view individual results */}
            </GridItem>
            <GridItem width="wide">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Compare versions side-by-side</h3>
                <Body size="l">
                  Compare evaluations of 2 app variants to understand if your
                  changes improved or regressed quality. Review individual
                  questions side-by-side in the Trace Comparison UI to find
                  differences, debug regressions, and inform your next version.
                </Body>
              </div>
              <FakeImage />
              {/* Product GIF of the eval results compare UI, along with opening the trace compare UI */}
            </GridItem>
            <GridItem width="wide" direction="reverse">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">CI/CD support</h3>
                <Body size="l">
                  Automatically run evaluations in CI/CD workflows to
                  systematically validate that every change improves quality,
                  preventing regressions.
                </Body>
              </div>
              <FakeImage />
              {/* Animation of running eval in CI/CD workflow?? */}
            </GridItem>
          </Grid>
        </div>
        <GetStartedWithMLflow />
        <SocialWidget />
      </div>
    </Layout>
  );
}
