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

export default function Evaluations() {
  return (
    <Layout variant="red">
      <div
        className="flex flex-col bg-[linear-gradient(to_top,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-2.png')]
 bg-center bg-no-repeat w-full pt-42 pb-20 py-20 bg-size-[100%_340px]"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-container">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <SectionLabel color="red" label="EVALUATIONS" />
            <h1 className="text-center text-wrap max-w-2xl">
              Evaluation to measure and improve quality
            </h1>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-gray-600">
              Confidently evaluate quality in development and production to
              identify issues and iteratively test improvements.
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
              Accurately evaluate free-form language outputs with LLM judges
            </div>
          </div>
          <Grid columns={2}>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Pre-built LLM judges</h3>
                <p className="text-gray-600 text-lg">
                  Quickly start with built-in LLM judges for safety,
                  hallucination, retrieval quality, and relevance. Our
                  research-backed judges provide accurate, reliable quality
                  evaluation aligned with human expertise.
                </p>
              </div>
              <FakeImage />
              {/* Visual rendering of the built in judges with an image of our quality being the best in the middle (grab image from blog)*/}
            </GridItem>
            <GridItem direction="reverse">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Customized LLM judges</h3>
                <p className="text-gray-600 text-lg">
                  Adapt our base model to create custom LLM judges tailored to
                  your business needs, aligning with your human expert's
                  judgment.
                </p>
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

                <p className="text-gray-600">
                  Curate high-scoring traces for regression datasets and
                  low-scoring ones for evaluation datasets to use offline to
                  improve quality.
                </p>
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
                <p className="text-gray-600 text-lg">
                  MLflow's Evaluation SDK lets you test new application variants
                  (prompts, models, code) against evaluation and regression
                  datasets. Each variant is linked to its evaluation results,
                  enabling tracking of improvements over time.
                </p>
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
                  <p className="text-gray-600">
                    Customize evaluation to measure any aspect of your app's
                    quality or performance using our custom metrics SDK. Convert
                    any Python function—from regex to custom logic—into a
                    metric.
                  </p>
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
                <p className="text-gray-600 text-lg">
                  Use MLflow's Evaluation UI to visualize a summary of your
                  evals and view results record-by-record to quickly identify
                  root causes and further improvement opportunities.
                </p>
              </div>
              <FakeImage />
              {/* Product GIF of the evaluation UI and using it to filter and view individual results */}
            </GridItem>
            <GridItem width="wide">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Compare versions side-by-side</h3>
                <p className="text-gray-600 text-lg">
                  Compare evaluations of 2 app variants to understand if your
                  changes improved or regressed quality. Review individual
                  questions side-by-side in the Trace Comparison UI to find
                  differences, debug regressions, and inform your next version.
                </p>
              </div>
              <FakeImage />
              {/* Product GIF of the eval results compare UI, along with opening the trace compare UI */}
            </GridItem>
            <GridItem width="wide" direction="reverse">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">CI/CD support</h3>
                <p className="text-gray-600 text-lg">
                  Automatically run evaluations in CI/CD workflows to
                  systematically validate that every change improves quality,
                  preventing regressions.
                </p>
              </div>
              <FakeImage />
              {/* Animation of running eval in CI/CD workflow?? */}
            </GridItem>
          </Grid>
        </div>
        <GetStartedWithMLflow />
        <SocialWidget variant="red" />
      </div>
    </Layout>
  );
}
