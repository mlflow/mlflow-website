import {
  Layout,
  SectionLabel,
  Button,
  Grid,
  GridRow,
  GridItem,
  GetStartedWithMLflow,
  SocialWidget,
} from "../../components";

const FakeImage = () => (
  <div className="w-[600px] h-[400px] bg-black rounded-lg"></div>
);

export default function Evaluations() {
  return (
    <Layout variant="red">
      <div
        className="flex flex-col bg-[linear-gradient(to_top,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-2.png')]
 bg-center bg-no-repeat bg-cover w-full pt-42 pb-20 py-20 bg-size-[100%_340px]"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-7xl mx-auto">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <SectionLabel color="red" label="EVALUATIONS" />
            <h1 className="text-center text-wrap">
              Adapt to evolving user behavior with production log evaluation
              (evaluation datasets)
            </h1>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white/60">
              Use production logs to understand user behavior, turning
              low-quality responses into evaluation datasets and high-quality
              responses into regression tests.
            </p>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white/60">
              Replay these datasets to evaluate new prompts or app variants in
              development so you can ship the best variants to ensure your
              application continues to deliver high-quality responses as user
              behavior evolves.
            </p>
            <Button>Get Started</Button>
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
                  <h3 className="text-white">Evaluation datasets</h3>
                </div>

                <p className="text-white/60">
                  Curate top-scoring traces into regression datasets and
                  low-scoring traces needing improvement into evaluation
                  datasets. Manually add additional examples. Datasets track
                  lineage and version history.
                </p>
              </div>
              <FakeImage />
            </GridItem>
          </GridRow>
          <GridRow>
            <GridItem className="py-10 pr-0 md:pr-10 gap-10">
              <FakeImage />
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Test new app / prompt variants</h3>
                <p className="text-white/60 text-lg">
                  Use MLflow’s Evaluation SDK to test new variants (prompts,
                  models, code changes, etc) against the evaluation and
                  regression datasets. Every variant is linked to its evaluation
                  results so you can track improvements over time.
                </p>
              </div>
            </GridItem>
            <GridItem className="py-10 pl-0 md:pl-10 gap-10">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Intuitive evaluation review UIs</h3>
                <p className="text-white/60 text-lg">
                  Review evaluation results using MLflow’s evaluation UI that
                  visualizes record-by-record results, compares diffs between
                  variants, and provides judge-driven insights into root causes
                  to validate changes and identify further improvement
                  opportunities.
                </p>
                <p className="text-white/60 text-lg">
                  Use MLflow’s metric charts to understand the tradeoff between
                  cost/latency and quality.
                </p>
              </div>
              <FakeImage />
            </GridItem>
          </GridRow>
          <GridRow>
            <GridItem className="py-10 pr-0 md:pr-10 gap-10">
              <FakeImage />
              <div className="flex flex-col gap-4">
                <h3 className="text-white">CI/CD support</h3>
                <p className="text-white/60 text-lg">
                  Automatically run evaluations in CI/CD workflows so you can
                  systematically validate every PR drives quality improvements,
                  not quality regressions.
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
