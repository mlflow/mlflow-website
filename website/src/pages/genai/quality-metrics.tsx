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

export default function QualityMetrics() {
  return (
    <Layout variant="red">
      <div
        className="flex flex-col bg-[linear-gradient(to_top,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-2.png')]
 bg-center bg-no-repeat bg-cover w-full pt-42 pb-20 py-20 bg-size-[100%_340px]"
      >
        <div className="flex flex-col gap-16 w-full max-w-7xl mx-auto px-6 md:px-20">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <SectionLabel color="red" label="QUALITY METRICS" />
            <h1 className="text-center text-wrap">
              Measure and improve quality with human-aligned, automated metrics
              (LLM judges)
            </h1>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white/60">
              Capture and convert expert feedback into metrics (LLM judges) that
              understand your business requirements and can measure the nuances
              of plain-language GenAI outputs.
            </p>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white/60">
              Use these metrics to evaluate, monitor, and improve quality in
              development and production at scale, without waiting for human
              review.
            </p>
            <GetStartedButton />
          </div>
          <div className="w-[800px] h-[450px] bg-black rounded-lg mx-auto"></div>
        </div>
      </div>
      <div className="flex flex-col gap-40 w-full max-w-7xl mx-auto px-6 md:px-20">
        <Grid>
          <GridRow>
            <GridItem className="flex flex-col md:flex-row gap-6 md:gap-20 py-10 justify-between items-center">
              <div className="flex flex-col gap-10">
                <div className="flex flex-col gap-4">
                  <h3 className="text-white">
                    Define once—use in dev and production
                  </h3>
                  <p className="text-white/60">
                    Define once—use in dev and production
                  </p>
                </div>

                <div className="flex flex-col pl-4 border-l border-white/8 gap-2">
                  <span className="text-white text-sm">DEVELOPMENT</span>
                  <span className="text-white/60">
                    Evaluate every new variant offline so you can drive
                    iterative improvements in quality and verify that changes
                    don’t cause regressions
                  </span>
                </div>
                <div className="flex flex-col pl-4 border-l border-white/8 gap-2">
                  <span className="text-white text-sm">PRODUCTION</span>
                  <span className="text-white/60">
                    Evaluate every live response and set alerts, giving you
                    always-on monitoring of production quality.
                  </span>
                </div>
              </div>
              <FakeImage />
            </GridItem>
          </GridRow>
          <GridRow>
            <GridItem className="py-10 pr-0 md:pr-10 gap-10">
              <FakeImage />
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Best-in-class judges—ready to go</h3>
                <p className="text-white/60 text-lg">
                  Get started quickly with out-of-the-box judges for safety,
                  hallucination, retrieval quality, relevance, and other common
                  aspects of quality evaluation. Our research team has tuned
                  these judges to agree with human experts, giving you accurate,
                  reliable quality evaluation.
                </p>
              </div>
            </GridItem>
            <GridItem className="py-10 pl-0 md:pl-10 gap-10">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  Customized LLM judges for your use case
                </h3>
                <p className="text-white/60 text-lg">
                  Adapt our base judge model to create custom judges tailored to
                  your business requirements that agree with your human experts’
                  judgment.
                </p>
              </div>
              <FakeImage />
            </GridItem>
          </GridRow>
          <GridRow>
            <GridItem className="flex flex-col md:flex-row gap-6 md:gap-20 py-10 justify-between items-center">
              <FakeImage />
              <div className="flex flex-col gap-10">
                <div className="flex flex-col gap-4">
                  <h3 className="text-white">Customized code-based metrics</h3>
                  <p className="text-white/60">
                    Further customize evaluation to measure any aspect of your
                    application’s quality using our custom metrics SDK to write
                    Python functions that track any metric, from regex checks to
                    custom business logic.
                  </p>
                </div>
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
