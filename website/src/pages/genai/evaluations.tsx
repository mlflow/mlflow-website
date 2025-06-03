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

export default function Evaluations() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="Evaluations"
        title="Evaluation to measure and improve quality"
        body="Confidently evaluate quality in development and production to identify issues and iteratively test improvements."
        hasGetStartedButton
      >
        <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
      </AboveTheFold>

      <Section
        id="quality-metrics"
        title="Accurately evaluate free-form language outputs with LLM judges"
      >
        <Grid columns={2}>
          <GridItem>
            <Card
              title="Pre-built LLM judges"
              body="Quickly start with built-in LLM judges for safety, hallucination, retrieval quality, and relevance. Our research-backed judges provide accurate, reliable quality evaluation aligned with human expertise."
              image={<FakeImage />}
              // Visual rendering of the built in judges with an image of our quality being the best in the middle (grab image from blog)
            />
          </GridItem>
          <GridItem direction="reverse">
            <Card
              title="Customized LLM judges"
              body="Adapt our base model to create custom LLM judges tailored to your business needs, aligning with your human expert's judgment."
              image={<FakeImage />}
              // Animation showing humans providing feedback and it being synthesized into an llm judge
            />
          </GridItem>
        </Grid>
      </Section>

      <Section title="Use production traffic to drive offline improvements">
        <Grid columns={2}>
          <GridItem width="wide">
            <Card
              title="Evaluation datasets"
              body="Curate high-scoring traces for regression datasets and low-scoring ones for evaluation datasets to use offline to improve quality."
              image={<FakeImage />}
              // Show selecting traces from monitoring UI and saving into eval set
            />
          </GridItem>
        </Grid>
      </Section>

      <Section title="Iteratively improve quality through evaluation">
        <Grid columns={2}>
          <GridItem width="wide" direction="reverse">
            <Card
              title="Test new app / prompt variants"
              body="MLflow's Evaluation SDK lets you test new application variants (prompts, models, code) against evaluation and regression datasets. Each variant is linked to its evaluation results, enabling tracking of improvements over time."
              image={<FakeImage />}
              // Product GIF of running mlflow.evaluate and then seeing the eval results list UI
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Customize with code-based metrics"
              body="Customize evaluation to measure any aspect of your app's quality or performance using our custom metrics SDK. Convert any Python function—from regex to custom logic—into a metric."
              image={<FakeImage />}
              // Code snippet of a custom metric function
            />
          </GridItem>
          <GridItem width="wide" direction="reverse">
            <Card
              title="Identify root causes with evaluation review UIs"
              body="Use MLflow's Evaluation UI to visualize a summary of your evals and view results record-by-record to quickly identify root causes and further improvement opportunities."
              image={<FakeImage />}
              // Product GIF of the evaluation UI and using it to filter and view individual results
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Compare versions side-by-side"
              body="Compare evaluations of 2 app variants to understand if your changes improved or regressed quality. Review individual questions side-by-side in the Trace Comparison UI to find differences, debug regressions, and inform your next version."
              image={<FakeImage />}
              // Product GIF of the eval results compare UI, along with opening the trace compare UI
            />
          </GridItem>
          <GridItem width="wide" direction="reverse">
            <Card
              title="CI/CD support"
              body="Automatically run evaluations in CI/CD workflows to systematically validate that every change improves quality, preventing regressions."
              image={<FakeImage />}
              // Animation of running eval in CI/CD workflow??
            />
          </GridItem>
        </Grid>
      </Section>

      <BelowTheFold />
    </Layout>
  );
}
