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
import CardHero from "@site/static/img/GenAI_evaluations/GenAI_evaluations_hero.png";
import Card1 from "@site/static/img/GenAI_evaluations/GenAI_evaluations_1.png";
import Card2 from "@site/static/img/GenAI_evaluations/GenAI_evaluations_2.png";
import Card3 from "@site/static/img/GenAI_evaluations/GenAI_evaluations_3.png";
import Card4 from "@site/static/img/GenAI_evaluations/GenAI_evaluations_4.png";
import Card5 from "@site/static/img/GenAI_evaluations/GenAI_evaluations_5.png";
import Card6 from "@site/static/img/GenAI_evaluations/GenAI_evaluations_6.png";
import Card7 from "@site/static/img/GenAI_evaluations/GenAI_evaluations_7.png";
import Card8 from "@site/static/img/GenAI_evaluations/GenAI_evaluations_8.png";

export default function Evaluations() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="Evaluations"
        title="Evaluation to measure and improve quality"
        body="Confidently evaluate quality in development and production to identify issues and iteratively test improvements."
        hasGetStartedButton
      >
        <HeroImage src={CardHero} alt="" />
      </AboveTheFold>

      <Section
        id="quality-metrics"
        title="Accurately evaluate free-form language outputs with LLM judges"
      >
        <Grid columns={2}>
          <GridItem width="wide">
            <Card
              title="Pre-built LLM judges"
              body="Quickly start with built-in LLM judges for safety, hallucination, retrieval quality, and relevance. Our research-backed judges provide accurate, reliable quality evaluation aligned with human expertise."
              image={<img src={Card1} alt="" />}
              // Visual rendering of the built in judges with an image of our quality being the best in the middle (grab image from blog)
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Customized LLM judges"
              body="Adapt our base model to create custom LLM judges tailored to your business needs, aligning with your human expert's judgment."
              image={<img src={Card2} alt="" />}
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
              image={<img src={Card3} alt="" />}
              // Show selecting traces from monitoring UI and saving into eval set
            />
          </GridItem>
        </Grid>
      </Section>

      <Section title="Iteratively improve quality through evaluation">
        <Grid columns={2}>
          <GridItem width="wide">
            <Card
              title="Test new app / prompt variants"
              body="MLflow's Evaluation SDK lets you test new application variants (prompts, models, code) against evaluation and regression datasets. Each variant is linked to its evaluation results, enabling tracking of improvements over time."
              image={<img src={Card4} alt="" />}
              // Product GIF of running mlflow.evaluate and then seeing the eval results list UI
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Customize with code-based metrics"
              body="Customize evaluation to measure any aspect of your app's quality or performance using our custom metrics SDK. Convert any Python function—from regex to custom logic—into a metric."
              image={<img src={Card5} alt="" />}
              // Code snippet of a custom metric function
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Identify root causes with evaluation review UIs"
              body="Use MLflow's Evaluation UI to visualize a summary of your evals and view results record-by-record to quickly identify root causes and further improvement opportunities."
              image={<img src={Card6} alt="" />}
              // Product GIF of the evaluation UI and using it to filter and view individual results
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="Compare versions side-by-side"
              body="Compare evaluations of 2 app variants to understand if your changes improved or regressed quality. Review individual questions side-by-side in the Trace Comparison UI to find differences, debug regressions, and inform your next version."
              image={<img src={Card7} alt="" />}
              // Product GIF of the eval results compare UI, along with opening the trace compare UI
            />
          </GridItem>
          <GridItem width="wide">
            <Card
              title="CI/CD support"
              body="Automatically run evaluations in CI/CD workflows to systematically validate that every change improves quality, preventing regressions."
              image={<img src={Card8} alt="" />}
              // Animation of running eval in CI/CD workflow??
            />
          </GridItem>
        </Grid>
      </Section>

      <BelowTheFold />
    </Layout>
  );
}
