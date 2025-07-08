import { Grid, GridItem, Section, Card } from "../../components";

export const ValuePropWidget = () => {
  return (
    <Section label="Why us?" title="Why MLflow is unique">
      <Grid columns={2}>
        <GridItem>
          <Card
            title="Open, Flexible, and Extensible"
            body="Open-source and extensible, MLflow prevents vendor lock-in by integrating with the GenAI/ML ecosystem and using open protocols for data ownership, adapting to your existing and future stacks."
          />
        </GridItem>
        <GridItem>
          <Card
            title="Unified, End-to-End MLOps and AI Observability"
            body="MLflow offers a unified platform for the entire GenAI and ML model lifecycle, simplifying the experience and boosting collaboration by reducing tool integration friction."
          />
        </GridItem>
        <GridItem>
          <Card
            title="Framework neutrality"
            body="MLflow's framework-agnostic design is one of its strongest differentiators. Unlike proprietary solutions that lock you into specific ecosystems, MLflow works seamlessly with all popular ML and GenAI frameworks."
          />
        </GridItem>
        <GridItem>
          <Card
            title="Enterprise adoption"
            body="MLflow's impact extends beyond its technical capabilities. Created by Databricks, it has become one of the most widely adopted MLOps tools in the industry, with integration support from major cloud providers."
          />
        </GridItem>
      </Grid>
    </Section>
  );
};
