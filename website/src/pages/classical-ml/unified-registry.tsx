import {
  Layout,
  Grid,
  GridItem,
  Body,
  AboveTheFold,
  BelowTheFold,
} from "../../components";

const FakeImage = () => (
  <div className="w-full aspect-[3/2] bg-black rounded-lg border border-[rgba(255,255,255,0.08)]"></div>
);

export default function UnifiedRegistry() {
  return (
    <Layout variant="blue" direction="up">
      <AboveTheFold
        sectionLabel="Unified registry"
        title="Centralized Model Governance and Discovery"
        body="Streamline your ML workflows with MLflow's comprehensive model registry for version control, approvals, and deployment management"
        hasGetStartedButton
      >
        <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
      </AboveTheFold>

      <Grid columns={2}>
        <GridItem>
          <div className="flex flex-col gap-4">
            <h3 className="text-white">Seamless Unity Catalog Integration</h3>
            <Body size="l">
              MLflow Model Registry integrates directly with Unity Catalog to
              provide enterprise-grade governance across your entire ML asset
              portfolio. Apply consistent security policies, lineage tracking,
              and access controls to both data and models through a unified
              permission system.
            </Body>
          </div>
          <FakeImage />
        </GridItem>
        <GridItem>
          <div className="flex flex-col gap-4">
            <h3 className="text-white">Stage-Based Model Lifecycle</h3>
            <Body size="l">
              Move models through customizable staging environments
              (Development, Staging, Production, or any stage alias you choose)
              with built-in approval workflow capabilities and automated
              notifications. Maintain complete audit trails of model transitions
              with detailed metadata about who approved changes and when they
              occurred.
            </Body>
          </div>
          <FakeImage />
        </GridItem>
        <GridItem width="wide">
          <div className="flex flex-col gap-4">
            <h3 className="text-white">Model Deployment Flexibility</h3>
            <Body size="l">
              Deploy models as containers, batch jobs, or REST endpoints with
              MLflow's streamlined deployment capabilities that eliminate
              boilerplate code. Use model aliases to create named references
              that enable seamless model updates in production without changing
              your application code.
            </Body>
          </div>
          <FakeImage />
        </GridItem>
      </Grid>

      <BelowTheFold />
    </Layout>
  );
}
