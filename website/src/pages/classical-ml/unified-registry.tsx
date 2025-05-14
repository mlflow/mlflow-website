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

export default function UnifiedRegistry() {
  return (
    <Layout variant="blue">
      <div
        className="flex flex-col bg-[linear-gradient(to_top,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-3.png')]
 bg-center bg-no-repeat bg-cover w-full pt-42 pb-20 py-20 bg-size-[100%_340px]"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-container">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <SectionLabel color="green" label="UNIFIED REGISTRY" />
            <h1 className="text-center text-wrap">
              Centralized Model Governance and Discovery
            </h1>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white/60">
              Streamline your ML workflows with MLflow's comprehensive model
              registry for version control, approvals, and deployment management
            </p>
            <GetStartedButton />
          </div>
          <div className="w-[800px] h-[450px] bg-black rounded-lg mx-auto"></div>
        </div>
      </div>
      <div className="flex flex-col gap-40 w-full px-6 md:px-20 max-w-container">
        <Grid>
          <GridRow>
            <GridItem className="py-10 pr-0 md:pr-10 gap-10">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  Seamless Unity Catalog Integration
                </h3>
                <p className="text-white/60 text-lg">
                  MLflow Model Registry integrates directly with Unity Catalog
                  to provide enterprise-grade governance across your entire ML
                  asset portfolio. Apply consistent security policies, lineage
                  tracking, and access controls to both data and models through
                  a unified permission system.
                </p>
              </div>
              <FakeImage />
            </GridItem>
            <GridItem className="py-10 pl-0 md:pl-10 gap-10">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Stage-Based Model Lifecycle</h3>
                <p className="text-white/60 text-lg">
                  Move models through customizable staging environments
                  (Development, Staging, Production, or any stage alias you
                  choose) with built-in approval workflow capabilities and
                  automated notifications. Maintain complete audit trails of
                  model transitions with detailed metadata about who approved
                  changes and when they occurred.
                </p>
              </div>
              <FakeImage />
            </GridItem>
          </GridRow>
          <GridRow>
            <GridItem className="flex flex-col md:flex-row gap-6 md:gap-20 py-10 justify-between items-center">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Model Deployment Flexibility</h3>
                <p className="text-white/60 text-lg">
                  Deploy models as containers, batch jobs, or REST endpoints
                  with MLflow's streamlined deployment capabilities that
                  eliminate boilerplate code. Use model aliases to create named
                  references that enable seamless model updates in production
                  without changing your application code.
                </p>
              </div>
              <FakeImage />
            </GridItem>
          </GridRow>
        </Grid>
        <GetStartedWithMLflow variant="blue" />
        <SocialWidget variant="red" />
      </div>
    </Layout>
  );
}
