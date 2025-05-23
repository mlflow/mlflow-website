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

export default function PromptRegistryVersioning() {
  return (
    <Layout variant="red">
      <div
        className="flex flex-col bg-[linear-gradient(to_top,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-2.png')]
 bg-center bg-no-repeat w-full pt-42 pb-20 py-20 bg-size-[100%_340px]"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-container">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <SectionLabel color="red" label="Prompt & Version Management" />
            <h1 className="text-center text-wrap">
              Prompt & Version Management 🔨
            </h1>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white/60">
              Manage prompts and track versions of GenAI applications. Create, store, and version prompts in the Prompt Registry, and track and compare different versions of GenAI applications to ensure quality and maintainability.
            </p>

            <GetStartedButton />
          </div>
          <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
        </div>
      </div>
      <div className="flex flex-col gap-40 pt-20 w-full px-6 md:px-20 max-w-container">
        {/* Prompt Registry Section Start */}
        <div className="flex flex-col gap-10">
          <div className="flex flex-col w-full items-center justify-center gap-1">
            <div role="heading" aria-level={2} className="text-4xl">
              Prompt Registry
            </div>
          </div>
          <Grid columns={2}>
            <GridItem direction="reverse">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Prompt Registry</h3>
                <p className="text-white/60 text-lg">
                  Centrally manage your prompts with robust version control, aliasing for deployments, and lineage tracking. Create, edit, and evaluate prompts, use them in applications and deployed environments, and manage their lifecycles effectively.
                </p>
              </div>
              <FakeImage />
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h4 className="text-white">Create and Edit Prompts</h4>
                <p className="text-white/60">
                  Define prompt templates with variables, manage versions with commit messages and metadata, and compare changes.
                </p>
              </div>
              <FakeImage />
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h4 className="text-white">Use Prompts in Apps</h4>
                <p className="text-white/60">
                  Load prompts from the registry using URIs, bind variables, and integrate with frameworks like LangChain or LlamaIndex. Log prompt versions with MLflow Models for lineage.
                </p>
              </div>
              <FakeImage />
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h4 className="text-white">Evaluate Prompts</h4>
                <p className="text-white/60">
                  Set up evaluation experiments, compare different prompt versions, analyze results, and select the most effective prompts.
                </p>
              </div>
              <FakeImage />
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h4 className="text-white">Manage Prompt Lifecycles with Aliases</h4>
                <p className="text-white/60">
                  Use aliases (e.g., development, staging, production) to manage prompt versions across environments and implement governance.
                </p>
              </div>
              <FakeImage />
            </GridItem>
          </Grid>
        </div>
        {/* Prompt Registry Section End */}

        {/* Version Tracking Section Start */}
        <div className="flex flex-col gap-10">
          <div className="flex flex-col w-full items-center justify-center gap-1">
            <div role="heading" aria-level={2} className="text-4xl">
              Version Tracking
            </div>
          </div>
          <Grid columns={2}>
            <GridItem width="wide">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Version Tracking</h3>
                <p className="text-white/60 text-lg">
                  Track different versions of your GenAI applications using LoggedModels. Link evaluation results, traces, and prompt versions to specific application versions. Optionally package application code for deployment and compare versions to understand performance impacts.
                </p>
              </div>
              <FakeImage />
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h4 className="text-white">Track Application Versions with MLflow</h4>
                <p className="text-white/60">
                  Use LoggedModel as a central metadata record linking to external code (e.g., Git commits), prompt versions, and configurations. Set active models for associating evaluations and traces.
                </p>
              </div>
              <FakeImage />
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h4 className="text-white">Optionally Package App Code & Files</h4>
                <p className="text-white/60">
                  Bundle application code, dependencies, and artifacts into a LoggedModel for deployment, especially for environments like Databricks Model Serving.
                </p>
              </div>
              <FakeImage />
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h4 className="text-white">Compare App Versions</h4>
                <p className="text-white/60">
                  Compare different LoggedModel versions using metrics like performance, cost, and quality scores to make data-driven decisions.
                </p>
              </div>
              <FakeImage />
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h4 className="text-white">Link Evaluation Results and Traces to App Versions</h4>
                <p className="text-white/60">
                  Automatically link evaluation metrics, outputs, and traces from `mlflow.genai.evaluate()` and autologging back to the specific LoggedModel version.
                </p>
              </div>
              <FakeImage />
            </GridItem>
          </Grid>
        </div>
        {/* Version Tracking Section End */}

        <GetStartedWithMLflow />
        <SocialWidget variant="red" />
      </div>
    </Layout>
  );
}
