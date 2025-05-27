import {
  Layout,
  Grid,
  GridItem,
  GetStartedWithMLflow,
  SocialWidget,
  GetStartedButton,
  Body,
  AboveTheFold,
} from "../../components";

const FakeImage = () => (
  <div className="w-full aspect-[3/2] bg-black rounded-lg border border-[rgba(255,255,255,0.08)]"></div>
);

export default function PromptRegistryVersioning() {
  return (
    <Layout variant="red" direction="up">
      <AboveTheFold
        sectionLabel="Prompt registry & versioning"
        title="Prompt registry & versioning"
        body="Lorem ipsum"
      >
        <div className="flex flex-col gap-16 items-center -mt-10">
          <GetStartedButton />
          <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
        </div>
      </AboveTheFold>
      <div className="flex flex-col gap-40 w-full px-6 md:px-20 max-w-container">
        <Grid columns={2}>
          <GridItem>
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Card 1</h3>
              <Body size="l">Lorem ipsum</Body>
            </div>
            <FakeImage />
          </GridItem>
          <GridItem>
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Card 2</h3>
              <Body size="l">Lorem ipsum</Body>
            </div>
            <FakeImage />
          </GridItem>
        </Grid>

        <GetStartedWithMLflow />
        <SocialWidget />
      </div>
    </Layout>
  );
}
