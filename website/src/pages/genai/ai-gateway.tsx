import {
  Layout,
  Grid,
  GridItem,
  AboveTheFold,
  BelowTheFold,
  Card,
} from "../../components";

const FakeImage = () => (
  <div className="w-full aspect-[3/2] bg-black rounded-lg border border-[rgba(255,255,255,0.08)]"></div>
);

export default function AiGateway() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="AI gateway"
        title="Build AI Systems with confidence"
        body="The AI developer platform to build AI applications and models with confidence"
        hasGetStartedButton
      >
        <div className="w-full max-w-[800px] aspect-video bg-black rounded-lg mx-auto"></div>
      </AboveTheFold>

      <Grid columns={2}>
        <GridItem>
          <Card
            title="Feature heading"
            body="Lorem ipsum dolor sit amet consectetur adipisicing elit. Quisquam, quos."
            image={<FakeImage />}
          />
        </GridItem>
        <GridItem>
          <Card
            title="Feature heading"
            body="Lorem ipsum dolor sit amet consectetur adipisicing elit. Quisquam, quos."
            image={<FakeImage />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Feature heading"
            body="Lorem ipsum dolor sit amet consectetur adipisicing elit. Quisquam, quos."
            image={<FakeImage />}
          />
        </GridItem>
      </Grid>

      <BelowTheFold />
    </Layout>
  );
}
