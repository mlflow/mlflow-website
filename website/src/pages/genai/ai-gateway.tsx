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

export default function AiGateway() {
  return (
    <Layout variant="red">
      <div
        className="flex flex-col bg-[linear-gradient(to_top,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-2.png')]
 bg-center bg-no-repeat w-full pt-42 pb-20 py-20 bg-size-[100%_340px]"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-container">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <SectionLabel color="red" label="AI GATEWAY" />
            <h1 className="text-center text-wrap">
              Build AI Systems with confidence
            </h1>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-gray-600">
              The AI developer platform to build AI applications and models with
              confidence
            </p>
            <GetStartedButton />
          </div>
          <div className="w-[800px] h-[450px] bg-black rounded-lg mx-auto"></div>
        </div>
      </div>
      <div className="flex flex-col gap-40 w-full px-6 md:px-20 max-w-container">
        <Grid columns={2}>
          <GridItem>
            <div className="flex flex-col gap-10">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">Feature heading</h3>
              </div>

              <p className="text-gray-600">
                Lorem ipsum dolor sit amet consectetur adipisicing elit.
                Quisquam, quos.
              </p>
            </div>
            <FakeImage />
          </GridItem>
          <GridItem>
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Feature heading</h3>
              <p className="text-gray-600 text-lg">
                Lorem ipsum dolor sit amet consectetur adipisicing elit.
                Quisquam, quos. your code base.
              </p>
            </div>
            <FakeImage />
          </GridItem>
          <GridItem width="wide">
            <div className="flex flex-col gap-4">
              <h3 className="text-white">Feature heading</h3>
              <p className="text-gray-600 text-lg">
                Lorem ipsum dolor sit amet consectetur adipisicing elit.
                Quisquam, quos.
              </p>
            </div>
            <FakeImage />
          </GridItem>
        </Grid>
        <GetStartedWithMLflow />
        <SocialWidget variant="red" />
      </div>
    </Layout>
  );
}
