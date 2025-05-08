import {
  Layout,
  SectionLabel,
  Button,
  Grid,
  GridRow,
  GridItem,
  GetStartedWithMLflow,
  SocialWidget,
} from "../components";

const FakeImage = () => (
  <div className="w-[600px] h-[400px] bg-black rounded-lg"></div>
);

export default function QualityMetrics() {
  return (
    <Layout variant="blue">
      <div
        className="flex flex-col bg-[linear-gradient(to_top,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-3.png')]
 bg-center bg-no-repeat bg-cover w-full pt-42 pb-20 py-20 bg-size-[100%_340px]"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <SectionLabel color="green" label="TRACKING" />
            <h1 className="text-center text-wrap">
              Lorem ipsum dolor sit amet
            </h1>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto">
              Cupidatat veniam commodo cupidatat ex non sit.
            </p>
            <Button>Get Started</Button>
          </div>
          <div className="w-[800px] h-[450px] bg-black rounded-lg mx-auto"></div>
        </div>
      </div>
      <div className="flex flex-col gap-40 w-full px-6 md:px-20">
        <Grid>
          <GridRow>
            <GridItem className="py-10 pr-0 md:pr-10 gap-10">
              <div className="flex flex-col gap-4">
                <h3>Lorem ipsum dolor sit amet</h3>
                <p className="text-white/60 text-lg">
                  Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat
                  ex non sit. Lorem eu proident elit Lorem tempor ea id aute
                  dolore Lorem labore cupidatat. Ex aliquip commodo irure.
                </p>
              </div>
              <FakeImage />
            </GridItem>
            <GridItem className="py-10 pl-0 md:pl-10 gap-10">
              <div className="flex flex-col gap-4">
                <h3>Lorem ipsum dolor sit amet</h3>
                <p className="text-white/60 text-lg">
                  Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat
                  ex non sit. Lorem eu proident elit Lorem tempor ea id aute
                  dolore Lorem labore cupidatat. Ex aliquip commodo irure.
                </p>
              </div>
              <FakeImage />
            </GridItem>
          </GridRow>
          <GridRow>
            <GridItem className="py-10 pr-0 md:pr-10 gap-10">
              <div className="flex flex-col gap-4">
                <h3>Lorem ipsum dolor sit amet</h3>
                <p className="text-white/60 text-lg">
                  Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat
                  ex non sit. Lorem eu proident elit Lorem tempor ea id aute
                  dolore Lorem labore cupidatat. Ex aliquip commodo irure.
                </p>
              </div>
              <FakeImage />
            </GridItem>
            <GridItem className="py-10 pl-0 md:pl-10 gap-10">
              <div className="flex flex-col gap-4">
                <h3>Lorem ipsum dolor sit amet</h3>
                <p className="text-white/60 text-lg">
                  Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat
                  ex non sit. Lorem eu proident elit Lorem tempor ea id aute
                  dolore Lorem labore cupidatat. Ex aliquip commodo irure.
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
