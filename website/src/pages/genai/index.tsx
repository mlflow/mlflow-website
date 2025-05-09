import {
  Layout,
  CopyCommand,
  VerticalTabs,
  VerticalTabsList,
  VerticalTabsTrigger,
  VerticalTabsContent,
  SectionLabel,
  LogosCarousel,
  Grid,
  GridRow,
  GridItem,
  GetStartedWithMLflow,
  LatestNews,
  SocialWidget,
} from "../../components";

export default function GenAi(): JSX.Element {
  return (
    <Layout variant="red">
      <div
        className="flex flex-col bg-[linear-gradient(to_bottom,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-2.png')]
 bg-top bg-no-repeat bg-cover w-full pt-42 pb-20 py-20"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <h1 className="text-center text-wrap">
              Ship high-quality GenAI, fast
            </h1>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto">
              Traditional software and ML tests aren’t built for GenAI’s
              free-form language, making it difficult for teams to measure and
              improve quality.
            </p>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto">
              MLflow combines metrics that reliably measure GenAI quality with
              trace observability so you can measure, improve, and monitor
              quality, cost, and latency.
            </p>
          </div>
          <div className="flex flex-col md:flex-row gap-10">
            <CopyCommand code="pip install <package name>" />
          </div>
        </div>
      </div>
      <div className="flex flex-col gap-40 w-full px-6 md:px-20">
        <div className="flex flex-col w-full items-center justify-center gap-16">
          <div className="flex flex-col w-full items-center justify-center gap-6">
            <SectionLabel color="red" label="CORE FEATURES" />
            <h1>Build confidently, deploy seamlessly</h1>
            <p>Tackle the challenges of building GenAI head on</p>
          </div>
          <VerticalTabs defaultValue="tab1" className="w-full my-12 px-10">
            <VerticalTabsList>
              <VerticalTabsTrigger
                value="tab1"
                label="LLM Judges"
                description="Capture and debug application logs with end-to-end observability.."
              />
              <VerticalTabsTrigger
                value="tab2"
                label="Tracing"
                description="Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat ex non sit lorem eu proident."
              />
              <VerticalTabsTrigger
                value="tab3"
                label="Evaluation Datasets"
                description="Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat ex non sit lorem eu proident."
              />
              <VerticalTabsTrigger
                value="tab4"
                label="Review app"
                description="Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat ex non sit lorem eu proident."
              />
              <VerticalTabsTrigger
                value="tab5"
                label="Enterprise-Ready Governance"
                description="Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat ex non sit lorem eu proident."
              />
            </VerticalTabsList>
            <VerticalTabsContent value="tab1">
              <img src="/img/demo-image.png" />
            </VerticalTabsContent>
            <VerticalTabsContent value="tab2">
              <img src="/img/demo-image.png" />
            </VerticalTabsContent>
            <VerticalTabsContent value="tab3">
              <img src="/img/demo-image.png" />
            </VerticalTabsContent>

            <VerticalTabsContent value="tab4">
              <img src="/img/demo-image.png" />
            </VerticalTabsContent>

            <VerticalTabsContent value="tab5">
              <img src="/img/demo-image.png" />
            </VerticalTabsContent>
          </VerticalTabs>
        </div>
        <LogosCarousel
          images={[
            "img/companies/databricks.svg",
            "img/companies/microsoft.svg",
            "img/companies/meta.svg",
            "img/companies/mosaicml.svg",
            "img/companies/zillow.svg",
            "img/companies/toyota.svg",
            "img/companies/booking.svg",
            "img/companies/wix.svg",
            "img/companies/accenture.svg",
            "img/companies/asml.svg",
          ]}
        />
        <div className="flex flex-col items-center justify-center gap-16">
          <div className="flex flex-col gap-6">
            <SectionLabel color="red" label="WHY US?" />
            <h1>Why MLflow is unique</h1>
          </div>
          <Grid>
            <GridRow>
              <GridItem className="py-10 pr-0 md:pr-10 gap-10">
                <div className="flex flex-col gap-4">
                  <h3>Built on top of a data platform</h3>
                  <p className="text-white/60">
                    Evaluation and monitoring are workflows that generate data.
                    Easily use the data from your evaluation & monitoring
                    workflows to build dashboards & apps using the features of
                    the Databricks platform.
                  </p>
                </div>
              </GridItem>
              <GridItem className="py-10 pl-0 md:pl-10 gap-10">
                <div className="flex flex-col gap-4">
                  <h3>Integrated and unified governance</h3>
                  <p className="text-white/60">
                    Tightly integrated with Unity Catalog, which offers unified,
                    enterprise-grade governance across all your data and ai
                    assets - including all assets created by MLflow.
                  </p>
                </div>
              </GridItem>
            </GridRow>
            <GridRow>
              <GridItem className="py-10 pr-0 md:pr-10 gap-10">
                <div className="flex flex-col gap-4">
                  <h3>Data intelligence</h3>
                  <p className="text-white/60">
                    Data Intelligence that helps make the developer workflow for
                    improving quality faster/more efficient - LLM judges that
                    are tuned to understand your business data; topic detection
                    & classification that help understand your users.
                  </p>
                </div>
              </GridItem>
              <GridItem className="py-10 pl-0 md:pl-10 gap-10">
                <div className="flex flex-col gap-4">
                  <h3>Secure and scalable</h3>
                  <p className="text-white/60">
                    Databricks is a trusted vendor and we host the managed
                    version - no need to self host since we are a trusted
                    vendor.
                  </p>
                </div>
              </GridItem>
            </GridRow>
          </Grid>
        </div>
        <GetStartedWithMLflow />
        <LatestNews variant="red" />
        <SocialWidget variant="red" />
      </div>
    </Layout>
  );
}
