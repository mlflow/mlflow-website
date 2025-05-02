import useDocusaurusContext from "@docusaurus/useDocusaurusContext";

import {
  Button,
  SectionLabel,
  CopyCommand,
  LogosCarousel,
  Grid,
  GridRow,
  GridItem,
  VerticalTabs,
  VerticalTabsList,
  VerticalTabsTrigger,
  VerticalTabsContent,
} from "../components";

export default function Home(): JSX.Element {
  const { siteConfig } = useDocusaurusContext();

  return (
    <div className="flex flex-col items-center justify-center bg-[#0E1416] gap-8">
      <h1 className="text-4xl font-bold">New homepage</h1>

      <div className="flex flex-col gap-2 mb-50 px-20 w-full">
        <h3>Grid example 1</h3>
        <Grid>
          <GridRow>
            <GridItem>
              <Button variant="primary" size="small">
                Small Primary Button
              </Button>
            </GridItem>
            <GridItem>
              <Button variant="primary" size="medium">
                Medium Primary Button
              </Button>
            </GridItem>
            <GridItem>
              <Button variant="primary" size="large">
                Large Primary Button
              </Button>
            </GridItem>
          </GridRow>
        </Grid>
      </div>
      <div className="flex flex-col gap-2 mb-50 px-20 w-full">
        <h3>Grid example 2</h3>
        <Grid>
          <GridRow>
            <GridItem>
              <Button variant="secondary" size="small">
                Small Secondary Button
              </Button>
            </GridItem>
          </GridRow>
          <GridRow>
            <GridItem>
              <Button variant="secondary" size="medium">
                Medium Secondary Button
              </Button>
            </GridItem>
            <GridItem>
              <Button variant="secondary" size="large">
                Large Secondary Button
              </Button>
            </GridItem>
          </GridRow>
          <GridRow>
            <GridItem>
              <Button variant="secondary" size="large" width="full">
                Full Width Secondary Button
              </Button>
            </GridItem>
          </GridRow>
        </Grid>
      </div>
      <div className="flex flex-col gap-2 mb-10 px-20 w-full">
        <h3>Grid example 3</h3>

        <Grid>
          <GridRow>
            <GridItem>
              <Button variant="outline" size="small">
                Small Outline Button
              </Button>
            </GridItem>
          </GridRow>
          <GridRow>
            <GridItem>
              <Button variant="outline" size="medium">
                Medium Outline Button
              </Button>
            </GridItem>
            <GridItem>
              <Button variant="outline" size="large">
                Large Outline Button
              </Button>
            </GridItem>
          </GridRow>
          <GridRow>
            <GridItem>
              <Button variant="outline" size="large" width="full">
                Full Width Outline Button
              </Button>
            </GridItem>
            <GridItem>
              <Button variant="outline" size="large" width="full">
                Full Width Outline Button
              </Button>
            </GridItem>
          </GridRow>
        </Grid>
      </div>

      <div className="flex flex-row gap-12 w-full justify-center">
        <div className="flex flex-col gap-5 min-w-[300px]">
          <SectionLabel label="Core Features" color="red" />
          <SectionLabel label="Core Features" color="green" />
        </div>
      </div>
      <CopyCommand code="pip install mlflow" />
      <LogosCarousel>
        <img
          className="mx-4 inline h-16"
          src="https://tailwindcss.com/plus-assets/img/logos/158x48/transistor-logo-white.svg"
          alt="Transistor"
        />
        <img
          className="mx-4 inline h-16"
          src="https://tailwindcss.com/plus-assets/img/logos/158x48/reform-logo-white.svg"
          alt="Reform"
        />
        <img
          className="mx-4 inline h-16"
          src="https://tailwindcss.com/plus-assets/img/logos/158x48/tuple-logo-white.svg"
          alt="Tuple"
        />
        <img
          className="mx-4 inline h-16"
          src="https://tailwindcss.com/plus-assets/img/logos/158x48/savvycal-logo-white.svg"
          alt="SavvyCal"
        />
        <img
          className="mx-4 inline h-16"
          src="https://tailwindcss.com/plus-assets/img/logos/158x48/statamic-logo-white.svg"
          alt="Statamic"
        />
        <img
          className="mx-4 inline h-16"
          src="https://tailwindcss.com/plus-assets/img/logos/158x48/reform-logo-white.svg"
          alt="Reform"
        />
        <img
          className="mx-4 inline h-16"
          src="https://tailwindcss.com/plus-assets/img/logos/158x48/tuple-logo-white.svg"
          alt="Tuple"
        />
        <img
          className="mx-4 inline h-16"
          src="https://tailwindcss.com/plus-assets/img/logos/158x48/savvycal-logo-white.svg"
          alt="SavvyCal"
        />
      </LogosCarousel>
      <VerticalTabs defaultValue="tab1" className="w-full my-12 px-10">
        <VerticalTabsList>
          <VerticalTabsTrigger
            value="tab1"
            label="LLM Judges"
            description="Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat ex non sit lorem eu proident."
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
  );
}
