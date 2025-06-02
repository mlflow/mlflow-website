import { Grid, GridItem, LogosCarousel, Section } from "..";

export const Testimonials = () => {
  return (
    <Section
      label="Customers"
      title="Trusted by thousands of businesses and research teams"
    >
      {/* <Grid>
        <GridItem>
          <span className="text-gray-600 text-lg">
            Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat ex non
            sit. Lorem eu proident elit Lorem tempor ea id aute dolore Lorem
            labore cupidatat.
          </span>
        </GridItem>
        <GridItem>
          <span className="text-gray-600 text-lg">
            Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat ex non
            sit. Lorem eu proident elit Lorem tempor ea id aute dolore Lorem
            labore cupidatat.
          </span>
        </GridItem>
        <GridItem>
          <span className="text-gray-600 text-lg">
            Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat ex non
            sit. Lorem eu proident elit Lorem tempor ea id aute dolore Lorem
            labore cupidatat.
          </span>
        </GridItem>
      </Grid> */}
      <LogosCarousel />
    </Section>
  );
};
