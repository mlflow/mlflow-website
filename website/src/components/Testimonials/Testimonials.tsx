import { SectionLabel } from "../SectionLabel/SectionLabel";
import { Grid, GridItem } from "../Grid/Grid";
import { Heading } from "../Typography/Heading";

export const Testimonials = () => {
  return (
    <div className="flex flex-col gap-16">
      <div className="flex flex-col gap-6 items-center">
        <SectionLabel label="Customers" />
        <Heading level={2}>
          Trusted by thousands of businesses and research teams
        </Heading>
      </div>
      <Grid>
        <GridItem className="p-6">
          <span className="text-gray-600 text-lg">
            Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat ex non
            sit. Lorem eu proident elit Lorem tempor ea id aute dolore Lorem
            labore cupidatat.
          </span>
        </GridItem>
        <GridItem className="p-6">
          <span className="text-gray-600 text-lg">
            Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat ex non
            sit. Lorem eu proident elit Lorem tempor ea id aute dolore Lorem
            labore cupidatat.
          </span>
        </GridItem>
        <GridItem className="p-6">
          <span className="text-gray-600 text-lg">
            Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat ex non
            sit. Lorem eu proident elit Lorem tempor ea id aute dolore Lorem
            labore cupidatat.
          </span>
        </GridItem>
      </Grid>
    </div>
  );
};
