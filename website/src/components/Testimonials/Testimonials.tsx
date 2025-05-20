import { SectionLabel } from "../SectionLabel/SectionLabel";
import { Grid, GridItem } from "../Grid/Grid";

export const Testimonials = () => {
  return (
    <div className="flex flex-col gap-16">
      <div className="flex flex-col gap-6 items-center">
        <SectionLabel color="red" label="Customers" />
        <h1 className="max-w-2xl text-center text-wrap">
          Trusted by thousands of businesses and research teams
        </h1>
      </div>
      <Grid>
        <GridItem className="p-6">
          <span className="text-white/60 text-lg">
            Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat ex non
            sit. Lorem eu proident elit Lorem tempor ea id aute dolore Lorem
            labore cupidatat.
          </span>
        </GridItem>
        <GridItem className="p-6">
          <span className="text-white/60 text-lg">
            Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat ex non
            sit. Lorem eu proident elit Lorem tempor ea id aute dolore Lorem
            labore cupidatat.
          </span>
        </GridItem>
        <GridItem className="p-6">
          <span className="text-white/60 text-lg">
            Ex aliquip commodo irure. Cupidatat veniam commodo cupidatat ex non
            sit. Lorem eu proident elit Lorem tempor ea id aute dolore Lorem
            labore cupidatat.
          </span>
        </GridItem>
      </Grid>
    </div>
  );
};
