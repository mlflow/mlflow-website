import GithubIcon from "@site/static/img/social/github.svg";
import YoutubeIcon from "@site/static/img/social/youtube.svg";
import BookIcon from "@site/static/img/social/book.svg";

import { SectionLabel } from "../SectionLabel/SectionLabel";
import { SocialWidgetItem } from "../SocialWidgetItem/SocialWidgetItem";
import { Grid, GridRow, GridItem } from "../Grid/Grid";

interface Props {
  variant: "red" | "green";
}

export const SocialWidget = ({ variant }: Props) => {
  return (
    <div className="flex flex-col w-full gap-16">
      <div className="flex flex-col w-full gap-6 items-center justify-center text-center">
        <SectionLabel label="GET INVOLVED" color={variant} />
        <h1>Connect with the community</h1>
        <p className="text-white/60 text-lg">
          Connect with thousands of customers using MLflow
        </p>
      </div>
      <Grid>
        <GridRow>
          <GridItem>
            <SocialWidgetItem
              href="https://github.com/mlflow/mlflow"
              icon={<GithubIcon />}
              label="GitHub"
              description="20k followers"
            />
          </GridItem>
          <GridItem>
            <SocialWidgetItem
              href="https://www.youtube.com/@mlflowoss"
              icon={<YoutubeIcon />}
              label="YouTube"
              description="View tutorials"
            />
          </GridItem>
          <GridItem>
            <SocialWidgetItem
              href="https://mlflow.org/docs/latest/"
              icon={<BookIcon />}
              label="Docs"
              description="Read documentation"
            />
          </GridItem>
        </GridRow>
      </Grid>
    </div>
  );
};
