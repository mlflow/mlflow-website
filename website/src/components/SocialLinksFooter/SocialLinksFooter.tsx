import { useGitHubStars } from "../../hooks/useGitHubStars";
import { SocialWidgetItem } from "../SocialWidgetItem/SocialWidgetItem";
import { Grid } from "../Grid/Grid";
import GithubIcon from "@site/static/img/social/github.svg";
import YoutubeIcon from "@site/static/img/social/youtube.svg";
import BookIcon from "@site/static/img/social/book.svg";
import LinkedinIcon from "@site/static/img/social/linkedin.svg";
import XIcon from "@site/static/img/social/x.svg";
import SlackIcon from "@site/static/img/social/slack.svg";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";

export function SocialLinksFooter() {
  const gitHubStars = useGitHubStars();

  return (
    <div
      style={{
        background: "#0E1416",
        padding: "48px 40px",
      }}
    >
      <Grid className="px-10">
        <SocialWidgetItem
          href={MLFLOW_GENAI_DOCS_URL}
          icon={
            <div>
              <BookIcon />
            </div>
          }
          label="Documentation"
          description="Read Docs"
        />
        <SocialWidgetItem
          href="https://github.com/mlflow/mlflow"
          icon={<GithubIcon />}
          label="GitHub"
          description={gitHubStars ? `${gitHubStars} stars` : "Star us"}
        />
        <SocialWidgetItem
          href="https://www.linkedin.com/company/mlflow-org"
          icon={<LinkedinIcon />}
          label="LinkedIn"
          description="75k followers"
        />
        <SocialWidgetItem
          href="https://www.youtube.com/@mlflowoss"
          icon={<YoutubeIcon />}
          label="YouTube"
          description="View tutorials"
        />
        <SocialWidgetItem
          href="https://x.com/mlflow"
          icon={<XIcon />}
          label="X"
          description="Follow us on X"
        />
        <SocialWidgetItem
          href="https://go.mlflow.org/slack"
          icon={<SlackIcon />}
          label="Slack"
          description="Join our Slack"
        />
      </Grid>
    </div>
  );
}
