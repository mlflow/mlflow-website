import GithubIcon from "@site/static/img/social/github.svg";
import YoutubeIcon from "@site/static/img/social/youtube.svg";
import BookIcon from "@site/static/img/social/book.svg";
import LinkedinIcon from "@site/static/img/social/linkedin.svg";

import { SectionLabel } from "../Section/SectionLabel";
import { SocialWidgetItem } from "../SocialWidgetItem/SocialWidgetItem";
import { Grid } from "../Grid/Grid";
import { Heading } from "../Typography/Heading";
import { Body } from "../Typography/Body";
import { MLFLOW_DOCS_URL } from "@site/src/constants";

const socials = [
  {
    key: "docs",
    icon: (
      <div>
        <BookIcon />
      </div>
    ),
    label: "Documentation",
    description: "Read documentation",
    href: MLFLOW_DOCS_URL,
  },
  {
    key: "github",
    icon: <GithubIcon />,
    label: "GitHub",
    description: "20k stars",
    href: "https://github.com/mlflow/mlflow",
  },
  {
    key: "linkedin",
    icon: <LinkedinIcon />,
    label: "LinkedIn",
    description: "69k followers",
    href: "https://www.linkedin.com/company/mlflow-org",
  },
  {
    key: "youtube",
    icon: <YoutubeIcon />,
    label: "YouTube",
    description: "View tutorials",
    href: "https://www.youtube.com/@mlflowoss",
  },
];

export const SocialWidget = () => {
  return (
    <div className="flex flex-col w-full gap-16">
      <div className="flex flex-col w-full gap-6 items-center justify-center text-center">
        <SectionLabel label="GET INVOLVED" />
        <Heading level={2}>Connect with the open source community</Heading>
        <Body size="l">Join millions of MLflow users</Body>
      </div>
      <Grid className="px-10">
        {socials.map((social) => (
          <SocialWidgetItem
            key={social.key}
            href={social.href}
            icon={social.icon}
            label={social.label}
            description={social.description}
          />
        ))}
      </Grid>
    </div>
  );
};
