import GithubIcon from "@site/static/img/social/github.svg";
import YoutubeIcon from "@site/static/img/social/youtube.svg";
import BookIcon from "@site/static/img/social/book.svg";
import LinkedinIcon from "@site/static/img/social/linkedin.svg";

import { SectionLabel } from "../SectionLabel/SectionLabel";
import { SocialWidgetItem } from "../SocialWidgetItem/SocialWidgetItem";
import { Grid, GridItem } from "../Grid/Grid";
import { Heading } from "../Typography/Heading";
import { Body } from "../Typography/Body";

const socials = [
  {
    key: "github",
    icon: <GithubIcon />,
    label: "GitHub",
    description: "20k followers",
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
  {
    key: "docs",
    icon: (
      <div>
        <BookIcon />
      </div>
    ),
    label: "Documentation",
    description: "Read documentation",
    href: "https://mlflow.org/docs/latest/",
  },
];

export const SocialWidget = () => {
  return (
    <div className="flex flex-col w-full gap-16">
      <div className="flex flex-col w-full gap-6 items-center justify-center text-center">
        <SectionLabel label="GET INVOLVED" />
        <Heading level={2}>Connect with the community</Heading>
        <Body size="l">Connect with thousands of customers using MLflow</Body>
      </div>
      <Grid>
        {socials.map((social) => (
          <GridItem key={social.key} className="p-0">
            <SocialWidgetItem
              href={social.href}
              icon={social.icon}
              label={social.label}
              description={social.description}
            />
          </GridItem>
        ))}
      </Grid>
    </div>
  );
};
