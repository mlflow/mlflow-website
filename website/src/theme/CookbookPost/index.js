import React from "react";
import {
  BlogPostProvider,
  useBlogPost,
} from "@docusaurus/plugin-content-blog/client";
import BlogPostItem from "@theme/BlogPostItem";
import TOC from "@theme/TOC";
import CookbookLayout from "@site/src/theme/CookbookLayout";
import Link from "@docusaurus/Link";
import { Grid } from "../../components";
import { SocialWidgetItem } from "../../components/SocialWidgetItem/SocialWidgetItem";
import { useGitHubStars } from "../../hooks/useGitHubStars";
import { MLFLOW_DOCS_URL } from "@site/src/constants";
import GithubIcon from "@site/static/img/social/github.svg";
import YoutubeIcon from "@site/static/img/social/youtube.svg";
import BookIcon from "@site/static/img/social/book.svg";
import LinkedinIcon from "@site/static/img/social/linkedin.svg";
import XIcon from "@site/static/img/social/x.svg";
import SlackIcon from "@site/static/img/social/slack.svg";

const socials = [
  {
    key: "docs",
    icon: (
      <div>
        <BookIcon />
      </div>
    ),
    label: "Documentation",
    description: "Read Docs",
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
  {
    key: "x",
    icon: <XIcon />,
    label: "X",
    description: "Follow us on X",
    href: "https://x.com/mlflow",
  },
  {
    key: "slack",
    icon: <SlackIcon />,
    label: "Slack",
    description: "Join our Slack",
    href: "https://go.mlflow.org/slack",
  },
];

// Ordered sequences for grouped cookbooks. Pages within a group
// navigate through the group instead of using chronological order.
const COOKBOOK_SEQUENCES = [
  [
    "/cookbook/databricks-genie",
    "/cookbook/genie-tracing-pipeline",
    "/cookbook/genie-evaluation-judges",
    "/cookbook/genie-space-analyzer",
  ],
];

// Build a lookup: permalink -> { prev, next } within its sequence.
const SEQ_NAV = {};
for (const seq of COOKBOOK_SEQUENCES) {
  for (let i = 0; i < seq.length; i++) {
    SEQ_NAV[seq[i]] = {
      prev: i > 0 ? seq[i - 1] : null,
      next: i < seq.length - 1 ? seq[i + 1] : null,
    };
  }
}

function CookbookTags({ tags }) {
  if (!tags?.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5 mb-4">
      {tags.map((tag) => (
        <Link
          key={tag.permalink}
          href={tag.permalink}
          className="rounded-md bg-white/8 px-2.5 py-1 text-xs !text-white/50 hover:!text-white hover:bg-white/12 transition-colors"
        >
          {tag.label}
        </Link>
      ))}
    </div>
  );
}

function CookbookPaginator({ prevItem, nextItem }) {
  if (!prevItem && !nextItem) return null;
  return (
    <nav
      className="flex justify-between items-center mt-12 pt-8"
      style={{ borderTop: "1px solid rgba(255, 255, 255, 0.1)" }}
    >
      <div>
        {prevItem && (
          <Link
            href={prevItem.permalink}
            className="!text-white/60 hover:!text-white transition-colors text-lg font-semibold"
          >
            {"<"} {prevItem.title}
          </Link>
        )}
      </div>
      <div>
        {nextItem && (
          <Link
            href={nextItem.permalink}
            className="!text-white/60 hover:!text-white transition-colors text-lg font-semibold"
          >
            {nextItem.title} {">"}
          </Link>
        )}
      </div>
    </nav>
  );
}

// Build a flat ordered list from the sidebar, respecting group order.
// Then look up prev/next by position in that list.
function resolveNav(permalink, sidebar) {
  const items = sidebar?.items || [];
  if (!items.length) return { prev: null, next: null };

  // Collect all child permalinks so we can skip them at top level.
  const childSet = new Set();
  for (const seq of COOKBOOK_SEQUENCES) {
    for (let i = 1; i < seq.length; i++) {
      childSet.add(seq[i]);
    }
  }

  // Separate ungrouped top-level items from group parents so that
  // groups appear at the end, matching the sidebar render order.
  const topLevel = items.filter((it) => !childSet.has(it.permalink));
  const ungrouped = topLevel.filter(
    (it) => !COOKBOOK_SEQUENCES.some((seq) => seq[0] === it.permalink),
  );
  const groupParents = topLevel.filter((it) =>
    COOKBOOK_SEQUENCES.some((seq) => seq[0] === it.permalink),
  );

  // Build the flat ordered list: ungrouped first, then each group
  // parent followed by its children.
  const ordered = [...ungrouped];
  for (const parent of groupParents) {
    ordered.push(parent);
    for (const seq of COOKBOOK_SEQUENCES) {
      if (seq[0] === parent.permalink) {
        for (let i = 1; i < seq.length; i++) {
          const child = items.find((it) => it.permalink === seq[i]);
          if (child) ordered.push(child);
        }
      }
    }
  }

  const idx = ordered.findIndex((it) => it.permalink === permalink);
  if (idx === -1) return { prev: null, next: null };

  const prev =
    idx > 0
      ? { permalink: ordered[idx - 1].permalink, title: ordered[idx - 1].title }
      : null;
  const next =
    idx < ordered.length - 1
      ? { permalink: ordered[idx + 1].permalink, title: ordered[idx + 1].title }
      : null;

  return { prev, next };
}

function SocialTiles() {
  const stars = useGitHubStars();
  return (
    <div className="mt-16">
      <Grid>
        {socials.map((social) => (
          <SocialWidgetItem
            key={social.key}
            href={social.href}
            icon={social.icon}
            label={social.label}
            description={
              social.key === "github" && stars
                ? `${stars}+ stars`
                : social.description
            }
          />
        ))}
      </Grid>
    </div>
  );
}

function CookbookPostContent({
  sidebar,
  children,
  BlogPostContent,
  layoutProps,
}) {
  const { metadata, toc } = useBlogPost();
  const { prev, next } = resolveNav(metadata.permalink, sidebar);
  const { hide_table_of_contents: hideTableOfContents } =
    metadata.frontMatter || {};

  const tocElement =
    !hideTableOfContents && toc.length > 0 ? <TOC toc={toc} /> : undefined;

  return (
    <CookbookLayout
      sidebar={sidebar}
      toc={tocElement}
      footer={<SocialTiles />}
      {...layoutProps}
    >
      <BlogPostItem className="max-w-none">
        <CookbookTags tags={metadata.tags} />
        <BlogPostContent />
        <CookbookPaginator prevItem={prev} nextItem={next} />
      </BlogPostItem>
    </CookbookLayout>
  );
}

export default function CookbookPost({
  content: BlogPostContent,
  sidebar,
  ...layoutProps
}) {
  return (
    <BlogPostProvider content={BlogPostContent} isBlogPostPage>
      <CookbookPostContent
        sidebar={sidebar}
        BlogPostContent={BlogPostContent}
        layoutProps={layoutProps}
      />
    </BlogPostProvider>
  );
}
