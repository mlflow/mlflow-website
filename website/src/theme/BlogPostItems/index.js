import React from "react";
import { BlogPostProvider } from "@docusaurus/theme-common/internal";
import BlogPostItem from "@theme/BlogPostItem";
import { BLOGS } from "@site/src/posts";
import { RELEASES } from "@site/src/posts";
import Blog from "@site/src/components/Blog";
import Grid from "@site/src/components/Grid";
import ReleaseNote from "@site/src/components/ReleaseNote";

export default function BlogPostItems({
  items,
  component: BlogPostItemComponent = BlogPostItem,
}) {
  const isBlog = items.some(({ content }) =>
    content.metadata.permalink.startsWith("/blog/"),
  );
  return (
    <Grid>
      {isBlog
        ? BLOGS.map((blog) => <Blog key={blog.id} blog={blog} />)
        : RELEASES.map((release) => (
            <ReleaseNote key={release.id} release={release} />
          ))}
    </Grid>
  );
}
