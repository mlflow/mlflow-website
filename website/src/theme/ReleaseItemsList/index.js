import React from "react";
import { BlogPostProvider } from "@docusaurus/plugin-content-blog/client";
import BlogPostItem from "@theme/BlogPostItem";
import ReleaseLayout from "@site/src/theme/ReleaseLayout";

export default function ReleaseItemsList({ items, ...layoutProps }) {
  return (
    <ReleaseLayout {...layoutProps}>
      {items.map(({ content: BlogPostContent }) => (
        <BlogPostProvider
          key={BlogPostContent.metadata.permalink}
          content={BlogPostContent}
        >
          <BlogPostItem className="max-w-4xl">
            <BlogPostContent />
          </BlogPostItem>
        </BlogPostProvider>
      ))}
    </ReleaseLayout>
  );
}
