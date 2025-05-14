import React from "react";
import { BlogPostProvider } from "@docusaurus/theme-common/internal";
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
          <BlogPostItem>
            <BlogPostContent />
          </BlogPostItem>
        </BlogPostProvider>
      ))}
    </ReleaseLayout>
  );
}
