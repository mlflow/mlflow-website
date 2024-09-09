import React from "react";
import { BlogPostProvider } from "@docusaurus/theme-common/internal";
import BlogPostItem from "@theme/BlogPostItem";
import { BLOGS } from "@site/src/posts";
import BlogItem from "@site/src/components/BlogItem";

export default function BlogPostItems({
  items,
  component: BlogPostItemComponent = BlogPostItem,
}) {
  const isBlog = items.some(({ content }) =>
    content.metadata.permalink.startsWith("/blog/"),
  );
  if (isBlog) {
    return (
      <>
        {items.map(({ content }) => (
          <BlogItem
            key={content.metadata.permalink}
            blog={content.metadata.frontMatter}
          />
        ))}
      </>
    );
  }

  return (
    <>
      {items.map(({ content: BlogPostContent }) => (
        <BlogPostProvider
          key={BlogPostContent.metadata.permalink}
          content={BlogPostContent}
        >
          <BlogPostItemComponent>
            <BlogPostContent />
          </BlogPostItemComponent>
        </BlogPostProvider>
      ))}
    </>
  );
}
