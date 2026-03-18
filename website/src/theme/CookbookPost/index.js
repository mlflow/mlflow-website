import React from "react";
import { BlogPostProvider } from "@docusaurus/plugin-content-blog/client";
import BlogPostItem from "@theme/BlogPostItem";
import CookbookLayout from "@site/src/theme/CookbookLayout";
import Link from "@docusaurus/Link";

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

export default function CookbookPost({
  content: BlogPostContent,
  sidebar,
  ...layoutProps
}) {
  const { metadata } = BlogPostContent;
  return (
    <CookbookLayout sidebar={sidebar} {...layoutProps}>
      <BlogPostProvider content={BlogPostContent}>
        <BlogPostItem className="max-w-none">
          <CookbookTags tags={metadata.tags} />
          <BlogPostContent />
        </BlogPostItem>
      </BlogPostProvider>
    </CookbookLayout>
  );
}
