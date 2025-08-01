import React from "react";
import ThemeLayout from "@theme/Layout";
import { BlogPostProvider } from "@docusaurus/plugin-content-blog/client";

import { Layout } from "../../components";
import BlogSidebar from "@theme/BlogSidebar";
import BlogPostItem from "@theme/BlogPostItem";

export default function ReleasePost({
  content: BlogPostContent,
  sidebar,
  ...layoutProps
}) {
  return (
    <ThemeLayout {...layoutProps}>
      <Layout>
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-row">
            <div className="contents text-white **:[nav_ul]:!mb-6">
              <BlogSidebar sidebar={sidebar} />
            </div>
            <BlogPostProvider content={BlogPostContent}>
              <BlogPostItem className="max-w-prose **:[h2>a]:!text-2xl">
                <BlogPostContent />
              </BlogPostItem>
            </BlogPostProvider>
          </div>
        </div>
      </Layout>
    </ThemeLayout>
  );
}
