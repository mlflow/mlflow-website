import React from "react";
import ThemeLayout from "@theme/Layout";
import { BlogPostProvider } from "@docusaurus/theme-common/internal";

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
        <div className="pt-24 max-w-7xl mx-auto">
          <div className="flex flex-row">
            <BlogSidebar className="text-white" sidebar={sidebar} />
            <BlogPostProvider content={BlogPostContent}>
              <BlogPostItem>
                <BlogPostContent />
              </BlogPostItem>
            </BlogPostProvider>
          </div>
        </div>
      </Layout>
    </ThemeLayout>
  );
}
