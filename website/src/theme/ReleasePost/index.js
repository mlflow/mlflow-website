import React from "react";
import ThemeLayout from "@theme/Layout";

import { Layout } from "../../components";
import BlogSidebar from "@theme/BlogSidebar";

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
            <div className="flex flex-col">
              <h1 className="text-white">{BlogPostContent.metadata.title}</h1>
              <BlogPostContent />
            </div>
          </div>
        </div>
      </Layout>
    </ThemeLayout>
  );
}
