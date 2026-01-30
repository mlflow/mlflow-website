import React from "react";
import ThemeLayout from "@theme/Layout";

import { Layout } from "../../components";
import BlogSidebar from "@theme/BlogSidebar";

export default function ReleaseLayout(props) {
  const { sidebar, toc, children, ...layoutProps } = props;
  return (
    <ThemeLayout {...layoutProps}>
      <Layout>
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-row gap-12">
            <div className="contents text-white **:[nav_ul]:!mb-6">
              <BlogSidebar sidebar={sidebar} />
            </div>
            <main
              itemScope
              itemType="https://schema.org/Blog"
              className="hyphens-auto break-word"
            >
              {children}
            </main>
            {toc && <div className="col col--2">{toc}</div>}
          </div>
        </div>
      </Layout>
    </ThemeLayout>
  );
}
