import React from "react";
import ThemeLayout from "@theme/Layout";
import { useWindowSize } from "@docusaurus/theme-common";

import { Layout } from "../../components";
import BlogSidebar from "@theme/BlogSidebar";

export default function ReleaseLayout(props) {
  const { sidebar, toc, children, ...layoutProps } = props;
  const windowSize = useWindowSize();
  const isDesktop = windowSize === "desktop";

  return (
    <ThemeLayout {...layoutProps}>
      <Layout>
        <div className="release-page max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row gap-6 md:gap-12">
            {isDesktop && (
              <div className="contents text-white **:[nav_ul]:!mb-6">
                <BlogSidebar sidebar={sidebar} />
              </div>
            )}
            <main
              itemScope
              itemType="https://schema.org/Blog"
              className="hyphens-auto break-word overflow-x-hidden w-full"
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
