import React from "react";
import clsx from "clsx";
import ThemeLayout from "@theme/Layout";

import { Layout } from "../../components";
import BlogSidebar from "@theme/BlogSidebar";

export default function ReleaseLayout(props) {
  const { sidebar, toc, children, ...layoutProps } = props;
  console.log("@@@ sidebar", props);
  return (
    <ThemeLayout {...layoutProps}>
      <Layout>
        <div className="pt-24 max-w-7xl mx-auto">
          <div className="flex flex-row">
            <BlogSidebar className="text-white" sidebar={sidebar} />
            <main itemScope itemType="https://schema.org/Blog">
              {children}
            </main>
            {toc && <div className="col col--2">{toc}</div>}
          </div>
        </div>
      </Layout>
    </ThemeLayout>
  );
}
