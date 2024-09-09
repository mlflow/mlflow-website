import React from "react";
import clsx from "clsx";
import Layout from "@theme/Layout";
import BlogSidebar from "@theme/BlogSidebar";
export default function BlogLayout(props) {
  const { sidebar, toc, children, ...layoutProps } = props;
  const isBlog = sidebar.items.some(({ permalink }) =>
    permalink.startsWith("/blog/"),
  );
  const hasSidebar = !isBlog;
  return (
    <Layout {...layoutProps}>
      <div className="container margin-vert--lg">
        <div className="row">
          {hasSidebar && <BlogSidebar sidebar={sidebar} />}
          <main
            className={clsx("col", {
              "col--7": hasSidebar,
              "col--9 col--offset-1": !hasSidebar,
            })}
            itemScope
            itemType="https://schema.org/Blog"
          >
            {children}
          </main>
          {toc && <div className="col col--2">{toc}</div>}
        </div>
      </div>
    </Layout>
  );
}
