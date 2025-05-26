import React from "react";
import { useLocation } from "@docusaurus/router";
import ThemeLayout from "@theme/Layout";
import useBaseUrl from "@docusaurus/useBaseUrl";

import { Layout } from "../../components";

export default function BlogLayout({ children, ...props }) {
  const location = useLocation();
  const blogUrl = useBaseUrl("blog");

  const isHomeOrTagsPage =
    location.pathname === blogUrl ||
    location.pathname === blogUrl + `/` ||
    location.pathname.includes("/blog/tags");

  return (
    <ThemeLayout {...props}>
      <Layout>
        <div
          className={
            isHomeOrTagsPage
              ? "flex flex-col px-6 md:px-20 pt-32 "
              : "flex flex-col mt-40 max-w-7xl mx-auto w-full px-6"
          }
        >
          {children}
        </div>
      </Layout>
    </ThemeLayout>
  );
}
