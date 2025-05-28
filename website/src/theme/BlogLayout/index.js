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
    location.pathname.includes(blogUrl + "/tags");

  return (
    <ThemeLayout {...props}>
      <Layout variant="colorful">
        <div
          className={
            isHomeOrTagsPage
              ? "flex flex-col md:px-20"
              : "flex flex-col max-w-7xl mx-auto w-full"
          }
        >
          {children}
        </div>
      </Layout>
    </ThemeLayout>
  );
}
