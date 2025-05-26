import React from "react";
import { useLocation } from "@docusaurus/router";
import ThemeLayout from "@theme/Layout";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";

import { Layout } from "../../components";

export default function BlogLayout({ children, ...props }) {
  const location = useLocation();
  const { siteConfig } = useDocusaurusContext();

  const isHomeOrTagsPage =
    location.pathname === `${siteConfig.baseUrl}blog` ||
    location.pathname === `${siteConfig.baseUrl}blog/` ||
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
