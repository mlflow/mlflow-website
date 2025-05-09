import React from "react";
import { useLocation } from "@docusaurus/router";

import { Layout } from "../../components";

export default function BlogLayout({ children, ...props }) {
  const location = useLocation();

  const isBlogHomePage = location.pathname === "/blog";

  return (
    <Layout>
      <div
        className={
          isBlogHomePage
            ? "flex flex-col px-6 md:px-20"
            : "flex flex-col mt-40 max-w-4xl mx-auto px-6"
        }
      >
        {children}
      </div>
    </Layout>
  );
}
