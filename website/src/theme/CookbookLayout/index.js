import React from "react";
import ThemeLayout from "@theme/Layout";
import Link from "@docusaurus/Link";
import { useLocation } from "@docusaurus/router";

import { Layout } from "../../components";

function CookbookSidebar({ sidebar }) {
  const location = useLocation();

  if (!sidebar?.items?.length) {
    return null;
  }

  return (
    <aside className="cookbook-sidebar hidden md:block sticky top-20 self-start w-64 shrink-0 overflow-y-auto max-h-[calc(100vh-6rem)] hidden-scrollbar">
      <nav>
        <div className="text-sm font-semibold text-white/50 uppercase tracking-wider mb-4 px-3">
          {sidebar.title}
        </div>
        <ul className="flex flex-col gap-0.5">
          {sidebar.items.map((item) => {
            const isActive = location.pathname === item.permalink;
            return (
              <li key={item.permalink}>
                <Link
                  href={item.permalink}
                  className={`block px-3 py-2 rounded-lg text-sm transition-colors ${
                    isActive
                      ? "!text-white bg-white/10 font-medium"
                      : "!text-white/60 hover:!text-white hover:bg-white/5"
                  }`}
                >
                  {item.title}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
    </aside>
  );
}

export default function CookbookLayout(props) {
  const { sidebar, toc, children, ...layoutProps } = props;

  return (
    <ThemeLayout {...layoutProps}>
      <Layout>
        <div className="cookbook-page max-w-7xl mx-auto w-full">
          <div className="flex flex-col md:flex-row gap-6 md:gap-10">
            <CookbookSidebar sidebar={sidebar} />
            <main
              itemScope
              itemType="https://schema.org/Blog"
              className="hyphens-auto break-word overflow-x-hidden flex-1 min-w-0"
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
