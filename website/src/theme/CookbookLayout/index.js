import React from "react";
import ThemeLayout from "@theme/Layout";
import Link from "@docusaurus/Link";
import { useLocation } from "@docusaurus/router";

import { Layout } from "../../components";

// Map parent permalink to ordered child permalinks for nested sidebar groups.
const COOKBOOK_GROUPS = {
  "/cookbook/databricks-genie": [
    "/cookbook/genie-tracing-pipeline",
    "/cookbook/genie-evaluation-judges",
    "/cookbook/genie-space-analyzer",
  ],
};

// Build a set of all child permalinks for quick lookup.
const CHILD_PERMALINKS = new Set(
  Object.values(COOKBOOK_GROUPS).flat(),
);

function SidebarLink({ item, isActive, indent }) {
  return (
    <li key={item.permalink}>
      <Link
        href={item.permalink}
        className={`block py-2 rounded-lg text-sm transition-colors ${
          indent ? "pl-6 pr-3" : "px-3"
        } ${
          isActive
            ? "!text-white bg-white/10 font-medium"
            : "!text-white/60 hover:!text-white hover:bg-white/5"
        }`}
      >
        {item.title}
      </Link>
    </li>
  );
}

function CookbookSidebar({ sidebar }) {
  const location = useLocation();

  if (!sidebar?.items?.length) {
    return null;
  }

  // Index items by permalink for child lookup.
  const itemsByPermalink = {};
  for (const item of sidebar.items) {
    itemsByPermalink[item.permalink] = item;
  }

  // Separate ungrouped items from group parents, keeping original order
  // within each bucket, then render ungrouped first and groups last.
  const topLevel = sidebar.items.filter(
    (item) => !CHILD_PERMALINKS.has(item.permalink),
  );
  const ungrouped = topLevel.filter(
    (item) => !COOKBOOK_GROUPS[item.permalink],
  );
  const grouped = topLevel.filter(
    (item) => !!COOKBOOK_GROUPS[item.permalink],
  );
  const orderedItems = [...ungrouped, ...grouped];

  return (
    <aside className="cookbook-sidebar hidden md:block sticky top-20 self-start w-72 shrink-0 overflow-y-auto max-h-[calc(100vh-6rem)] hidden-scrollbar">
      <nav>
        <Link
          href="/cookbook"
          className="block text-sm font-semibold !text-white/50 uppercase tracking-wider mb-4 px-3 hover:!text-white transition-colors"
        >
          {sidebar.title}
        </Link>
        <ul className="flex flex-col gap-0.5">
          {orderedItems.map((item) => {
              const isActive = location.pathname === item.permalink;
              const children = COOKBOOK_GROUPS[item.permalink];
              return (
                <React.Fragment key={item.permalink}>
                  <SidebarLink
                    item={item}
                    isActive={isActive}
                    indent={false}
                  />
                  {children &&
                    children.map((childPermalink) => {
                      const child = itemsByPermalink[childPermalink];
                      if (!child) return null;
                      const childActive =
                        location.pathname === child.permalink;
                      return (
                        <SidebarLink
                          key={child.permalink}
                          item={child}
                          isActive={childActive}
                          indent={true}
                        />
                      );
                    })}
                </React.Fragment>
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
        <div className="cookbook-page max-w-[90rem] mx-auto w-full">
          <div className="flex flex-col md:flex-row gap-6 md:gap-10">
            <CookbookSidebar sidebar={sidebar} />
            <main
              itemScope
              itemType="https://schema.org/Blog"
              className="hyphens-none break-words overflow-x-hidden flex-1 min-w-0"
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
