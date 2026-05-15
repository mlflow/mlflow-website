import React from "react";
import Link from "@docusaurus/Link";
import ThemeLayout from "@theme/Layout";
import BlogListPaginator from "@theme/BlogListPaginator";

import { Layout } from "../../components";
import { Heading } from "../../components";

export default function ArticleItemsList({ items, metadata }) {
  return (
    <ThemeLayout>
      <Layout>
        <div className="flex flex-col max-w-7xl mx-auto w-full gap-10 mb-16">
          <Heading level={1}>Articles</Heading>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {items.map(({ content }) => (
              <Link
                key={content.metadata.permalink}
                href={content.metadata.permalink}
                className="group flex flex-col rounded-2xl overflow-hidden bg-white/5 hover:bg-white/10 transition-colors"
              >
                <div className="w-full aspect-video overflow-hidden">
                  <img
                    src={content.frontMatter.image}
                    alt={content.metadata.title}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    loading="lazy"
                  />
                </div>
                <div className="p-5">
                  <span className="line-clamp-2 text-lg font-semibold text-white">
                    {content.metadata.title}
                  </span>
                </div>
              </Link>
            ))}
          </div>
          <BlogListPaginator metadata={metadata} />
        </div>
      </Layout>
    </ThemeLayout>
  );
}
