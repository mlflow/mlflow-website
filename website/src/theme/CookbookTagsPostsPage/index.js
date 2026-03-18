import React from "react";
import clsx from "clsx";
import {
  HtmlClassNameProvider,
  ThemeClassNames,
} from "@docusaurus/theme-common";
import { useBlogTagsPostsPageTitle } from "@docusaurus/theme-common/internal";
import CookbookLayout from "@site/src/theme/CookbookLayout";
import CookbookCard from "@site/src/theme/CookbookCard";
import { Heading } from "../../components";

function CookbookTagsPostsPageContent({ tag, items, sidebar, listMetadata }) {
  const title = useBlogTagsPostsPageTitle(tag);
  return (
    <CookbookLayout sidebar={sidebar}>
      <div className="flex flex-col gap-8">
        <Heading level={1}>{title}</Heading>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {items.map(({ content }) => (
            <CookbookCard
              key={content.metadata.permalink}
              metadata={content.metadata}
              frontMatter={content.frontMatter}
            />
          ))}
        </div>
      </div>
    </CookbookLayout>
  );
}

export default function CookbookTagsPostsPage(props) {
  return (
    <HtmlClassNameProvider
      className={clsx(
        ThemeClassNames.wrapper.blogPages,
        ThemeClassNames.page.blogTagPostListPage,
      )}
    >
      <CookbookTagsPostsPageContent {...props} />
    </HtmlClassNameProvider>
  );
}
