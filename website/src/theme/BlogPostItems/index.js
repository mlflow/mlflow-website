import React from "react";
import { BlogPostProvider } from "@docusaurus/theme-common/internal";
import BlogPostItem from "@theme/BlogPostItem";
import { SectionLabel, Grid, GridItem } from "../../components";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";

export default function BlogPostItems({
  items,
  component: BlogPostItemComponent = BlogPostItem,
}) {
  const { siteConfig } = useDocusaurusContext();

  const isBlog = items.some(({ content }) =>
    content.metadata.permalink.startsWith(`${siteConfig.baseUrl}blog/`),
  );

  if (isBlog) {
    const [firstBlogPost, ...restBlogPosts] = items;

    const blogPostsGrid = Array.from(
      { length: Math.ceil(restBlogPosts.length / 3) },
      (v, i) => restBlogPosts.slice(i * 3, i * 3 + 3),
    );

    return (
      <div className="flex flex-col gap-10 px-6 md:px-20 mb-10 max-w-7xl mx-auto">
        <div className="flex flex-col-reverse md:flex-row gap-10">
          <Link
            href={firstBlogPost.content.metadata.permalink}
            className="flex flex-col justify-center items-start gap-4"
          >
            <SectionLabel color="green" label="FEATURED" />
            <h1>{firstBlogPost.content.metadata.title}</h1>
            <p className="text-white/60">
              {firstBlogPost.content.metadata.description}
            </p>
          </Link>
          <Link href={firstBlogPost.content.metadata.permalink}>
            <div>
              <img
                src={firstBlogPost.content.frontMatter.thumbnail}
                alt={firstBlogPost.content.frontMatter.title}
                className="w-full h-full object-cover rounded-md"
              />
            </div>
          </Link>
        </div>

        <Grid columns={3}>
          {blogPostsGrid.flatMap((blogPostsRow) =>
            blogPostsRow.map((blogPost) => (
              <GridItem key={blogPost.content.metadata.permalink}>
                <Link
                  href={blogPost.content.metadata.permalink}
                  className="flex flex-col w-full h-full gap-4"
                >
                  <img
                    src={blogPost.content.frontMatter.thumbnail}
                    alt={blogPost.content.frontMatter.title}
                    className="object-contain rounded-md max-h-[210px] grow"
                  />
                  <span className="text-white/60">
                    {new Date(
                      blogPost.content.metadata.date,
                    ).toLocaleDateString("en-US", {
                      year: "numeric",
                      month: "short",
                      day: "numeric",
                    })}
                  </span>
                  <div
                    role="heading"
                    aria-level={3}
                    className="text-xl h-21 line-clamp-3"
                  >
                    {blogPost.content.metadata.title}
                  </div>
                </Link>
              </GridItem>
            )),
          )}
        </Grid>
      </div>
    );
  }

  return (
    <>
      {items.map(({ content: BlogPostContent }) => (
        <BlogPostProvider
          key={BlogPostContent.metadata.permalink}
          content={BlogPostContent}
        >
          <BlogPostItemComponent>
            <BlogPostContent />
          </BlogPostItemComponent>
        </BlogPostProvider>
      ))}
    </>
  );
}
