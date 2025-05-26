import React from "react";
import { BlogPostProvider } from "@docusaurus/plugin-content-blog/client";
import BlogPostItem from "@theme/BlogPostItem";
import { SectionLabel, Grid, GridItem, Heading, Body } from "../../components";
import Link from "@docusaurus/Link";
import useBaseUrl from "@docusaurus/useBaseUrl";

export default function BlogPostItems({
  items,
  component: BlogPostItemComponent = BlogPostItem,
}) {
  const blogUrl = useBaseUrl("blog");
  const isBlog = items.some(({ content }) =>
    content.metadata.permalink.startsWith(blogUrl),
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
            <Heading level={2}>{firstBlogPost.content.metadata.title}</Heading>
            <Body size="l">{firstBlogPost.content.metadata.description}</Body>
          </Link>
          <Link href={firstBlogPost.content.metadata.permalink}>
            <div>
              <img
                src={useBaseUrl(firstBlogPost.content.frontMatter.thumbnail)}
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
                    src={useBaseUrl(blogPost.content.frontMatter.thumbnail)}
                    alt={blogPost.content.frontMatter.title}
                    className="object-contain rounded-md max-h-[210px] grow"
                  />
                  <span className="text-gray-600">
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
