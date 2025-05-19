import React from "react";
import { BlogPostProvider } from "@docusaurus/theme-common/internal";
import BlogPostItem from "@theme/BlogPostItem";
import { SectionLabel, Grid, GridRow, GridItem } from "../../components";

export default function BlogPostItems({
  items,
  component: BlogPostItemComponent = BlogPostItem,
}) {
  const isBlog = items.some(({ content }) =>
    content.metadata.permalink.startsWith("/blog/"),
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
          <a
            href={firstBlogPost.content.metadata.permalink}
            className="flex flex-col justify-center items-start gap-4"
          >
            <SectionLabel color="green" label="FEATURED" />
            <h1>{firstBlogPost.content.metadata.title}</h1>
            <p className="text-white/60">
              {firstBlogPost.content.metadata.description}
            </p>
          </a>
          <a href={firstBlogPost.content.metadata.permalink}>
            <div>
              <img
                src={firstBlogPost.content.frontMatter.thumbnail}
                alt={firstBlogPost.content.frontMatter.title}
                className="w-full h-full object-cover rounded-md"
              />
            </div>
          </a>
        </div>

        <Grid>
          {blogPostsGrid.map((blogPostsRow, index) => (
            <GridRow key={index}>
              {blogPostsRow.map((blogPost) => (
                <GridItem
                  key={blogPost.content.metadata.permalink}
                  className="py-10 pl-0 pr-0 md:pl-10 md:pr-10 first:pl-0 last:pr-0 gap-6"
                >
                  <a
                    href={blogPost.content.metadata.permalink}
                    className="flex flex-col w-full"
                  >
                    <img
                      src={blogPost.content.frontMatter.thumbnail}
                      alt={blogPost.content.frontMatter.title}
                      className="object-cover rounded-md max-h-[210px]"
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
                    <h3>{blogPost.content.metadata.title}</h3>
                  </a>
                </GridItem>
              ))}
            </GridRow>
          ))}
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
