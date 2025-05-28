import Link from "@docusaurus/Link";

import { Grid, GridItem } from "../Grid/Grid";
import { SectionLabel } from "../Section/SectionLabel";
import { Button } from "../Button/Button";
import { Heading } from "../Typography/Heading";
import blogPosts from "@site/.docusaurus/blog-posts.json";
import type { BlogContent } from "@docusaurus/plugin-content-blog";
import useBaseUrl from "@docusaurus/useBaseUrl";

export const LatestNews = () => {
  const posts = blogPosts.slice(0, 3) as unknown as BlogContent["blogPosts"];

  const viewAllLinkNode = (
    <Link href="/blog" className="">
      <Button variant="outline" size="small">
        View all
      </Button>
    </Link>
  );

  return (
    <div className="flex flex-col gap-4 w-full items-center md:item-start">
      <div className="flex flex-row justify-between items-center gap-4 flex-wrap w-full">
        <div className="flex flex-col gap-6 items-start">
          <SectionLabel label="Blog" />
          <Heading level={2}>Latest news</Heading>
        </div>
        <div className="hidden md:block">{viewAllLinkNode}</div>
      </div>
      <Grid>
        {posts.map((post) => (
          <GridItem key={post.metadata.permalink}>
            <Link
              href={post.metadata.permalink}
              className="flex flex-col gap-6 h-full justify-between"
            >
              <div className="flex flex-col gap-6">
                <span className="text-gray-500">
                  {new Date(post.metadata.date).toLocaleDateString("en-us", {
                    month: "short",
                    day: "numeric",
                    year: "numeric",
                  })}
                </span>
                <h3 className="overflow-hidden text-ellipsis text-pretty">
                  {post.metadata.title}
                </h3>
              </div>
              <img
                src={useBaseUrl(post.metadata.frontMatter.thumbnail as string)}
                alt={post.metadata.title}
                className="rounded-2xl md:max-h-[210px] object-cover max-w-full"
              />
            </Link>
          </GridItem>
        ))}
      </Grid>
      <div className="block md:hidden">{viewAllLinkNode}</div>
    </div>
  );
};
