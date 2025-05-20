import Link from "@docusaurus/Link";

import { BLOGS } from "../../posts";
import { Grid, GridItem } from "../Grid/Grid";
import { SectionLabel } from "../SectionLabel/SectionLabel";
import { Button } from "../Button/Button";

interface Props {
  variant: "red" | "green";
}

export const LatestNews = ({ variant }: Props) => {
  const posts = BLOGS.slice(0, 3);

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
          <SectionLabel color={variant} label="Blog" />
          <h1>Latest news</h1>
        </div>
        <div className="hidden md:block">{viewAllLinkNode}</div>
      </div>
      <Grid>
        {posts.map((post) => (
          <GridItem key={post.path}>
            <Link
              href={post.path}
              className="flex flex-col gap-6 h-full justify-between"
            >
              <div className="flex flex-col gap-6">
                <span className="text-white/50">
                  {new Date(post.date).toLocaleDateString("en-us", {
                    month: "short",
                    day: "numeric",
                    year: "numeric",
                  })}
                </span>
                <h3 className="overflow-hidden text-ellipsis text-pretty">
                  {post.title}
                </h3>
              </div>
              <img
                src={post.thumbnail}
                alt={post.title}
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
