import React from "react";
import Link from "@docusaurus/Link";

export default function CookbookCard({ metadata, frontMatter }) {
  return (
    <Link
      key={metadata.permalink}
      href={metadata.permalink}
      className="group flex flex-col gap-1.5 rounded-xl border border-white/8 bg-white/3 px-6 py-5 transition-colors hover:border-white/16 hover:bg-white/6"
    >
      <div className="flex items-center gap-3">
        <span className="text-base font-medium !text-white group-hover:!text-white">
          {metadata.title}
        </span>
      </div>
      {metadata.description && (
        <span className="text-sm !text-white/50 line-clamp-2">
          {metadata.description}
        </span>
      )}
      {frontMatter.tags?.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mt-1">
          {metadata.tags.map((tag) => (
            <span
              key={tag.label}
              className="rounded-md bg-white/8 px-2 py-0.5 text-xs !text-white/50"
            >
              {tag.label}
            </span>
          ))}
        </div>
      )}
    </Link>
  );
}
