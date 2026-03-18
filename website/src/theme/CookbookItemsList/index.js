import React from "react";
import CookbookLayout from "@site/src/theme/CookbookLayout";
import CookbookCard from "@site/src/theme/CookbookCard";
import { Heading, Body } from "../../components";

export default function CookbookItemsList({ items, sidebar, ...layoutProps }) {
  return (
    <CookbookLayout {...layoutProps}>
      <div className="flex flex-col gap-8">
        <div className="flex flex-col gap-2">
          <Heading level={1}>Cookbook</Heading>
          <Body size="l" className="!text-white/60">
            Hands-on guides for building, evaluating, and observing GenAI
            applications with MLflow — from tracing and prompt management to
            LLM-as-a-judge evaluation and production deployment.
          </Body>
        </div>

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
