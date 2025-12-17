import { PropsWithChildren, ReactNode } from "react";
import { Body, Heading, SectionLabel } from "..";
import useBrokenLinks from "@docusaurus/useBrokenLinks";

type Props = PropsWithChildren<{
  label?: string;
  title: ReactNode;
  body?: ReactNode;
  id?: string;
  headingLevel?: 1 | 2 | 3 | 4 | 5;
  align?: "left" | "center";
}>;

export function Section({
  id,
  label,
  title,
  body,
  children,
  headingLevel = 1,
  align = "left",
}: Props) {
  useBrokenLinks().collectAnchor(id);
  return (
    <div
      id={id}
      className="flex flex-col w-full items-center justify-center gap-16"
    >
      <div className="flex flex-col w-full max-w-5xl items-center justify-center gap-12">
        {label && <SectionLabel label={label} />}
        <Heading level={headingLevel} aria-level={2}>
          {title}
        </Heading>
        {body && (
          <Body size="l" align={align}>
            {body}
          </Body>
        )}
      </div>
      {children}
    </div>
  );
}
