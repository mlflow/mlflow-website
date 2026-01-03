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
  ambient?: boolean;
}>;

const AmbientGlow = () => (
  <div className="absolute inset-0 pointer-events-none overflow-hidden">
    <div className="absolute top-1/4 left-1/4 w-80 h-80 rounded-full blur-3xl" style={{ background: 'rgba(120, 110, 220, 0.06)' }} />
    <div className="absolute bottom-1/4 right-1/4 w-80 h-80 rounded-full blur-3xl" style={{ background: 'rgba(130, 100, 230, 0.05)' }} />
    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 rounded-full blur-3xl" style={{ background: 'rgba(110, 120, 240, 0.04)' }} />
  </div>
);

export function Section({
  id,
  label,
  title,
  body,
  children,
  headingLevel = 1,
  align = "left",
  ambient = false,
}: Props) {
  useBrokenLinks().collectAnchor(id);
  return (
    <div
      id={id}
      className="relative flex flex-col w-full items-center justify-center gap-16"
    >
      {ambient && <AmbientGlow />}
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
