import { PropsWithChildren } from "react";
import { Body, Heading, SectionLabel } from "..";

type Props = PropsWithChildren<{
  label?: string;
  title: string;
  body?: string;
}>;

export function Section({ label, title, body, children }: Props) {
  return (
    <div className="flex flex-col w-full items-center justify-center gap-16">
      <div className="flex flex-col w-full items-center justify-center gap-6">
        {label && <SectionLabel label={label} />}
        <Heading level={2}>{title}</Heading>
        {body && <Body size="l">{body}</Body>}
      </div>
      {children}
    </div>
  );
}
