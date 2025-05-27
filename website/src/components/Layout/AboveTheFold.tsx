import { PropsWithChildren } from "react";
import { Heading, Body, SectionLabel } from "../index";

type Props = PropsWithChildren<{
  title: string;
  body: string | string[];
  sectionLabel?: string;
}>;

export function AboveTheFold({ children, title, body, sectionLabel }: Props) {
  const bodyParts = Array.isArray(body) ? body : [body];
  return (
    <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-container">
      <div className="flex flex-col justify-center items-center gap-6 w-full max-w-3xl mx-auto">
        {sectionLabel && <SectionLabel label={sectionLabel} />}
        <Heading level={1}>{title}</Heading>
        <div>
          {bodyParts.map((part, index) => (
            <Body key={index} size="l" align="center">
              {part}
            </Body>
          ))}
        </div>
      </div>
      {children}
    </div>
  );
}
