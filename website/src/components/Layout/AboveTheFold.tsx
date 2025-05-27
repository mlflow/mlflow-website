import { PropsWithChildren } from "react";
import { Heading, Body, SectionLabel, GetStartedButton } from "../index";

type Props = PropsWithChildren<{
  title: string;
  body: string | string[];
  sectionLabel?: string;
  hasGetStartedButton?: boolean;
}>;

export function AboveTheFold({
  children,
  title,
  body,
  sectionLabel,
  hasGetStartedButton,
}: Props) {
  const bodyParts = Array.isArray(body) ? body : [body];
  return (
    <div className="flex flex-col gap-16">
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
        {hasGetStartedButton && <GetStartedButton />}
      </div>
      {children}
    </div>
  );
}
