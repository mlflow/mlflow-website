import { PropsWithChildren } from "react";
import { Heading, Body, SectionLabel, GetStartedButton } from "../index";

type Props = PropsWithChildren<{
  title: string;
  body: string | string[];
  sectionLabel?: string;
  hasGetStartedButton?: true | string;
  bodyColor?: "default" | "white";
}>;

export function AboveTheFold({
  children,
  title,
  body,
  sectionLabel,
  hasGetStartedButton,
  bodyColor,
}: Props) {
  const bodyParts = Array.isArray(body) ? body : [body];
  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex flex-col justify-center items-center flex-1 gap-6 w-full max-w-5xl mx-auto px-4">
        {sectionLabel && <SectionLabel label={sectionLabel} />}
        <div className="max-w-4xl mx-auto">
          <Heading level={1}>{title}</Heading>
        </div>
        <div className="mx-auto text-pretty">
          {bodyParts.map((part, index) => (
            <Body key={index} size="l" align="center" color={bodyColor}>
              {part}
            </Body>
          ))}
        </div>
        {hasGetStartedButton && (
          <GetStartedButton
            link={
              hasGetStartedButton === true ? undefined : hasGetStartedButton
            }
          />
        )}
      </div>
      {children}
    </div>
  );
}
