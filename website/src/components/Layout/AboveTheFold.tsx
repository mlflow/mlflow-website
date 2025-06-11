import { PropsWithChildren, ReactNode } from "react";
import { Heading, Body, SectionLabel, GetStartedButton } from "../index";
import { cva, VariantProps } from "class-variance-authority";

type Props = VariantProps<typeof innerWrapper> &
  PropsWithChildren<{
    title: string | ReactNode;
    body: string | string[];
    sectionLabel?: string;
    hasGetStartedButton?: true | string;
    bodyColor?: "default" | "white";
  }>;

const innerWrapper = cva(
  "flex flex-col justify-center items-center flex-1 gap-6 w-full max-w-5xl mx-auto md:px-4",
  {
    variants: {
      minHeight: {
        small: "min-h-[350px]",
        default: "min-h-[550px]",
      },
    },
  },
);

export function AboveTheFold({
  children,
  title,
  body,
  sectionLabel,
  hasGetStartedButton,
  bodyColor,
  minHeight = "default",
}: Props) {
  const bodyParts = Array.isArray(body) ? body : [body];
  return (
    <div className="flex flex-col gap-6">
      <div className={innerWrapper({ minHeight })}>
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
