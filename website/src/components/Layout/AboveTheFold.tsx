import { PropsWithChildren, ReactNode } from "react";
import { Heading, Body, SectionLabel, GetStartedButton } from "../index";
import { cva, VariantProps } from "class-variance-authority";

type Props = VariantProps<typeof innerWrapper> &
  PropsWithChildren<{
    title: string;
    body: ReactNode | ReactNode[];
    sectionLabel?: string;
    hasGetStartedButton?: true | string;
    actions?: ReactNode;
    bodyColor?: "default" | "white";
    bodySize?: "s" | "m" | "l" | "xl";
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
  actions,
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
        {actions && (
          <div className="flex flex-wrap justify-center items-center gap-4">
            {actions}
          </div>
        )}
      </div>
      {children}
    </div>
  );
}
