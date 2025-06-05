import Link from "@docusaurus/Link";
import { Body, Button, Heading } from "..";
import { ComponentProps, ReactNode } from "react";
import { cva, VariantProps } from "class-variance-authority";

type Props = {
  title: string;
  body: string | string[];
  bodySize?: ComponentProps<typeof Body>["size"];
  cta?: {
    href: string;
    text: string;
    prominent?: boolean;
  };
  image?: ReactNode;
} & VariantProps<typeof contentWrapper> &
  VariantProps<typeof imageWrapper>;

const contentWrapper = cva("flex flex-col gap-4", {
  variants: {
    padded: {
      true: "p-6",
      false: "",
    },
  },
});

const imageWrapper = cva("w-full relative", {
  variants: {
    rounded: {
      true: "rounded-lg overflow-hidden",
      false: "",
    },
  },
});

export function Card({
  title,
  body,
  bodySize = "l",
  cta,
  image,
  padded = false,
  rounded = true,
}: Props) {
  const bodyParts = Array.isArray(body) ? body : [body];
  return (
    <>
      <div className={contentWrapper({ padded })}>
        <Heading level={5} aria-level={3}>
          {title}
        </Heading>
        {bodyParts.map((part, index) => (
          <Body key={index} size={bodySize}>
            {part}
          </Body>
        ))}
        {cta && (
          <Link href={cta.href}>
            <Button
              variant={cta.prominent ? "primary" : "outline"}
              size={cta.prominent ? "medium" : "small"}
            >
              {cta.text}
            </Button>
          </Link>
        )}
      </div>
      {image && (
        <div className={imageWrapper({ rounded })}>
          {image}
          <div className="absolute inset-0 bg-black/5 pointer-events-none" />
        </div>
      )}
    </>
  );
}
