import Link from "@docusaurus/Link";
import { useLocation } from "@docusaurus/router";
import useBaseUrl from "@docusaurus/useBaseUrl";
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

type ThemeColor = {
  startColor: string;
  endColor: string;
} | null;

function getThemeColor(pathname: string): ThemeColor {
  if (pathname.startsWith(useBaseUrl("/genai"))) {
    return {
      startColor: "#EB1700",
      endColor: "#4A121A",
    };
  }
  if (pathname.startsWith(useBaseUrl("/classical-ml"))) {
    return {
      startColor: "#54c7ec",
      endColor: "#0A2342",
    };
  }
  return null;
}

export function Card({
  title,
  body,
  bodySize = "l",
  cta,
  image,
  padded = false,
}: Props) {
  const bodyParts = Array.isArray(body) ? body : [body];
  const location = useLocation();
  const colors = getThemeColor(location.pathname);

  return (
    <>
      <div className={contentWrapper({ padded })}>
        <Heading level={3}>{title}</Heading>
        <div>
          {bodyParts.map((part, index) => (
            <Body key={index} size={bodySize} margin="tight">
              {part}
            </Body>
          ))}
        </div>
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
        <div
          className="relative"
          style={
            colors
              ? {
                  paddingTop: "3%",
                  paddingLeft: "3%",
                  backgroundImage: `linear-gradient(to bottom, ${colors.startColor} 0%, ${colors.startColor} 50%, ${colors.endColor} 100%)`,
                }
              : undefined
          }
        >
          {(() => {
            const rounded = colors
              ? "rounded-tl-3xl rounded-tr-3xl rounded-bl-3xl"
              : "";
            return (
              <div className={`relative overflow-hidden ${rounded}`}>
                {image}
                <div
                  className={`absolute inset-0 bg-black/5 pointer-events-none ${rounded}`}
                />
              </div>
            );
          })()}
        </div>
      )}
    </>
  );
}
