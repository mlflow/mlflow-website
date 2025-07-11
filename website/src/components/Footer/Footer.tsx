import { cva } from "class-variance-authority";

import Logo from "@site/static/img/mlflow-logo-white.svg";

import { FooterMenuItem } from "../FooterMenuItem/FooterMenuItem";
import { MLFLOW_DOCS_URL } from "@site/src/constants";

const footerVariants = cva("relative pb-30 flex flex-col pt-30");

type Variant = "blue" | "red" | "colorful";

export function getColors(variant: Variant) {
  switch (variant) {
    case "blue":
      return {
        center: "oklch(0.66 0.11 221.53)",
        left: "navy 40%",
        right: "teal 40%",
      };
    case "red":
      return {
        center: "var(--color-brand-red)",
        left: "black 10%",
        right: "oklch(0.91 0.09 326.28) 40%",
      };
    case "colorful":
      return {
        center: "var(--color-brand-red)",
        left: "oklch(0.33 0.15 328.37) 80%",
        right: "oklch(0.66 0.17 248.82) 100%",
      };
  }
}

export const Footer = ({ variant }: { variant: Variant }) => {
  const colors = getColors(variant);
  return (
    <footer className={footerVariants()}>
      <div
        className="absolute inset-0 pointer-events-none mask-intersect"
        style={{
          backgroundImage: `
            repeating-linear-gradient(
              to right,
              rgba(0, 0, 0, 0.05),
              rgba(0, 0, 0, 0.25) 18px,
              transparent 2px,
              transparent 10px
            ),
            radial-gradient(
              circle at center,
              ${colors.center} 0%,
              transparent 60%
            ),
            linear-gradient(to right, color-mix(in srgb, ${colors.center}, ${colors.left}), color-mix(in srgb, ${colors.center}, ${colors.right}))
          `,
          maskImage: `
            radial-gradient(ellipse at center bottom, black 60%, transparent 80%),
            linear-gradient(to top, black 10%, transparent 40%)
          `,
        }}
      />
      <div className="flex flex-row justify-between items-start px-6 lg:px-20 gap-10 xs:gap-0 max-w-container">
        <div className="flex flex-col gap-8">
          <Logo className="h-[36px] shrink-0" />
          <div className="text-xs text-gray-800 text-left md:text-nowrap md:w-0">
            © 2025 MLflow Project, a Series of LF Projects, LLC.
          </div>
        </div>

        <div className="flex flex-col flex-wrap justify-end md:text-right md:flex-row gap-x-10 lg:gap-x-20 gap-y-5 w-2/5 md:w-auto md:pt-2 max-w-fit">
          <FooterMenuItem href="/">Components</FooterMenuItem>
          <FooterMenuItem href="/releases">Releases</FooterMenuItem>
          <FooterMenuItem href="/blog">Blog</FooterMenuItem>
          <FooterMenuItem href={MLFLOW_DOCS_URL}>Docs</FooterMenuItem>
          <FooterMenuItem href="/ambassadors">
            Ambassador Program
          </FooterMenuItem>
        </div>
      </div>
    </footer>
  );
};
