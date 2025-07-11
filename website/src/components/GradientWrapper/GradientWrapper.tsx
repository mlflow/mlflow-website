import { cx } from "class-variance-authority";
import type { CSSProperties, PropsWithChildren } from "react";

export type Variant = "blue" | "red" | "colorful";
type Direction = "up" | "down";

function getColors(variant: Variant) {
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

export function getGradientStyles(
  variant: Variant,
  direction: Direction = "up",
  radial = true,
  height?: number,
): CSSProperties {
  const colors = getColors(variant);
  return {
    position: "absolute",
    inset: 0,
    pointerEvents: "none",
    maskComposite: "intersect",
    height,
    backgroundImage: `
      repeating-linear-gradient(
        to right,
        rgba(0, 0, 0, 0.05),
        rgba(0, 0, 0, 0.25) ${direction === "down" ? "24px" : "18px"},
        transparent 2px,
        transparent 10px
      ),
      radial-gradient(
        circle at ${direction === "down" ? "top" : "bottom"} center,
        ${colors.center} 0%,
        transparent 60%
      ),
      linear-gradient(to right, color-mix(in srgb, ${colors.center}, ${colors.left}), color-mix(in srgb, ${colors.center}, ${colors.right}))
    `,
    maskImage: `
      ${radial ? "radial-gradient(ellipse at center bottom, black 60%, transparent 80%)," : ""}
      linear-gradient(to ${direction === "down" ? "bottom" : "top"}, black ${direction === "down" ? "40%" : "10%"}, transparent ${direction === "down" ? "90%" : "40%"})
    `,
  };
}

type Props = PropsWithChildren<{
  element?: keyof HTMLElementTagNameMap;
  variant: Variant;
  direction?: Direction;
  radial: boolean;
  height?: number;
  className?: string;
}>;

export function GradientWrapper({
  element: Element = "div",
  variant,
  direction = "up",
  radial,
  children,
  height,
  className,
}: Props) {
  return (
    <Element className={cx("relative", className)}>
      <div style={getGradientStyles(variant, direction, radial, height)} />
      <div className="z-1">{children}</div>
    </Element>
  );
}
