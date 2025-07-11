import { Header } from "../Header/Header";
import { Footer, getColors } from "../Footer/Footer";
import { cva } from "class-variance-authority";
import { createContext, PropsWithChildren, useContext } from "react";
import { useLocation } from "@docusaurus/router";
import useBaseUrl from "@docusaurus/useBaseUrl";

type Props = PropsWithChildren;

const wrapper = cva("flex flex-col gap-20 bg-no-repeat w-full py-32 relative");

export const LayoutContext = createContext<"red" | "blue" | "colorful">(
  "colorful",
);

export function useLayoutVariant() {
  const variant = useContext(LayoutContext);
  if (variant === undefined) {
    throw new Error("useLayoutVariant must be used within a Layout");
  }
  return variant;
}

function getLayoutType(pathname: string) {
  const genAI = useBaseUrl("/genai");
  const classicalML = useBaseUrl("/classical-ml");

  if (pathname.startsWith(genAI)) {
    if (pathname === genAI || pathname === `${genAI}/`) {
      return "genai";
    } else {
      return "genai-subpage";
    }
  } else if (pathname.startsWith(classicalML)) {
    if (pathname === classicalML || pathname === `${classicalML}/`) {
      return "classical-ml";
    } else {
      return "classical-ml-subpage";
    }
  } else if (pathname === useBaseUrl("/")) {
    return "home";
  } else {
    return "default";
  }
}

export const Layout = ({ children }: Props) => {
  const location = useLocation();

  const layoutType = getLayoutType(location.pathname);
  const variant = layoutType.startsWith("genai")
    ? "red"
    : layoutType.startsWith("classical-ml")
      ? "blue"
      : layoutType === "home"
        ? "colorful"
        : null;
  const direction = layoutType.endsWith("subpage") ? "up" : "down";
  const colors = getColors(variant);

  return (
    <LayoutContext.Provider value={variant}>
      <div className="flex flex-col min-h-screen w-full bg-[#0E1416]">
        <Header />
        <main className="flex flex-col">
          <div className={wrapper()}>
            <div
              className="absolute inset-0 pointer-events-none mask-intersect h-[820px]"
              style={{
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
                  linear-gradient(to ${direction === "down" ? "bottom" : "top"}, black ${direction === "down" ? "40%" : "10%"}, transparent ${direction === "down" ? "90%" : "40%"})
                `,
              }}
            />
            <div className="flex flex-col gap-24 w-full px-6 md:px-20 max-w-container z-1000">
              {children}
            </div>
          </div>
        </main>
        <Footer variant={variant === null ? "colorful" : variant} />
      </div>
    </LayoutContext.Provider>
  );
};
