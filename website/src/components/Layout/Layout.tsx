import { Header } from "../Header/Header";
import { Footer } from "../Footer/Footer";
import { cva } from "class-variance-authority";
import { createContext, PropsWithChildren, useContext } from "react";
import { useLocation } from "@docusaurus/router";
import useBaseUrl from "@docusaurus/useBaseUrl";
import { GradientWrapper } from "../GradientWrapper/GradientWrapper";

type Props = PropsWithChildren;

const wrapper = cva("flex flex-col gap-20 bg-no-repeat w-full py-32");

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
  const blog = useBaseUrl("/blog");

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
  } else if (pathname.startsWith(blog)) {
    return "blog";
  } else if (pathname.startsWith(useBaseUrl("/releases"))) {
    return "releases";
  } else if (pathname === useBaseUrl("/")) {
    return "home";
  } else {
    return "default";
  }
}

export const Layout = ({ children }: Props) => {
  const location = useLocation();

  const layoutType = getLayoutType(location.pathname);
  const hasReducedGradient = layoutType === "blog" || layoutType === "releases";
  const variant = layoutType.startsWith("genai")
    ? "red"
    : layoutType.startsWith("classical-ml")
      ? "blue"
      : "colorful";
  const direction = layoutType.endsWith("subpage") ? "up" : "down";

  return (
    <LayoutContext.Provider value={variant}>
      <div className="flex flex-col min-h-screen w-full bg-[#0E1416]">
        <Header />
        <main className="flex flex-col">
          <GradientWrapper
            className={wrapper()}
            height={hasReducedGradient ? 56 : 820}
            variant={variant}
            direction={direction}
          >
            <div className="flex flex-col gap-36 w-full px-6 md:px-20 max-w-container">
              {children}
            </div>
          </GradientWrapper>
        </main>
        <Footer variant={variant} />
      </div>
    </LayoutContext.Provider>
  );
};
