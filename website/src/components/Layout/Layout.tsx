import { Header } from "../Header/Header";
import { Footer } from "../Footer/Footer";
import { cva } from "class-variance-authority";
import { createContext, PropsWithChildren, useContext } from "react";
import { useLocation } from "@docusaurus/router";
import useBaseUrl from "@docusaurus/useBaseUrl";

type Props = PropsWithChildren;

const wrapper = cva(
  "flex flex-col gap-20 bg-no-repeat w-full pt-42 pb-20 py-20",
  {
    variants: {
      variant: {
        red: "",
        blue: "",
        colorful: "",
      },
      direction: {
        down: "bg-size-[auto_1000px] 2xl:bg-size-[100%_1000px] bg-[center_top]",
        up: "bg-size-[auto_820px] 2xl:bg-size-[100%_820px] bg-[center_top]",
      },
    },
    compoundVariants: [
      {
        variant: "red",
        direction: "down",
        className: [
          "bg-[linear-gradient(to_bottom,rgba(12,20,20,0)_0%,rgba(12,20,20,0)_50%,rgba(14,20,20,100)_75%),url('/img/background-image-2.png')]",
        ],
      },
      {
        variant: "blue",
        direction: "down",
        className: [
          "bg-[linear-gradient(to_bottom,rgba(12,20,20,0)_0%,rgba(12,20,20,0)_50%,rgba(14,20,20,100)_75%),url('/img/background-image-3.png')]",
        ],
      },
      {
        variant: "colorful",
        direction: "down",
        className: [
          "bg-[linear-gradient(to_bottom,rgba(12,20,20,0)_0%,rgba(12,20,20,0)_50%,rgba(14,20,20,100)_75%),url('/img/background-image-1.png')]",
        ],
      },
      {
        variant: "red",
        direction: "up",
        className: [
          "bg-[linear-gradient(to_top,rgba(12,20,20,0)_0%,rgba(12,20,20,0)_10%,rgba(14,20,20,100)_40%),url('/img/background-image-2-flipped.png')]",
        ],
      },
      {
        variant: "blue",
        direction: "up",
        className: [
          "bg-[linear-gradient(to_top,rgba(11,20,20,0)_0%,rgba(12,20,20,0)_10%,rgba(14,20,20,100)_40%),url('/img/background-image-3-flipped.png')]",
        ],
      },
      {
        variant: "colorful",
        direction: "up",
        className: [
          "bg-[linear-gradient(to_top,rgba(11,20,20,0)_0%,rgba(12,20,20,0)_10%,rgba(14,20,20,100)_40%),url('/img/background-image-1-flipped.png')]",
        ],
      },
    ],
    defaultVariants: {
      variant: "colorful",
      direction: "down",
    },
  },
);

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

  return (
    <LayoutContext.Provider value={variant}>
      <div className="flex flex-col min-h-screen w-full bg-[#0E1416]">
        <Header layoutType={layoutType} />
        <main className="flex flex-col">
          <div className={wrapper({ variant, direction })}>
            <div className="flex flex-col gap-24 w-full px-6 md:px-20 max-w-container">
              {children}
            </div>
          </div>
        </main>
        <Footer variant={variant === null ? "colorful" : variant} />
      </div>
    </LayoutContext.Provider>
  );
};
