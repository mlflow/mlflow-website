import { Header } from "../Header/Header";
import { Footer } from "../Footer/Footer";
import { cva, VariantProps } from "class-variance-authority";
import { createContext, PropsWithChildren, useContext } from "react";

type Props = PropsWithChildren<VariantProps<typeof wrapper>>;

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
        down: "bg-size-[100%_700px] bg-[center_top]",
        up: "bg-size-[100%_350px] bg-[center_350px]",
      },
    },
    compoundVariants: [
      {
        variant: "red",
        direction: "down",
        className:
          "bg-[linear-gradient(to_bottom,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-2.png')]",
      },
      {
        variant: "blue",
        direction: "down",
        className:
          "bg-[linear-gradient(to_bottom,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-3.png')]",
      },
      {
        variant: "colorful",
        direction: "down",
        className:
          "bg-[linear-gradient(to_bottom,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-1.png')]",
      },
      {
        variant: "red",
        direction: "up",
        className: [
          "bg-[linear-gradient(to_top,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-2.png')]",
          "",
        ],
      },
      {
        variant: "blue",
        direction: "up",
        className:
          "bg-[linear-gradient(to_top,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-3.png')]",
      },
      {
        variant: "colorful",
        direction: "up",
        className:
          "bg-[linear-gradient(to_top,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-1.png')]",
      },
    ],
    defaultVariants: {
      variant: "colorful",
      direction: "down",
    },
  },
);

const LayoutContext = createContext<"red" | "blue" | "colorful">("colorful");

export function useLayoutVariant() {
  const variant = useContext(LayoutContext);
  if (variant === undefined) {
    throw new Error("useLayoutVariant must be used within a Layout");
  }
  return variant;
}

export const Layout = ({ children, variant, direction }: Props) => {
  return (
    <LayoutContext.Provider value={variant}>
      <div className="flex flex-col min-h-screen w-full bg-[#0E1416]">
        <Header />
        <main className="flex flex-col">
          <div className={wrapper({ variant, direction })}>{children}</div>
        </main>
        <Footer variant={variant} />
      </div>
    </LayoutContext.Provider>
  );
};
