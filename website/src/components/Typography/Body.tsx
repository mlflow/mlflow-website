import { cva, type VariantProps } from "class-variance-authority";
import type { PropsWithChildren } from "react";

const styles = cva(["font-regular leading-[140%] tracking-[0%]"], {
  variants: {
    size: {
      xl: "text-[22px]",
      l: "text-[18px]",
      m: "text-[16px]",
      s: "text-[14px]",
    },
    align: {
      left: "text-left",
      center: "text-center",
    },
    margin: {
      tight: "",
      regular: "mb-4",
    },
    color: {
      default: "text-gray-600",
      white: "text-white",
    },
  },
});

export const Body = ({
  size,
  align,
  margin = "regular",
  color = "default",
  children,
}: PropsWithChildren<VariantProps<typeof styles>>) => {
  return (
    <div role="paragraph" className={styles({ size, align, margin, color })}>
      {children}
    </div>
  );
};
