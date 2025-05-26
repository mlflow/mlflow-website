import { cva, type VariantProps } from "class-variance-authority";
import type { PropsWithChildren } from "react";

const styles = cva("uppercase leading-[120%] tracking-[8%]", {
  variants: {
    size: {
      l: "font-medium text-[14px]",
      s: "font-semibold text-[12px]",
    },
  },
});

export const Label = ({
  size,
  children,
}: PropsWithChildren<VariantProps<typeof styles>>) => {
  return <span className={styles({ size })}>{children}</span>;
};
