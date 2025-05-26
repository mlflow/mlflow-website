import { cva, type VariantProps } from "class-variance-authority";
import type { PropsWithChildren } from "react";

const styles = cva(
  ["font-regular leading-[140%] tracking-[0%]", "text-gray-600"],
  {
    variants: {
      size: {
        l: "text-[18px]",
        m: "text-[16px]",
        s: "text-[14px]",
      },
    },
  },
);

export const Body = ({
  size,
  children,
}: PropsWithChildren<VariantProps<typeof styles>>) => {
  return <p className={styles({ size })}>{children}</p>;
};
