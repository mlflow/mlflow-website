import { cva, type VariantProps } from "class-variance-authority";
import type { PropsWithChildren } from "react";

const styles = cva("text-center text-wrap", {
  variants: {
    level: {
      1: "font-light text-[64px] leading-[100%] tracking-[-3%]",
      2: "font-light text-[52px] leading-[120%] tracking-[-1%] text-white",
      3: "font-regular text-[40px] leading-[120%] tracking-[-1%]",
      4: "font-regular text-[32px] leading-[120%] tracking-[0%]",
      5: "font-medium text-[24px] leading-[120%] tracking-[0%]",
    },
  },
});

export const Heading = ({
  level,
  children,
}: PropsWithChildren<VariantProps<typeof styles>>) => {
  return (
    <div role="heading" aria-level={level} className={styles({ level })}>
      {children}
    </div>
  );
};
