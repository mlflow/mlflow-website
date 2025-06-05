import { cva, type VariantProps } from "class-variance-authority";
import type { PropsWithChildren } from "react";

const styles = cva("text-balance", {
  variants: {
    level: {
      1: "font-light text-[64px] leading-[100%] tracking-[-3%] text-center",
      2: "font-light text-[52px] leading-[120%] tracking-[-1%] text-center text-white",
      3: "font-regular text-[40px] leading-[120%] tracking-[-1%]",
      4: "font-regular text-[32px] leading-[120%] tracking-[0%]",
      5: "font-medium text-[24px] leading-[120%] tracking-[0%]",
    },
  },
});

type Props = PropsWithChildren<VariantProps<typeof styles>> & {
  "aria-level"?: 1 | 2 | 3 | 4 | 5;
};

export const Heading = ({
  level,
  children,
  "aria-level": ariaLevel = level, // use stylistic level by default, but allow aria-level override
}: Props) => {
  return (
    <div role="heading" aria-level={ariaLevel} className={styles({ level })}>
      {children}
    </div>
  );
};
