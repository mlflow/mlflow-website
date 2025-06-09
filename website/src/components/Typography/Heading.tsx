import { cva, type VariantProps } from "class-variance-authority";
import type { PropsWithChildren } from "react";

const styles = cva("hyphens-auto break-word", {
  variants: {
    level: {
      1: "text-balance font-light text-[64px] leading-[100%] tracking-[-3%] text-center",
      2: "text-wrap font-light text-[52px] leading-[120%] tracking-[-1%] text-center text-white",
      3: "text-wrap font-regular text-[40px] leading-[120%] tracking-[-1%]",
      4: "text-wrap font-regular text-[32px] leading-[120%] tracking-[0%]",
      5: "text-wrap font-medium text-[24px] leading-[120%] tracking-[0%]",
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
    <div
      role="heading"
      aria-level={ariaLevel}
      className={styles({ level })}
      lang="en"
    >
      {children}
    </div>
  );
};
