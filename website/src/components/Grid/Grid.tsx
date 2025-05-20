import { PropsWithChildren } from "react";

import { cva, VariantProps } from "class-variance-authority";
import { twMerge } from "tailwind-merge";

const gridVariants = cva(
  [
    "grid w-full bg-[rgba(255,255,255,0.08)] gap-[1px]",
    "border-[rgba(255,255,255,0.08)] border-t border-b",
  ],
  {
    variants: {
      columns: {
        2: "md:grid-cols-2",
        3: "md:grid-cols-3",
        4: "md:grid-cols-4",
      },
    },
  },
);

export const grid: typeof gridVariants = (variants) =>
  twMerge(gridVariants(variants));

export const Grid = ({
  children,
  className,
  columns,
}: VariantProps<typeof gridVariants> &
  PropsWithChildren<{ className?: string }>) => {
  return <div className={grid({ columns, className })}>{children}</div>;
};

const gridItemVariants = cva(
  ["flex flex-col items-start p-10 gap-20 justify-between w-full bg-[#0E1416]"],
  {
    variants: {
      width: {
        narrow: "",
        wide: "md:col-span-2 md:flex-row *:flex-1 md:items-center",
      },
      direction: {
        reverse: "md:flex-col-reverse",
      },
    },
    compoundVariants: [
      {
        width: "wide",
        direction: "reverse",
        className: "md:flex-row-reverse",
      },
    ],
    defaultVariants: {
      width: "narrow",
    },
  },
);

export const gridItem: typeof gridItemVariants = (variants) =>
  twMerge(gridItemVariants(variants));

export const GridItem = ({
  children,
  className,
  width,
  direction,
}: VariantProps<typeof gridItem> &
  PropsWithChildren<{ className?: string }>) => {
  return (
    <div className={gridItem({ width, direction, className })}>{children}</div>
  );
};
