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
        default: "md:auto-cols-fr md:grid-flow-col",
        2: "md:grid-cols-2",
        3: "md:grid-cols-3",
        4: "md:grid-cols-4",
      },
      "lg-columns": {
        2: "lg:grid-cols-2",
        3: "lg:grid-cols-3",
        4: "lg:grid-cols-4",
      },
    },
    defaultVariants: {
      columns: "default",
    },
  },
);

export const grid: typeof gridVariants = (variants) =>
  twMerge(gridVariants(variants));

export const Grid = ({
  children,
  className,
  ...variants
}: VariantProps<typeof gridVariants> &
  PropsWithChildren<{ className?: string }>) => {
  return <div className={grid({ className, ...variants })}>{children}</div>;
};

const gridItemVariants = cva(
  ["flex flex-col items-start p-10 gap-20 justify-between w-full bg-[#0E1416]"],
  {
    variants: {
      width: {
        narrow: "",
        wide: "md:col-span-2 md:flex-row md:*:flex-1 md:items-center",
      },
      "lg-width": {
        narrow: "lg:col-span-1",
        wide: "lg:col-span-2 lg:flex-row lg:*:flex-1 lg:items-center",
      },
      direction: {
        normal: "md:flex-col",
        reverse: "md:flex-col-reverse",
      },
      "lg-direction": {
        normal: "lg:flex-col",
        reverse: "lg:flex-col-reverse",
      },
    },
    compoundVariants: [
      {
        width: "wide",
        direction: "reverse",
        className: "md:flex-row-reverse",
      },
      {
        width: "wide",
        direction: "normal",
        className: "md:flex-row",
      },
      {
        "lg-width": "wide",
        "lg-direction": "reverse",
        className: "lg:flex-row-reverse",
      },
      {
        "lg-width": "wide",
        "lg-direction": "normal",
        className: "lg:flex-row",
      },
    ],
    defaultVariants: {
      width: "narrow",
      direction: "normal",
    },
  },
);

export const gridItem: typeof gridItemVariants = (variants) =>
  twMerge(gridItemVariants(variants));

export const GridItem = ({
  children,
  className,
  ...variants
}: VariantProps<typeof gridItem> &
  PropsWithChildren<{ className?: string }>) => {
  return <div className={gridItem({ className, ...variants })}>{children}</div>;
};
