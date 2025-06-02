import { PropsWithChildren } from "react";

import { cva, VariantProps } from "class-variance-authority";
import { twMerge } from "tailwind-merge";
import React from "react";

// NOTE: this and the following negative margin styles are to hide the outer padding of the grid items, so it's flush with the container.
const gridContainer = cva("w-full overflow-hidden");

const gridVariants = cva(
  [
    "grid w-full bg-[rgba(255,255,255,0.08)] gap-[1px]",
    "border-[rgba(255,255,255,0.08)] border-t border-b",
    "-mx-10 w-[calc(100%+var(--spacing)*10*2)]",
  ],
  {
    variants: {
      columns: {
        default: "md:auto-cols-fr md:grid-flow-col",
        2: "md:grid-cols-2",
        3: "md:grid-cols-3",
        4: "md:grid-cols-4",
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
  columns,
}: VariantProps<typeof gridVariants> &
  PropsWithChildren<{ className?: string }>) => {
  const numItems = React.Children.count(children);
  const remainder =
    typeof columns === "number" ? columns - (numItems % columns) : null;

  return (
    <div className={gridContainer()}>
      <div className={grid({ columns, className })}>
        {children}
        {remainder
          ? Array.from({ length: remainder }, (_, i) => <GridItem key={i} />)
          : null}
      </div>
    </div>
  );
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
  width,
  direction,
}: PropsWithChildren<VariantProps<typeof gridItem>>) => {
  return <div className={gridItem({ width, direction })}>{children}</div>;
};
