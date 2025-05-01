import { PropsWithChildren } from "react";

export const Grid = ({ children }: PropsWithChildren) => {
  return <div className="flex flex-col w-full">{children}</div>;
};

export const GridRow = ({ children }: PropsWithChildren) => {
  return (
    <div className="flex flex-col md:flex-row border-[rgba(255,255,255,0.08)] md:border-t md:last:border-b w-full">
      {children}
    </div>
  );
};

export const GridItem = ({ children }: PropsWithChildren) => {
  return (
    <div className="flex flex-col border-[rgba(255,255,255,0.08)] border-t last:border-b md:last:border-b-0 md:border-t-0 md:border-l md:first:border-l-0 py-10 px-0 md:px-10 md:first:pl-0 md:last:pr-0 w-full">
      {children}
    </div>
  );
};
