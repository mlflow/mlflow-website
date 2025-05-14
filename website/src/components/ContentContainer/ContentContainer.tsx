import { ReactNode } from "react";

export const ContentContainer = ({ children }: { children: ReactNode }) => {
  return <div className="flex flex-col max-w-7xl mx-auto">{children}</div>;
};
