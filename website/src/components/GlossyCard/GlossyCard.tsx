import { PropsWithChildren } from "react";
import { ReactNode } from "react";

export const GlossyCard = ({ children }: PropsWithChildren) => {
  return (
    <div className="flex flex-col rounded-4xl bg-white/16 backdrop-blur-[24px] w-full justify-between">
      {children}
    </div>
  );
};

export function GlossyCardContainer({ children }: PropsWithChildren) {
  return <div className="flex flex-col md:flex-row gap-10">{children}</div>;
}
