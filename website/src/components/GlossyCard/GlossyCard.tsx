import { PropsWithChildren } from "react";

export const GlossyCard = ({ children }: PropsWithChildren) => {
  return (
    <div className="flex flex-col rounded-4xl overflow-hidden bg-white/16 backdrop-blur-[8px] w-full justify-between border-white/20 border shadow-2xl">
      {children}
    </div>
  );
};

export function GlossyCardContainer({ children }: PropsWithChildren) {
  return <div className="flex flex-col md:flex-row gap-10">{children}</div>;
}
