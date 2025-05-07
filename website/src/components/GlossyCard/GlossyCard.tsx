import { PropsWithChildren } from "react";
import { ReactNode } from "react";

interface Props extends PropsWithChildren {
  image: ReactNode;
}

export const GlossyCard = ({ image, children }: Props) => {
  return (
    <div className="flex flex-col rounded-4xl bg-white/16 backdrop-blur-[24px] w-full">
      <div className="flex flex-col p-6 gap-4">{children}</div>
      <div className="w-full bg-[#0E1416]/60 min-h-[270px] rounded-b-4xl"></div>
    </div>
  );
};
