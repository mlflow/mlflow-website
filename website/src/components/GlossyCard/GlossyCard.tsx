import clsx from "clsx";
import { PropsWithChildren } from "react";

type GlossyCardProps = PropsWithChildren<{
  className?: string;
}>;

export const GlossyCard = ({ children, className }: GlossyCardProps) => {
  return (
    <div
      className={clsx(
        "relative flex h-full flex-col justify-between overflow-hidden rounded-4xl",
        "border border-white/12 bg-[linear-gradient(135deg,#1f2f63,#0e1425)] backdrop-blur-2xl shadow-[0_40px_110px_rgba(0,0,0,0.55)]",
        "transition-transform duration-200 ease-out hover:-translate-y-1 hover:border-white/25",
        className,
      )}
    >
      <div
        className="pointer-events-none absolute inset-0 opacity-55 bg-[radial-gradient(circle_at_20%_15%,rgba(255,255,255,0.12),transparent_42%),radial-gradient(circle_at_80%_10%,rgba(120,162,255,0.14),transparent_38%),radial-gradient(circle_at_50%_120%,rgba(56,189,248,0.12),transparent_50%)]"
        aria-hidden
      />
      <div
        className="pointer-events-none absolute inset-px rounded-[28px] border border-white/8"
        aria-hidden
      />
      <div className="relative z-10 flex h-full flex-col">{children}</div>
    </div>
  );
};

type GlossyCardContainerProps = PropsWithChildren<{
  className?: string;
}>;

export function GlossyCardContainer({
  children,
  className,
}: GlossyCardContainerProps) {
  return (
    <div className={clsx("grid w-full gap-8 md:grid-cols-2", className)}>
      {children}
    </div>
  );
}
