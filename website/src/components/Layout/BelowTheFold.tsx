import { PropsWithChildren } from "react";
import { SocialWidget, GetStartedWithMLflow } from "..";

export function BelowTheFold({ children }: PropsWithChildren) {
  return (
    <>
      <GetStartedWithMLflow />
      {children}
      <SocialWidget />
    </>
  );
}
