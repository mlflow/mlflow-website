import { PropsWithChildren } from "react";
import { SocialWidget, GetStartedWithMLflow } from "..";
import { ContentType } from "../types";

interface BelowTheFoldProps extends PropsWithChildren {
  contentType?: ContentType;
  hideGetStarted?: boolean;
}

export function BelowTheFold({
  children,
  contentType,
  hideGetStarted,
}: BelowTheFoldProps) {
  return (
    <>
      {!hideGetStarted && <GetStartedWithMLflow contentType={contentType} />}
      {children}
      <SocialWidget />
    </>
  );
}
