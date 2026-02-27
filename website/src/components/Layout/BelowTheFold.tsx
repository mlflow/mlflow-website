import { PropsWithChildren } from "react";
import { SocialWidget, GetStartedWithMLflow } from "..";
import { ContentType } from "../types";

interface BelowTheFoldProps extends PropsWithChildren {
  contentType?: ContentType;
  hideGetStarted?: boolean;
  hideSocialWidget?: boolean;
}

export function BelowTheFold({
  children,
  contentType,
  hideGetStarted,
  hideSocialWidget,
}: BelowTheFoldProps) {
  return (
    <>
      {!hideGetStarted && <GetStartedWithMLflow contentType={contentType} />}
      {children}
      {!hideSocialWidget && <SocialWidget />}
    </>
  );
}
