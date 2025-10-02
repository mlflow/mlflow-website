import { PropsWithChildren } from "react";
import { SocialWidget, GetStartedWithMLflow } from "..";
import { ContentType } from "../types";

interface BelowTheFoldProps extends PropsWithChildren {
  contentType?: ContentType;
}

export function BelowTheFold({ children, contentType }: BelowTheFoldProps) {
  return (
    <>
      <GetStartedWithMLflow contentType={contentType} />
      {children}
      <SocialWidget />
    </>
  );
}
