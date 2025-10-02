import { PropsWithChildren } from "react";
import { SocialWidget, GetStartedWithMLflow } from "..";

type ContentType = "genai" | "classical-ml";

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
