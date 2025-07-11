import { cva } from "class-variance-authority";

import Logo from "@site/static/img/mlflow-logo-white.svg";

import { FooterMenuItem } from "../FooterMenuItem/FooterMenuItem";
import { MLFLOW_DOCS_URL } from "@site/src/constants";
import {
  GradientWrapper,
  type Variant,
} from "../GradientWrapper/GradientWrapper";

const footerVariants = cva("pb-30 flex flex-col pt-30");

export const Footer = ({ variant }: { variant: Variant }) => {
  return (
    <GradientWrapper
      variant={variant}
      radial
      className={footerVariants()}
      element="footer"
    >
      <div className="flex flex-row justify-between items-start px-6 lg:px-20 gap-10 xs:gap-0 max-w-container">
        <div className="flex flex-col gap-8">
          <Logo className="h-[36px] shrink-0" />
          <div className="text-xs text-gray-800 text-left md:text-nowrap md:w-0">
            © 2025 MLflow Project, a Series of LF Projects, LLC.
          </div>
        </div>

        <div className="flex flex-col flex-wrap justify-end md:text-right md:flex-row gap-x-10 lg:gap-x-20 gap-y-5 w-2/5 md:w-auto md:pt-2 max-w-fit">
          <FooterMenuItem href="/">Components</FooterMenuItem>
          <FooterMenuItem href="/releases">Releases</FooterMenuItem>
          <FooterMenuItem href="/blog">Blog</FooterMenuItem>
          <FooterMenuItem href={MLFLOW_DOCS_URL}>Docs</FooterMenuItem>
          <FooterMenuItem href="/ambassadors">
            Ambassador Program
          </FooterMenuItem>
        </div>
      </div>
    </GradientWrapper>
  );
};
