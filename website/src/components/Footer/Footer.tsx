import { cva, type VariantProps } from "class-variance-authority";

import Logo from "@site/static/img/mlflow-logo-white.svg";

import { FooterMenuItem } from "../FooterMenuItem/FooterMenuItem";
import { MLFLOW_DOCS_URL } from "@site/src/constants";

const footerVariants = cva(
  "pb-30 flex flex-col pt-30 bg-bottom bg-no-repeat bg-cover text-center bg-size-[auto_360px] 2xl:bg-size-[100%_360px]",
  {
    variants: {
      variant: {
        blue: [
          "bg-[linear-gradient(to_top,rgba(12,20,20,0)_0%,rgba(12,20,20,0)_10%,rgba(14,20,20,100)_40%),url('/img/footer-blue-bg.png')]",
        ],
        red: [
          "bg-[linear-gradient(to_top,rgba(12,20,20,0)_0%,rgba(12,20,20,0)_10%,rgba(14,20,20,100)_40%),url('/img/footer-red-bg.png')]",
        ],
        colorful: [
          "bg-[linear-gradient(to_top,rgba(12,20,20,0)_0%,rgba(12,20,20,0)_10%,rgba(14,20,20,100)_40%),url('/img/footer-colorful-bg.png')]",
        ],
      },
    },
  },
);

export const Footer = ({ variant }: VariantProps<typeof footerVariants>) => {
  return (
    <footer className={footerVariants({ variant })}>
      <div className="flex flex-row justify-between items-start px-6 lg:px-20 gap-10 xs:gap-0 max-w-container">
        <div className="flex flex-col gap-8">
          <Logo className="h-[36px] shrink-0" />
          <div className="text-xs text-gray-800 text-left md:text-nowrap md:w-0">
            © 2025 MLflow Project, a Series of LF Projects, LLC.
          </div>
        </div>

        <div className="flex flex-col md:flex-row gap-10">
          <FooterMenuItem href="/">Product</FooterMenuItem>
          <FooterMenuItem href="/releases">Releases</FooterMenuItem>
          <FooterMenuItem href="/blog">Blog</FooterMenuItem>
          <FooterMenuItem href={MLFLOW_DOCS_URL}>Docs</FooterMenuItem>
        </div>
      </div>
    </footer>
  );
};
