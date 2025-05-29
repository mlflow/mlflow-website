import { cva, type VariantProps } from "class-variance-authority";

import Logo from "@site/static/img/mlflow-logo-white.svg";

import { FooterMenuItem } from "../FooterMenuItem/FooterMenuItem";
import { cn } from "../../utils";

const footerVariants = cva(
  "pb-150 flex flex-col pt-37 bg-linear-to-b from-brand-black to-brand-black bg-bottom bg-no-repeat bg-cover w-full",
  {
    variants: {
      variant: {
        blue: ` bg-[url('/img/footer-blue-bg.png')] bg-size-[100%_340px]`,
        red: `bg-[url('/img/footer-red-bg.png')] bg-size-[100%_340px]`,
        colorful: `bg-[url('/img/footer-colorful-bg.png')] bg-size-[100%_340px]`,
      },
    },
  },
);

export const Footer = ({ variant }: VariantProps<typeof footerVariants>) => {
  return (
    <footer className={cn(footerVariants({ variant }))}>
      <div className="flex flex-row justify-between items-start md:items-center px-6 md:px-20 gap-10 xs:gap-0 max-w-container">
        <Logo className="h-[36px]" />

        <div className="flex flex-col md:flex-row gap-10">
          <FooterMenuItem href="/">Product</FooterMenuItem>
          <FooterMenuItem href="/releases">Releases</FooterMenuItem>
          <FooterMenuItem href="/blog">Blog</FooterMenuItem>
          <FooterMenuItem href="/docs/latest" data-noBrokenLinkCheck>
            Docs
          </FooterMenuItem>
        </div>
      </div>
    </footer>
  );
};
