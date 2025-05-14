import Link from "@docusaurus/Link";

import "./FooterMenuItem.module.css";

interface Props {
  href: string;
  label: string;
}

export const FooterMenuItem = ({ href, label }: Props) => {
  return (
    <div className="min-w-[120px]">
      <Link href={href}>{label}</Link>
    </div>
  );
};
