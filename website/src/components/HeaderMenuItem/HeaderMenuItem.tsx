import Link from "@docusaurus/Link";
import { cn } from "../../utils";
import DownIcon from "@site/static/img/chevron-down-small.svg";
interface Props {
  href?: string;
  label: string;
  className?: string;
  onMouseEnter?: () => void;
  hasDropdown?: boolean;
}

export const HeaderMenuItem = ({
  href,
  label,
  className,
  onMouseEnter,
  hasDropdown,
  ...props
}: Props) => {
  return (
    <Link
      href={href}
      className={cn(
        "flex items-center gap-2 py-2 text-white text-lg w-full md:w-auto cursor-pointer transition-colors duration-200 hover:!text-white/60",
        className,
      )}
      {...props}
    >
      {label}

      {hasDropdown && <DownIcon className="w-6 h-6" />}
    </Link>
  );
};
