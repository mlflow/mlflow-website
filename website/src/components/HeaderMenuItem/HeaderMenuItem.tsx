import { cn } from "../../utils";

interface Props {
  href: string;
  label: string;
  className?: string;
}

export const HeaderMenuItem = ({ href, label, className }: Props) => {
  return (
    <a
      href={href}
      className={cn("block py-2 text-gray-900 w-full md:w-auto", className)}
    >
      {label}
    </a>
  );
};
