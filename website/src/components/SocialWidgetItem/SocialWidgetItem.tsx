import Link from "@docusaurus/Link";
import IconRight from "@site/static/img/social/icon-right.svg";

interface Props {
  href: string;
  icon: React.ReactNode;
  label: string;
  description: string;
}

export const SocialWidgetItem = ({ href, icon, label, description }: Props) => {
  return (
    <Link
      href={href}
      target="_blank"
      className="flex sm:flex-row md:flex-col xl:flex-row group w-full sm:items-center md:items-start xl:items-center p-8 h-full cursor-pointer gap-6 hover:bg-white/4"
    >
      {icon}
      <div className="flex flex-row justify-between items-center w-full">
        <div className="flex flex-col">
          <span className="text-white text-lg font-medium">{label}</span>
          <span className="text-gray-600 text-base">{description}</span>
        </div>
        <div className="invisible group-hover:visible">
          <IconRight />
        </div>
      </div>
    </Link>
  );
};
