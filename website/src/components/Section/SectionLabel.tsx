import { useLayoutVariant } from "../Layout/Layout";

interface Props {
  label: string;
}

export const SectionLabel = ({ label }: Props) => {
  const color = useLayoutVariant() === "blue" ? "green" : "red";
  const colorClass = color === "red" ? "bg-brand-red" : "bg-brand-teal";

  return (
    <div className="flex flex-row gap-3 justify-center items-center">
      <div className={`w-2 h-2 rotate-45 ${colorClass}`}></div>
      <div className="text-sm font-medium uppercase text-white">{label}</div>
    </div>
  );
};
