interface Props {
  label: string;
  color: "red" | "green";
}

export const SectionLabel = ({ label, color }: Props) => {
  const colorClass = color === "red" ? "bg-brand-red" : "bg-brand-teal";

  return (
    <div className="flex flex-row gap-3 justify-center items-center">
      <div className={`w-2 h-2 rotate-45 ${colorClass}`}></div>
      <div className="text-sm font-medium uppercase text-white">{label}</div>
    </div>
  );
};
