interface Props {
  label: string;
  color: 'red' | 'green'
}

export const SectionLabel = ({ label, color }: Props) => {
  const colorClass = color === 'red' ? 'bg-[#EB1700]' : 'bg-[#44EDBC]';

  return (
    <div className="flex flex-row gap-3 justify-center items-center">
      <div className={`w-2 h-2 rotate-45 ${colorClass}`}></div>
      <div className="text-sm font-medium uppercase">{label}</div>
    </div>
  )
}
