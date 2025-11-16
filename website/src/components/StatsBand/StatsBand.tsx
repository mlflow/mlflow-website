type Stat = {
  value: string;
  label: string;
};

const stats: Stat[] = [
  { value: "25MM+", label: "PyPI downloads / month" },
  { value: "23,000+", label: "GitHub stars" },
  { value: "2MM+", label: "Docker pulls / month" },
];

export const StatsBand = () => {
  return (
      <div className="grid grid-cols-1 gap-4 text-center sm:grid-cols-3 mx-24">
        {stats.map((stat, idx) => (
          <div key={idx} className="flex flex-col items-center gap-1.5">
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#c7d2fe] via-[#93c5fd] to-[#67e8f9] text-3xl sm:text-4xl font-semibold leading-tight">
              {stat.value}
            </span>
            <span className="text-xs sm:text-sm text-white/70">{stat.label}</span>
          </div>
        ))}
      </div>
  );
};
