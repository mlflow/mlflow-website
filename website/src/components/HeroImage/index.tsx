export function HeroImage(props: React.ComponentProps<"img">) {
  return (
    <div className="relative w-full max-w-[800px] rounded-[16px] overflow-hidden mx-auto">
      <img {...props} />
      <div className="absolute inset-0 bg-black/5 pointer-events-none" />
    </div>
  );
}
