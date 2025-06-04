export function HeroImage(props: React.ComponentProps<"img">) {
  return (
    <div className="w-full max-w-[800px] rounded-lg overflow-hidden mx-auto">
      <img {...props} />
    </div>
  );
}
