export function HeroImage(props: React.ComponentProps<"img">) {
  return (
    <div className="relative w-full max-w-[1210px] mx-auto px-4">
      <img className="rounded-[16px]" {...props} />
    </div>
  );
}
