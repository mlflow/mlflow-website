interface Props {
  children: React.ReactNode;
}

export const LogosCarousel = ({ children }: Props) => {
  return (
    <div className="group relative overflow-hidden whitespace-nowrap py-10 [mask-image:_linear-gradient(to_right,_transparent_0,_white_128px,white_calc(100%-128px),_transparent_100%)] w-full">
      <div className="animate-slide-left-infinite inline-block w-full">
        {children}
      </div>

      <div className="animate-slide-left-infinite inline-block w-full">
        {children}
      </div>
    </div>
  );
};
