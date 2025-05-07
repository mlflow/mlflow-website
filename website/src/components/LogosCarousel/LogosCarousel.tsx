interface Props {
  images: string[];
}

export const LogosCarousel = ({ images }: Props) => {
  return (
    <div className="group relative overflow-hidden whitespace-nowrap py-10 [mask-image:_linear-gradient(to_right,_transparent_0,_white_128px,white_calc(100%-128px),_transparent_100%)] w-full">
      <div className="animate-slide-left-infinite inline-block w-full">
        {images.map((image) => (
          <img className="inline h-16 mx-20 opacity-20" src={image} />
        ))}
      </div>
    </div>
  );
};
