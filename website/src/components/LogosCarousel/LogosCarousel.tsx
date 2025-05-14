import useBaseUrl from "@docusaurus/useBaseUrl";

const images = [
  "/img/companies/databricks.svg",
  "/img/companies/microsoft.svg",
  "/img/companies/meta.svg",
  "/img/companies/mosaicml.svg",
  "/img/companies/zillow.svg",
  "/img/companies/toyota.svg",
  "/img/companies/booking.svg",
  "/img/companies/wix.svg",
  "/img/companies/accenture.svg",
  "/img/companies/asml.svg",
];

export const LogosCarousel = () => {
  return (
    <div className="group relative overflow-hidden whitespace-nowrap py-10 [mask-image:_linear-gradient(to_right,_transparent_0,_white_128px,white_calc(100%-128px),_transparent_100%)] w-full">
      <div className="animate-slide-left-infinite inline-block w-full">
        {images.map((image) => (
          <img
            className="inline h-16 mx-20 opacity-20"
            src={useBaseUrl(image)}
          />
        ))}
      </div>
    </div>
  );
};
