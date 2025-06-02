import useBaseUrl from "@docusaurus/useBaseUrl";
import { cx } from "class-variance-authority";

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

const container = cx("shrink-0 animate-slide-left-infinite");

export const LogosCarousel = () => {
  const items = images.map((image, index) => (
    <img
      key={index}
      className="inline h-16 mx-20 opacity-20"
      src={useBaseUrl(image)}
    />
  ));
  return (
    <div className="flex overflow-x-hidden py-10 [mask-image:_linear-gradient(to_right,_transparent_0,_white_128px,white_calc(100%-128px),_transparent_100%)] w-full">
      <div className={container}>{items}</div>
      <div className={container} aria-hidden>
        {items}
      </div>
    </div>
  );
};
