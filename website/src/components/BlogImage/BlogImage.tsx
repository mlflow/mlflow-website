import { CSSProperties } from "react";

type Props = {
  src: string;
  alt: string;
  width?: string;
  center?: boolean;
  style?: CSSProperties;
};

export function BlogImage({
  src,
  alt,
  width = "100%",
  center = false,
  style,
}: Props) {
  return (
    <img
      src={src}
      alt={alt}
      width={width}
      className={`rounded-lg shadow-md mb-4${center ? " mx-auto block" : ""}`}
      style={style}
    />
  );
}
