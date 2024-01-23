import React from "react";

const ArrowText = ({ text }: { text: React.ReactNode }): JSX.Element => {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: 8,
      }}
    >
      <img src="img/arrow.svg" alt="" style={{ height: "1.25rem" }} />
      <div style={{ lineHeight: "1.25rem" }}>{text}</div>
    </div>
  );
};

export default ArrowText;
