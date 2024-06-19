import React from "react";
import "./CommunityCard.css";

type CommunityCardProps = {
  className: string;
  title: string;
  content: React.ReactNode;
  dotPositions: Set<"top" | "bottom" | "left" | "right">;
};
const CommunityCard = ({
  className,
  title,
  content,
  dotPositions,
}: CommunityCardProps) => {
  const dots = [...Array.from(dotPositions)].map((dotPosition, idx) => {
    const circle = (
      <svg
        className="connector-circle"
        viewBox="0 0 10 10"
        xmlns="http://www.w3.org/2000/svg"
      >
        <circle cx="50%" cy="50%" r="4.5" />
      </svg>
    );

    let dotClassName;
    switch (dotPosition) {
      case "top":
        dotClassName = "dot-top";
        break;
      case "bottom":
        dotClassName = "dot-bottom";
        break;
      case "left":
        dotClassName = "dot-left";
        break;
      case "right":
        dotClassName = "dot-right";
        break;
      default:
        return null;
    }

    return (
      <div className={dotClassName} key={idx}>
        {circle}
      </div>
    );
  });

  return (
    <div className={`community-card ${className}`}>
      {dots}
      <h3>{title}</h3>
      {content}
    </div>
  );
};

export default CommunityCard;
