import React from "react";
import { AmbassadorCard } from "../AmbassadorCard/AmbassadorCard";
import ambassadors from "../../pages/ambassadors.json";

const AmbassadorGrid = () => (
  <div
    style={{
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
      gap: "20px",
    }}
  >
    {ambassadors.map((ambassador) => (
      <AmbassadorCard key={ambassador.title} {...ambassador} />
    ))}
  </div>
);

export default AmbassadorGrid;
