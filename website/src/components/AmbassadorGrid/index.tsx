import React from "react";
import { createAmbassadorCard } from "../AmbassadorCard";
import ambassadors from "../../pages/ambassadors.json";

const AmbassadorGrid = () => (
  <div
    style={{
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
      gap: "20px",
    }}
  >
    {ambassadors.map((ambassador, index) => (
      <React.Fragment key={index}>
        {createAmbassadorCard({
          title: ambassador.title,
          role: ambassador.role,
          company: ambassador.company,
          companyLink: ambassador.companyLink,
          personalLink: ambassador.personalLink,
          img: ambassador.img,
        })}
      </React.Fragment>
    ))}
  </div>
);

export default AmbassadorGrid;
