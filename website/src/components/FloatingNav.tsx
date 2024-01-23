import React from "react";
import "./FloatingNav.css";

const FloatingNav = ({
  sections,
  active,
  onClick,
}: {
  sections: string[];
  active: string;
  onClick: (section: string) => void;
}) => {
  return (
    <nav className="floating-nav">
      <ul>
        {sections.map((section) => (
          <li key={section} className={active === section ? "active" : ""}>
            <a href={`#${section}`} onClick={() => onClick(section)}>
              {section.toUpperCase().replace("-", " ")}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
};

export default FloatingNav;
