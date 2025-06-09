import React, { useState } from "react";
import { flushSync } from "react-dom";
import styles from "./styles.module.css";

const ExpandableGrid = ({ items, defaultVisibleCount, renderItem }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const visibleItems = isExpanded ? items : items.slice(0, defaultVisibleCount);

  return (
    <>
      <div
        className={`${styles.grid} ${!isExpanded ? styles.fadeOverlay : ""}`}
      >
        {visibleItems.map((item, index) => (
          <div key={index} className={styles.gridItem}>
            {renderItem(item, index)}
          </div>
        ))}
      </div>

      {items.length > defaultVisibleCount && (
        <div className={styles.buttonContainer}>
          <button
            onClick={(event) => {
              if (isExpanded) {
                flushSync(() => setIsExpanded(false));
                event.currentTarget.scrollIntoView({
                  behavior: "smooth",
                  block: "center",
                });
              } else {
                setIsExpanded(true);
              }
            }}
            className={styles.toggleButton}
          >
            {isExpanded ? "See Less ∧" : "See All ∨"}
          </button>
        </div>
      )}
    </>
  );
};

export default ExpandableGrid;
