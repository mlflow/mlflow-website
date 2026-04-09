import React, { ReactNode, useState } from "react";
import { flushSync } from "react-dom";
import styles from "./styles.module.css";

interface ExpandableGridProps<T> {
  items: T[];
  defaultVisibleCount: number;
  renderItem: (item: T, index: number) => ReactNode;
  seeMoreLabel?: string;
  seeLessLabel?: string;
}

const ExpandableGrid = <T,>({
  items,
  defaultVisibleCount,
  renderItem,
  seeMoreLabel = "See More ∨",
  seeLessLabel = "See Less ∧",
}: ExpandableGridProps<T>) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const visibleItems = isExpanded ? items : items.slice(0, defaultVisibleCount);
  const gridId = "expandable-grid";

  return (
    <>
      <div
        id={gridId}
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
            aria-expanded={isExpanded}
            aria-controls={gridId}
          >
            {isExpanded ? seeLessLabel : seeMoreLabel}
          </button>
        </div>
      )}
    </>
  );
};

export default ExpandableGrid;
