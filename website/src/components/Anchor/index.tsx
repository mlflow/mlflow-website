import React from "react";
import styles from "./styles.module.css";

const Anchor = React.forwardRef<HTMLAnchorElement, { id: string }>(
  ({ id }, ref) => {
    return (
      <div className={styles.container}>
        <a id={id} ref={ref} className={styles.a} href={`#${id}`}>
          {id.replace(/-/g, " ")}
        </a>
      </div>
    );
  }
);

export default Anchor;
