import React from "react";
import styles from "./styles.module.css";
import clsx from "clsx";

const CenterGrid = ({ children }: { children: React.ReactNode }) => {
  return <div className={styles.container}>{children}</div>;
};

export default CenterGrid;
