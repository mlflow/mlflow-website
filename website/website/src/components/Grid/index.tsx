import styles from "./styles.module.css";

const Grid = ({ children }: { children: React.ReactNode }) => {
  return <div className={styles.container}>{children}</div>;
};

export const DoubleGrid = ({ children }: { children: React.ReactNode }) => {
  return <div className={styles.containerDouble}>{children}</div>;
};

export default Grid;
