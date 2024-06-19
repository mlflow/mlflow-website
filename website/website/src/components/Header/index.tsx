import styles from "./styles.module.css";

export const H1 = ({ children }: { children: React.ReactNode }) => {
  return (
    <div className={styles.container}>
      <h1 className={styles.h1}>{children}</h1>
    </div>
  );
};

export const H2 = ({ children }: { children: React.ReactNode }) => {
  return (
    <div className={styles.container}>
      <h2 className={styles.h2}>{children}</h2>
    </div>
  );
};

export const H3 = ({ children }: { children: React.ReactNode }) => {
  return (
    <div className={styles.container}>
      <h3 className={styles.h3}>{children}</h3>
    </div>
  );
};
