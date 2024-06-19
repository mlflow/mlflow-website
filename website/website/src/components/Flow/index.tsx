import styles from "./styles.module.css";

const Flow = () => {
  return (
    <div className={styles.container}>
      <div className={styles.left}>
        <img className={styles.img} src="img/flow.svg" alt="" />
      </div>
      <div className={styles.center}>
        <img className={styles.centerImg} src="img/hero.png" alt="" />
      </div>
      <div className={styles.right}>
        <img className={styles.img} src="img/flow.svg" alt="" />
      </div>
    </div>
  );
};

export default Flow;
