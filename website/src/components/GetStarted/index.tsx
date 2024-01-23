import styles from "./styles.module.css";

const GetStarted = ({ text }: { text: string }) => {
  return (
    <div className={styles.container}>
      <a className={styles.a} href="docs/latest/getting-started/index.html">
        {text}
      </a>
    </div>
  );
};

export default GetStarted;
