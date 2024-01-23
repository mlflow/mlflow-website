import { Release } from "../../posts";
import ArrowText from "../ArrowText";
import clsx from "clsx";
import styles from "./styles.module.css";

const ReleaseNote = ({ release }: { release: Release }) => {
  const { title, authors, date, path, version } = release;
  const author = authors[0];
  return (
    <div className={clsx("card", styles.card)}>
      <div className={clsx("card__body", styles.cardBody)}>
        <a className={styles.a} href={path}>
          <p className={styles.p}>
            {new Date(date).toLocaleDateString("en-US", {
              year: "numeric",
              month: "short",
              day: "numeric",
            })}
          </p>
          <h4>{title}</h4>
          <p>{`We're happy to announce the release of MLflow ${version}.`}</p>
        </a>
      </div>
    </div>
  );
};

export default ReleaseNote;
