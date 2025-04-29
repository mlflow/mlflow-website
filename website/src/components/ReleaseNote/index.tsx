import { Release } from "../../posts";
import clsx from "clsx";
import styles from "./styles.module.css";
import Link from "@docusaurus/Link";

const ReleaseNote = ({ release }: { release: Release }) => {
  const { title, authors, date, path, version } = release;
  const author = authors[0];
  return (
    <div className={clsx("card", styles.card)}>
      <div className={clsx("card__body", styles.cardBody)}>
        <Link className={styles.a} to={path}>
          <p className={styles.p}>
            {new Date(date).toLocaleDateString("en-US", {
              year: "numeric",
              month: "short",
              day: "numeric",
            })}
          </p>
          <h4>{title}</h4>
          <p>{`We're happy to announce the release of MLflow ${version}.`}</p>
        </Link>
      </div>
    </div>
  );
};

export default ReleaseNote;
