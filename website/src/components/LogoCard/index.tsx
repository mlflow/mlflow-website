import clsx from "clsx";
import styles from "./styles.module.css";

const LogoCard = ({
  title,
  src,
  href,
}: {
  title: string;
  src: string;
  href: string;
}) => {
  return (
    <div className={clsx("card", styles.card)}>
      <div className={clsx("card__body", styles.cardBody)}>
        <a href={href} className={styles.a}>
          <div className={styles.container}>
            <img src={src} className={styles.img} />
            <div>
              <h2 className={styles.title}>{title}</h2>
              See the docs ↗︎
            </div>
          </div>
        </a>
      </div>
    </div>
  );
};

export default LogoCard;
