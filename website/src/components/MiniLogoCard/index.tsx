import clsx from "clsx";
import styles from "./styles.module.css";

const MiniLogoCard = ({
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
      <a href={href} className={styles.a}>
        <div className="card__body">
          <div className={styles.container}>
            <img src={src} alt={title} className={styles.img} />
            <h4 className={styles.title}>{title}</h4>
          </div>
        </div>
      </a>
    </div>
  );
};

export default MiniLogoCard;
