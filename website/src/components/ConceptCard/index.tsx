import styles from "./styles.module.css";
import clsx from "clsx";

const ConceptCard = ({
  title,
  logo,
  href,
}: {
  title: string;
  logo: string;
  href: string;
}) => {
  return (
    <div className={clsx("card", styles.card)}>
      <a href={href} className={styles.a}>
        <div className="card__body">
          <img src={logo} alt={logo} />
          <div style={{ marginTop: 24 }}></div>
          <h3>{title}</h3>
        </div>
      </a>
    </div>
  );
};

export default ConceptCard;
