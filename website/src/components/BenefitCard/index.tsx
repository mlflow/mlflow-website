import styles from "./styles.module.css";
import clsx from "clsx";

const BenefitCard = ({ title, body }: { title: string; body: string }) => {
  return (
    <div className={clsx("card", styles.card)}>
      <div className="card__body">
        <h2>{title}</h2>
        <p style={{ fontSize: 24 }}>{body}</p>
      </div>
    </div>
  );
};

export default BenefitCard;
