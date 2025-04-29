import React from "react";
import styles from "./styles.module.css";
import clsx from "clsx";

interface LearnCardProps {
  title: string;
  content: React.ReactNode;
  href: string;
  img?: string;
}

const LearnCard: React.FC<LearnCardProps> = ({ title, content, href, img }) => {
  return (
    <a href={href} className={clsx("card", styles.card, styles.a)}>
      <div className={clsx("card__body", styles.cardBody)}>
        <div className={styles.container}>
          <div className={styles.thumbnail}>
            <img src={img} alt="" className={styles.img} />
          </div>
          <div className={styles.body}>
            <h2>{title}</h2>
            {content}
          </div>
        </div>
      </div>
    </a>
  );
};
export default LearnCard;
