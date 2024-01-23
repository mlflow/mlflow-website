import styles from "./styles.module.css";
import Arrow from "../community-section/Arrow";
import clsx from "clsx";

const FeatureCard = ({
  items,
  img,
  href,
}: {
  items: string[];
  img: string;
  href: string;
}) => {
  return (
    <div className={clsx("card", styles.card)}>
      <div className="card__body" style={{ padding: 0 }}>
        <div className={styles.container}>
          <div className={styles.thumbnail}>
            <ul className={styles.ul}>
              {items.map((item, idx) => (
                <li key={idx} className={styles.li}>
                  {item}
                </li>
              ))}
            </ul>
            <div className={styles.arrow}>
              <Arrow />
              <a className={styles.a} href={href}>
                See how in the docs
              </a>
            </div>
          </div>
          <div className={styles.body}>
            <img src={img} className={styles.img} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default FeatureCard;
