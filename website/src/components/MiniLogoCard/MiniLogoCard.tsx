import clsx from "clsx";
import styles from "./styles.module.css";
import { ExternalLink } from "lucide-react";

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
      <a
        href={href}
        className={styles.a}
        target="_blank"
        rel="noreferrer noopener"
      >
        <div className={clsx(styles.inner, "card__body")}>
          <div className={styles.container}>
            <div className={styles.logoFlip}>
              <div className={styles.logoInner}>
                <div className={styles.logoFront}>
                  <img
                    src={src}
                    alt={title}
                    className={clsx(
                      styles.img,
                      title === "LangChain / LangGraph" && styles.imgLangChain,
                    )}
                  />
                </div>
                <div className={styles.logoBack} aria-hidden="true">
                  <span className={styles.backIcon}>
                    <ExternalLink className="w-[22px] h-[22px]" />
                  </span>
                </div>
              </div>
            </div>
            <h4 className={styles.title}>{title}</h4>
          </div>
        </div>
      </a>
    </div>
  );
};

export default MiniLogoCard;
