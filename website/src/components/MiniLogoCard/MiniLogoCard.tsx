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
      <a href={href} className={styles.a} target="_blank" rel="noreferrer noopener">
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
                    <svg
                      width="22"
                      height="22"
                      viewBox="0 0 24 24"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        d="M14 5h5m0 0v5m0-5L10 14"
                        stroke="currentColor"
                        strokeWidth="1.6"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                      <path
                        d="M12 5H8a3 3 0 0 0-3 3v8a3 3 0 0 0 3 3h8a3 3 0 0 0 3-3v-4"
                        stroke="currentColor"
                        strokeWidth="1.6"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        opacity="0.78"
                      />
                    </svg>
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
