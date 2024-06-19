import React from "react";
import styles from "./styles.module.css";

const AutoScroll = ({ images }: { images: string[] }) => {
  return (
    <div className={styles.container}>
      {Array.from({ length: 5 }).map((_, id) => (
        <div key={id} className={styles.section}>
          {images.map((image, id) => (
            <img key={id} src={image} alt={image} className={styles.img} />
          ))}
        </div>
      ))}
    </div>
  );
};

export default AutoScroll;
