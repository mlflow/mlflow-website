import clsx from "clsx";
import styles from "./styles.module.css";
import { useEffect, useState } from "react";

const ImageCarousel = ({ images }) => {
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    setActiveIndex(0);
  }, [images]);

  return (
    <div className={styles.carousel}>
      <img src={images[activeIndex]} className={styles.img} />
      {images.length > 1 && (
        <div className={styles.dotsContainer}>
          {images.map((_, index) => (
            <div
              key={index}
              className={clsx(styles.dot, {
                [styles.active]: index === activeIndex,
              })}
              onClick={() => setActiveIndex(index)}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default ImageCarousel;
