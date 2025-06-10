import React, { ComponentProps, useRef } from "react";
import {
  AnimatePresence,
  motion,
  useMotionValueEvent,
  useScroll,
} from "motion/react";
import { Card } from "..";

type Props = {
  cards: ComponentProps<typeof Card>[];
};

export const StickyGrid = ({ cards }: Props) => {
  const [activeCardIndex, setActiveCard] = React.useState(0);
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start 80px", "end 80px"],
  });

  useMotionValueEvent(scrollYProgress, "change", (latest) => {
    const cardsBreakpoints = cards.map((_, index) => index / cards.length);
    const closestBreakpointIndex = cardsBreakpoints.reduce(
      (acc, breakpoint, index) => {
        const distance = Math.abs(latest - breakpoint);
        if (distance < Math.abs(latest - cardsBreakpoints[acc])) {
          return index;
        }
        return acc;
      },
      0,
    );
    setActiveCard(closestBreakpointIndex);
  });

  return (
    <div className="w-full flex flex-row gap-20" ref={ref}>
      <div className="div relative flex flex-col items-start md:w-1/2">
        {cards.map(({ image, ...card }, index) => (
          <div
            key={index}
            className="border-[rgba(255,255,255,0.08)] border-t border-b aspect-[3/2] w-full md:sticky top-24 bg-brand-black flex flex-col justify-center gap-y-8 py-10"
          >
            <Card
              {...card}
              image={image && <div className="md:hidden">{image}</div>}
            />
          </div>
        ))}
      </div>
      <div className="sticky top-24 right-0 hidden aspect-[3/2] h-full w-1/2 overflow-hidden rounded-md md:block">
        <AnimatePresence initial={false}>
          <motion.div
            key={activeCardIndex}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            {cards[activeCardIndex].image}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
};
