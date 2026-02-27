import { useState, useRef } from "react";
import {
  motion,
  AnimatePresence,
  useScroll,
  useMotionValueEvent,
} from "motion/react";
import { FeatureMediaCard } from "./FeatureMediaCard";
import type { Feature } from "./features";

/** Linearly interpolate between two hex colours. */
const interpolateColor = (color1: string, color2: string, factor: number) => {
  const hex = (c: string) => parseInt(c, 16);
  const r1 = hex(color1.slice(1, 3)),
    g1 = hex(color1.slice(3, 5)),
    b1 = hex(color1.slice(5, 7));
  const r2 = hex(color2.slice(1, 3)),
    g2 = hex(color2.slice(3, 5)),
    b2 = hex(color2.slice(5, 7));
  const r = Math.round(r1 + (r2 - r1) * factor);
  const g = Math.round(g1 + (g2 - g1) * factor);
  const b = Math.round(b1 + (b2 - b1) * factor);
  return `rgb(${r}, ${g}, ${b})`;
};

/** QuickstartLink component with animated glow effect. */
const QuickstartLink = ({ href }: { href: string }) => (
  <motion.a
    href={href}
    target="_blank"
    rel="noreferrer noopener"
    className="relative inline-flex items-center gap-1 text-sm font-medium w-fit"
    whileHover="hover"
  >
    <motion.span
      className="absolute inset-0 rounded-md -z-10"
      style={{
        background:
          "linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(168, 85, 247, 0.3))",
        filter: "blur(8px)",
      }}
      animate={{
        opacity: [0.4, 0.8, 0.4],
        scale: [1, 1.05, 1],
      }}
      transition={{
        duration: 2,
        repeat: Infinity,
        ease: "easeInOut",
      }}
    />
    <span className="relative z-10 text-white px-3 py-1">Quickstart</span>
    <motion.span
      className="relative z-10 text-white pr-2"
      variants={{
        hover: { x: 4 },
      }}
    >
      →
    </motion.span>
  </motion.a>
);

/** Left‑side text panel – sticks while scrolling. */
const FeatureTextSection = ({
  feature,
  visibility = 1,
}: {
  feature: Feature;
  visibility?: number;
}) => {
  const titleColor = interpolateColor("#1a1a1a", "#ffffff", visibility);
  const descColor = interpolateColor("#1a1a1a", "#9ca3af", visibility);

  return (
    <div className="border-[rgba(255,255,255,0.08)] border-t border-b min-h-[350px] w-full lg:sticky top-24 bg-brand-black flex flex-col justify-center gap-y-8 py-10">
      <div className="flex flex-col gap-4">
        <h3 className="text-2xl font-bold" style={{ color: titleColor }}>
          {feature.title}
        </h3>
        <p className="leading-relaxed" style={{ color: descColor }}>
          {feature.description}
        </p>
        {feature.quickstartLink && (
          <div style={{ opacity: visibility }}>
            <QuickstartLink href={feature.quickstartLink} />
          </div>
        )}
      </div>

      {/* Mobile: Show image inline */}
      <div className="lg:hidden">
        <FeatureMediaCard feature={feature} />
      </div>
    </div>
  );
};

export const StickyFeaturesGrid = ({ features }: { features: Feature[] }) => {
  const [scrollProgress, setScrollProgress] = useState(0);
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start 40vh", "end 40vh"],
  });

  useMotionValueEvent(scrollYProgress, "change", (latest) => {
    setScrollProgress(latest);
  });

  const getVisibility = (index: number) => {
    const cardPosition = index / features.length;
    const distance = Math.abs(scrollProgress - cardPosition);
    const plateauRange = 0.08;
    const fadeRange = 0.25;

    if (distance <= plateauRange) return 1;
    const distanceFromPlateau = distance - plateauRange;
    return Math.max(0, 1 - distanceFromPlateau / fadeRange);
  };

  const activeFeatureIndex = Math.min(
    features.length - 1,
    Math.max(0, Math.round(scrollProgress * features.length)),
  );
  const activeFeature = features[activeFeatureIndex];

  return (
    <div className="w-full flex flex-row gap-8" ref={ref}>
      {/* Left: Stacking sticky text sections */}
      <div className="relative flex flex-col items-start lg:w-1/2">
        {features.map((feature, index) => (
          <FeatureTextSection
            key={feature.id}
            feature={feature}
            visibility={getVisibility(index)}
          />
        ))}
      </div>

      {/* Right: Sticky image panel (desktop only) */}
      <div className="sticky top-[40vh] right-0 hidden h-[350px] w-1/2 lg:block">
        <AnimatePresence mode="popLayout">
          <motion.div
            key={activeFeatureIndex}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4 }}
            className="h-full"
          >
            {activeFeature && <FeatureMediaCard feature={activeFeature} />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
};
