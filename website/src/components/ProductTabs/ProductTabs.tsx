import clsx from "clsx";
import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { categories } from "./features";
import { StickyFeaturesGrid } from "./StickyFeaturesGrid";

const UnderlineTabs = ({
  activeCategory,
  setActiveCategory,
}: {
  activeCategory: string;
  setActiveCategory: (id: string) => void;
}) => (
  <div className="flex justify-center">
    <div className="flex gap-8">
      {categories.map((category) => {
        const isActive = category.id === activeCategory;
        return (
          <button
            key={category.id}
            onClick={() => setActiveCategory(category.id)}
            className={clsx(
              "relative px-2 py-3 text-lg font-medium transition-colors",
              isActive ? "text-white" : "text-white/50 hover:text-white/70",
            )}
          >
            {category.label}
            {isActive && (
              <motion.div
                layoutId="activeUnderline"
                className="absolute bottom-0 left-0 right-0 h-[2px]"
                style={{
                  background:
                    "linear-gradient(90deg, #e05585, #9066cc, #5a8fd4)",
                }}
                transition={{ type: "spring", stiffness: 400, damping: 30 }}
              />
            )}
          </button>
        );
      })}
    </div>
  </div>
);

export function ProductTabs() {
  const [activeCategory, setActiveCategory] = useState(categories[0].id);
  const activeFeatures =
    categories.find((c) => c.id === activeCategory)?.features ?? [];

  return (
    <motion.div
      className="w-full flex flex-col gap-12"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6 }}
    >
      {/* Top-level category tabs */}
      <UnderlineTabs
        activeCategory={activeCategory}
        setActiveCategory={setActiveCategory}
      />

      {/* Features - sticky scroll layout */}
      <div className="px-4">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeCategory}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
          >
            <StickyFeaturesGrid features={activeFeatures} />
          </motion.div>
        </AnimatePresence>
      </div>
    </motion.div>
  );
}
