import clsx from "clsx";
import {
  ReactNode,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useEffect,
  useCallback,
} from "react";
import { motion } from "motion/react";
import styles from "./ProductTabs.module.css";
import EvaluationTabImg from "@site/static/img/GenAI_home/GenAI_evaluation_darkmode.png";
import MonitoringTabImg from "@site/static/img/GenAI_home/GenAI_monitor_darkmode.png";
import AnnotationTabImg from "@site/static/img/GenAI_home/GenAI_annotation_darkmode.png";
import PromptTabImg from "@site/static/img/GenAI_home/GenAI_prompts_darkmode.png";
import OptimizeTabImg from "@site/static/img/GenAI_home/GenAI_optimize_darkmode.png";

const defaultTabImage = "/img/GenAI_home/GenAI_trace_darkmode.png";

const MonitoringIcon = () => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 20 20"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className="text-white/70"
  >
    <path
      d="M10 3.5a5.5 5.5 0 0 0-5.5 5.5c0 2.2 1.24 4.12 3.05 5.02V15a2.45 2.45 0 0 0 2.45 2.45h0A2.45 2.45 0 0 0 12.45 15v-.98A5.48 5.48 0 0 0 15.5 9c0-3.03-2.47-5.5-5.5-5.5Z"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path d="M8 15h4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    <circle cx="10" cy="9" r="1" fill="currentColor" />
  </svg>
);

export const defaultProductTabs: Tab[] = [
  {
    id: "tracing",
    label: "Tracing",
    icon: "⎋",
    imageSrc: defaultTabImage,
    link: "https://mlflow.org/docs/latest/genai/tracing/",
    hotspots: [
      {
        id: "trace-breakdown",
        left: "0%",
        top: "22%",
        width: "25%",
        height: "78%",
        label: "Trace breakdown",
        description:
          "MLflow visualized the execution flow of your GenAI applications, including LLM calls, tool invocations, retrieval steps, and more.",
        direction: "right",
        link: "https://mlflow.org/docs/latest/genai/tracing/",
      },
      {
        id: "span-details",
        left: "25%",
        top: "22%",
        width: "52.5%",
        height: "78%",
        label: "Span details",
        description:
          "Each span represents a single step in the execution flow. They capture the inputs, outputs, token usage, latency, and many more metadata about the step.",
        direction: "top",
        link: "https://mlflow.org/docs/latest/genai/tracing/",
      },
      {
        id: "trace-assessment",
        left: "77.5%",
        top: "22%",
        width: "22.5%",
        height: "78%",
        label: "Feedback collection",
        description:
          "MLflow provides an UI and APIs for you to collect feedback from your users or domain experts on the quality of the application's output.",
        direction: "left",
        link: "https://mlflow.org/docs/latest/genai/tracing/collect-user-feedback/",
      },
      {
        id: "trace-info",
        left: "0%",
        top: "0%",
        width: "100%",
        height: "22%",
        label: "Trace info",
        description:
          "The trace header panel provides a summary of the trace, including the the latency, token usage, session ID, and more.",
        direction: "bottom",
        link: "https://mlflow.org/docs/latest/genai/tracing/",
      },
    ],
  },
  {
    id: "evaluation",
    label: "Evaluation",
    icon: "☑",
    imageSrc: EvaluationTabImg,
  },
  {
    id: "monitoring",
    label: "Monitoring",
    icon: <MonitoringIcon />,
    imageSrc: MonitoringTabImg,
  },
  {
    id: "annotation",
    label: "Annotation",
    icon: "☰",
    imageSrc: AnnotationTabImg,
  },
  { id: "prompt", label: "Prompt", icon: "⌘", imageSrc: PromptTabImg },
  {
    id: "optimize",
    label: "Optimize",
    icon: "⚙",
    imageSrc: OptimizeTabImg,
  },
  { id: "gateway", label: "Gateway", icon: "⇄", imageSrc: defaultTabImage },
  { id: "versioning", label: "Versioning", icon: "⟳", imageSrc: defaultTabImage },
];

type Tab = {
  id: string;
  label: string;
  imageSrc: string;
  icon?: ReactNode;
  hotspots?: Hotspot[];
  link?: string;
};

type Hotspot = {
  id: string;
  left: string; // percentage string e.g. "30%"
  top: string; // percentage string
  width: string; // percentage string
  height: string; // percentage string
  label: string;
  description?: string;
  direction?: "top" | "right" | "bottom" | "left";
  link?: string;
};

type Props = {
  tabs: Tab[];
};

const Bubble = ({
  label,
  description,
  direction = "top",
}: {
  label: string;
  description?: string;
  direction?: Hotspot["direction"];
}) => {
  const bubbleClass = clsx(styles.bubble, {
    [styles.top]: direction === "top",
    [styles.right]: direction === "right",
    [styles.bottom]: direction === "bottom",
    [styles.left]: direction === "left",
  });

  const arrowClass = clsx(styles.arrow, {
    [styles.topArrow]: direction === "top",
    [styles.rightArrow]: direction === "right",
    [styles.bottomArrow]: direction === "bottom",
    [styles.leftArrow]: direction === "left",
  });

  return (
    <motion.div 
      className={bubbleClass}
      initial={{ opacity: 0, scale: 0.9, y: direction === "top" ? 10 : direction === "bottom" ? -10 : 0, x: direction === "left" ? 10 : direction === "right" ? -10 : 0 }}
      animate={{ opacity: 1, scale: 1, y: 0, x: 0 }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
    >
      {/* Animated glow effect */}
      <motion.div
        className="absolute inset-0 rounded-lg bg-gradient-to-br from-blue-500/20 to-cyan-500/20 blur-md -z-10"
        animate={{
          scale: [1, 1.05, 1],
          opacity: [0.5, 0.8, 0.5],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
      
      <div className={styles.content}>
        <span className={styles.title}>
          <motion.span 
            className={styles.hintIcon} 
            aria-hidden
            animate={{
              rotate: [0, -5, 5, -5, 0],
            }}
            transition={{
              duration: 0.6,
              ease: "easeInOut",
            }}
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M12 3.5c-2.9 0-5.25 2.27-5.25 5.07 0 1.82.92 3.43 2.32 4.36.37.25.6.66.6 1.1v.48c0 .41.34.75.75.75h3.16c.41 0 .75-.34.75-.75v-.48c0-.44.22-.85.6-1.1 1.4-.93 2.32-2.54 2.32-4.36 0-2.8-2.35-5.07-5.25-5.07Z"
                stroke="white"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M10 19h4"
                stroke="white"
                strokeWidth="1.6"
                strokeLinecap="round"
              />
              <path
                d="M11 21h2"
                stroke="white"
                strokeWidth="1.6"
                strokeLinecap="round"
              />
            </svg>
          </motion.span>
          {label}
        </span>
        {description && (
          <motion.span 
            className={styles.description}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            {description}
          </motion.span>
        )}
      </div>
      <span className={arrowClass} aria-hidden />
    </motion.div>
  );
};

export function ProductTabs({ tabs }: Props) {
  const [activeTabId, setActiveTabId] = useState(tabs[0]?.id);
  const [indicator, setIndicator] = useState<{ width: number; left: number }>();

  const containerRef = useRef<HTMLDivElement>(null);
  const tabRefs = useRef<Record<string, HTMLButtonElement | null>>({});

  const activeTab = useMemo(
    () => tabs.find((tab) => tab.id === activeTabId) ?? tabs[0],
    [activeTabId, tabs],
  );

  const updateIndicator = useCallback(() => {
    if (!activeTab) return;
    const btn = tabRefs.current[activeTab.id];
    const container = containerRef.current;
    if (btn && container) {
      const btnRect = btn.getBoundingClientRect();
      const containerRect = container.getBoundingClientRect();
      setIndicator({
        width: btnRect.width,
        left: btnRect.left - containerRect.left,
      });
    }
  }, [activeTab]);

  useLayoutEffect(() => {
    updateIndicator();
  }, [activeTab, updateIndicator]);

  // Recompute on resize to keep indicator aligned
  useEffect(() => {
    const handler = () => updateIndicator();
    window.addEventListener("resize", handler);
    return () => window.removeEventListener("resize", handler);
  }, [updateIndicator]);

  if (!activeTab) {
    return null;
  }

  return (
    <div className="w-full flex flex-col gap-8">
      {/* Floating particles background */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        {[...Array(15)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-blue-400/20 rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              y: [0, -20, 0],
              opacity: [0, 0.6, 0],
              scale: [1, 1.5, 1],
            }}
            transition={{
              duration: 2 + Math.random() * 2,
              repeat: Infinity,
              delay: Math.random() * 2,
              ease: "easeInOut",
            }}
          />
        ))}
      </div>

      <div className="relative w-full" ref={containerRef}>
        <motion.div 
          className="flex flex-wrap justify-center gap-6 md:gap-8 pb-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        >
          {tabs.map((tab, index) => {
            const isActive = tab.id === activeTab.id;
            return (
              <motion.button
                key={tab.id}
                type="button"
                ref={(el) => {
                  tabRefs.current[tab.id] = el;
                }}
                onClick={() => setActiveTabId(tab.id)}
                className="group relative flex items-center gap-2 text-md font-medium focus:outline-none px-3 py-2 rounded-lg transition-all"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.05, ease: [0.16, 1, 0.3, 1] }}
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.98 }}
              >
                {/* Glow effect on hover */}
                <motion.div
                  className="absolute inset-0 rounded-lg bg-gradient-to-r from-blue-500/10 via-cyan-500/10 to-blue-500/10 opacity-0 group-hover:opacity-100 blur-md"
                  initial={false}
                  animate={{
                    opacity: isActive ? 0.3 : 0,
                  }}
                  transition={{ duration: 0.3 }}
                />
                
                {/* Active background */}
                {isActive && (
                  <motion.div
                    layoutId="activeTabBackground"
                    className="absolute inset-0 bg-white/5 backdrop-blur-sm rounded-lg border border-white/10"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                  />
                )}

                {/* Icon with animation */}
                {tab.icon && (
                  <motion.span 
                    className={clsx(
                      "text-base leading-none relative z-10 transition-all",
                      isActive ? "text-blue-400" : "text-white/60 group-hover:text-blue-300"
                    )}
                    animate={{
                      rotate: isActive ? [0, -10, 10, -10, 0] : 0,
                    }}
                    transition={{
                      duration: 0.5,
                      ease: "easeInOut",
                    }}
                  >
                    {tab.icon}
                  </motion.span>
                )}
                
                {/* Label */}
                <span
                  className={clsx(
                    "transition-all relative z-10 font-semibold",
                    isActive 
                      ? "text-white" 
                      : "text-white/70 group-hover:text-white/90",
                  )}
                >
                  {tab.label}
                </span>

                {/* Sparkle effect on active */}
                {isActive && (
                  <motion.div
                    className="absolute -right-1 -top-1 w-1.5 h-1.5 rounded-full bg-blue-400"
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ 
                      scale: [0, 1.2, 1],
                      opacity: [0, 1, 0.8],
                    }}
                    transition={{ duration: 0.4 }}
                  />
                )}
              </motion.button>
            );
          })}
        </motion.div>

        {/* Decorative line */}
        <div className="absolute left-0 right-0 bottom-0 border-b border-white/10" aria-hidden />
        
        {/* Animated indicator with glow */}
        <motion.div
          aria-hidden
          className="absolute bottom-0 h-0.5 rounded-full"
          style={{
            width: indicator?.width ?? 0,
            transform: `translateX(${indicator?.left ?? 0}px)`,
          }}
        >
          {/* Main bar */}
          <div className="absolute inset-0 bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-400 rounded-full" />
          
          {/* Glow effect */}
          <div className="absolute inset-0 bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-400 rounded-full blur-sm opacity-70" />
          
          {/* Moving shimmer */}
          <motion.div
            className="absolute inset-0 rounded-full overflow-hidden"
          >
            <motion.div
              className="absolute inset-y-0 w-[200%] bg-gradient-to-r from-transparent from-40% via-white via-50% to-transparent to-60%"
              animate={{
                x: ["-100%", "0%"],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: "linear",
              }}
            />
          </motion.div>
        </motion.div>
      </div>

      <motion.div 
        className="relative mx-16"
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
      >
        {/* Enhanced halo with animation */}
        <motion.div
          className="pointer-events-none absolute -inset-8 rounded-[30px] z-0"
          style={{
            background: "radial-gradient(circle at 50% 40%, rgba(59,130,246,0.16), transparent 65%), radial-gradient(circle at 80% 10%, rgba(255,255,255,0.10), transparent 55%)",
            filter: "blur(45px)",
            boxShadow: "0 0 70px rgba(59,130,246,0.08)",
          }}
          animate={{
            scale: [1, 1.05, 1],
            opacity: [0.8, 1, 0.8],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut",
          }}
          aria-hidden
        />

        <motion.div 
          className="relative border border-[rgba(255,255,255,0.08)] bg-[rgba(255,255,255,0.03)] rounded-2xl overflow-visible shadow-[0_20px_80px_rgba(0,0,0,0.35)] group/card"
          whileHover={{ scale: 1.01 }}
          transition={{ duration: 0.3, ease: "easeOut" }}
        >
          {/* Enhanced outer glow with animation */}
          <motion.div
            className="pointer-events-none absolute -inset-4 z-0"
            style={{
              background: "radial-gradient(circle at 30% 20%, rgba(59,130,246,0.18), transparent 55%), radial-gradient(circle at 80% 0%, rgba(253, 137, 137, 0.12), transparent 50%)",
              filter: "blur(40px)",
            }}
            animate={{
              opacity: [0.5, 0.8, 0.5],
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: "easeInOut",
            }}
            aria-hidden
          />

          {/* Enhanced glare - static for better performance */}
          <div
            className="pointer-events-none absolute inset-0 z-20 mix-blend-screen rounded-2xl"
            style={{
              background: "linear-gradient(130deg, rgba(96,165,250,0.16) 0%, rgba(255,255,255,0.04) 45%, transparent 60%), radial-gradient(120% 60% at 20% 10%, rgba(59,130,246,0.16), transparent 55%)",
              opacity: 0.32,
            }}
            aria-hidden
          />

          <div className="relative z-10">
            <img
              src={activeTab.imageSrc}
              alt={`${activeTab.label} screenshot`}
              className="w-full h-full object-cover shadow-[0_18px_50px_rgba(0,0,0,0.35)] rounded-2xl"
              loading="lazy"
            />

            {activeTab.hotspots?.map((spot) => (
              <SpotWithLink key={spot.id} spot={spot}>
                <motion.div 
                  className="absolute inset-0 rounded-md border border-white/30 bg-white/6 opacity-0 group-hover:opacity-100"
                  initial={false}
                  whileHover={{
                    backgroundColor: "rgba(255,255,255,0.12)",
                    borderColor: "rgba(255,255,255,0.5)",
                  }}
                  transition={{ duration: 0.2 }}
                />
                <div className="relative h-full w-full pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                  <Bubble
                    label={spot.label}
                    description={spot.description}
                    direction={spot.direction}
                  />
                </div>
              </SpotWithLink>
            ))}
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}

const SpotWithLink = ({ spot, children }: { spot: Hotspot; children: React.ReactNode }) => {
  const Wrapper = (props: any) => (spot.link ? <a {...props} /> : <div {...props} />);
  return (
    <Wrapper
      className="group absolute"
      href={spot.link}
      target={spot.link ? "_blank" : undefined}
      rel={spot.link ? "noreferrer noopener" : undefined}
      style={{
        left: spot.left,
        top: spot.top,
        width: spot.width,
        height: spot.height,
      }}
    >
      {children}
    </Wrapper>
  );
};
