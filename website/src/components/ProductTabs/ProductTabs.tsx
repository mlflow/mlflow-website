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
    <div className={bubbleClass}>
      <div className={styles.content}>
        <span className={styles.title}>
          <span className={styles.hintIcon} aria-hidden>
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
          </span>
          {label}
        </span>
        {description && <span className={styles.description}>{description}</span>}
      </div>
      <span className={arrowClass} aria-hidden />
    </div>
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
      <div className="relative w-full" ref={containerRef}>
        <div className="flex flex-wrap justify-center gap-6 md:gap-8 pb-4">
          {tabs.map((tab) => {
            const isActive = tab.id === activeTab.id;
            return (
              <button
                key={tab.id}
                type="button"
                ref={(el) => {
                  tabRefs.current[tab.id] = el;
                }}
                onClick={() => setActiveTabId(tab.id)}
                className="group flex items-center gap-2 text-md font-medium focus:outline-none"
              >
                {tab.icon && (
                  <span className="text-white/60 text-base leading-none">
                    {tab.icon}
                  </span>
                )}
                <span
                  className={clsx(
                    "transition-colors",
                    isActive ? "text-white" : "text-white/70 group-hover:text-white",
                  )}
                >
                  {tab.label}
                </span>
              </button>
            );
          })}
        </div>

        <div className="absolute left-0 right-0 bottom-0 border-b border-white/10" aria-hidden />
        <div
          aria-hidden
          className="absolute bottom-0 h-0.5 bg-[#3b82f6] transition-all duration-200"
          style={{
            width: indicator?.width ?? 0,
            transform: `translateX(${indicator?.left ?? 0}px)`,
          }}
        />
      </div>

      <div className="relative mx-16">
        {/* halo around the frame (sits outside the clipped card) */}
        <div
          className="pointer-events-none absolute -inset-8 rounded-[30px] z-0 bg-[radial-gradient(circle_at_50%_40%,rgba(59,130,246,0.16),transparent_65%),radial-gradient(circle_at_80%_10%,rgba(255,255,255,0.10),transparent_55%)] blur-[45px] shadow-[0_0_70px_rgba(59,130,246,0.08)]"
          aria-hidden
        />

        <div className="relative border border-[rgba(255,255,255,0.08)] bg-[rgba(255,255,255,0.03)] rounded-2xl overflow-visible shadow-[0_20px_80px_rgba(0,0,0,0.35)]">
          {/* outer glow behind the card */}
          <div
            className="pointer-events-none absolute -inset-4 blur-[40px] z-0 bg-[radial-gradient(circle_at_30%_20%,rgba(59,130,246,0.18),transparent_55%),radial-gradient(circle_at_80%_0%,rgba(253, 137, 137, 0.12),transparent_50%)]"
            aria-hidden
          />

          {/* glare and sheen over the image */}
          <div
            className="pointer-events-none absolute inset-0 z-20 mix-blend-screen opacity-32 bg-[linear-gradient(130deg,rgba(96,165,250,0.16)_0%,rgba(255,255,255,0.04)_45%,transparent_60%),radial-gradient(120%_60%_at_20%_10%,rgba(59,130,246,0.16),transparent_55%)]"
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
                <div className="absolute inset-0 rounded-md border border-white/30 bg-white/6 opacity-0 group-hover:opacity-100 transition duration-200" />
                <div className="relative h-full w-full pointer-events-none opacity-0 group-hover:opacity-100 transition duration-200">
                  <Bubble
                    label={spot.label}
                    description={spot.description}
                    direction={spot.direction}
                  />
                </div>
              </SpotWithLink>
            ))}
          </div>
        </div>
      </div>
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
