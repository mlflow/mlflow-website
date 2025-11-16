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

type Tab = {
  id: string;
  label: string;
  imageSrc: string;
  icon?: ReactNode;
  hotspots?: Hotspot[];
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
};

type Props = {
  tabs: Tab[];
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
                className="group flex items-center gap-2 text-sm font-medium focus:outline-none"
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

        <div className="relative border border-[rgba(255,255,255,0.08)] bg-[rgba(255,255,255,0.03)] rounded-2xl overflow-hidden shadow-[0_20px_80px_rgba(0,0,0,0.35)]">
          {/* outer glow behind the card */}
          <div
            className="pointer-events-none absolute -inset-4 blur-[40px] z-0 bg-[radial-gradient(circle_at_30%_20%,rgba(59,130,246,0.18),transparent_55%),radial-gradient(circle_at_80%_0%,rgba(255,255,255,0.12),transparent_50%)]"
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
              className="w-full h-full object-cover shadow-[0_18px_50px_rgba(0,0,0,0.35)]"
              loading="lazy"
            />

            {activeTab.hotspots?.map((spot) => (
              <div
                key={spot.id}
                className="group absolute"
                style={{
                  left: spot.left,
                  top: spot.top,
                  width: spot.width,
                  height: spot.height,
                }}
              >
                <div className="absolute inset-0 rounded-md border border-white/25 bg-white/10 opacity-0 group-hover:opacity-100 transition duration-200" />
                {(() => {
                  const dir = spot.direction ?? "top";
                  const baseTooltip =
                    "absolute min-w-[200px] min-h-[68px] max-w-[260px] px-4 py-3 rounded-xl bg-[rgba(14,20,22,0.7)] border border-white/20 text-white shadow-[0_12px_32px_rgba(0,0,0,0.45)] opacity-0 group-hover:opacity-100 transition duration-200 z-10";
                  const arrowBase = "absolute h-3 w-3 rotate-45 bg-[rgba(14,20,22,0.7)] border border-white/20";

                  const content = (
                    <div className="flex flex-col gap-1 leading-snug">
                      <span className="text-sm font-semibold">{spot.label}</span>
                      {spot.description && (
                        <span className="text-xs text-white/80">{spot.description}</span>
                      )}
                    </div>
                  );

                  switch (dir) {
                    case "right":
                      return (
                        <div className={clsx(baseTooltip, "left-full top-1/2 -translate-y-1/2 ml-3 bg-[rgba(14,20,22,0.7)]")}> 
                          {content}
                          <span className={clsx(arrowBase, "left-[-6px] top-1/2 -translate-y-1/2 bg-[rgba(14,20,22,0.7)]")}></span>
                        </div>
                      );
                    case "bottom":
                      return (
                        <div className={clsx(baseTooltip, "left-1/2 top-full -translate-x-1/2 mt-3 bg-[rgba(14,20,22,0.7)]")}> 
                          {content}
                          <span className={clsx(arrowBase, "left-1/2 -translate-x-1/2 top-[-6px] bg-[rgba(14,20,22,0.7)]")}></span>
                        </div>
                      );
                    case "left":
                      return (
                        <div className={clsx(baseTooltip, "right-full top-1/2 -translate-y-1/2 mr-3 bg-[rgba(14,20,22,0.7)]")}> 
                          {content}
                          <span className={clsx(arrowBase, "right-[-6px] top-1/2 -translate-y-1/2 bg-[rgba(14,20,22,0.7)]")}></span>
                        </div>
                      );
                    case "top":
                    default:
                      return (
                        <div className={clsx(baseTooltip, "left-1/2 -top-3 -translate-x-1/2 -translate-y-full bg-[rgba(14,20,22,0.7)]")}> 
                          {content}
                          <span className={clsx(arrowBase, "left-1/2 -translate-x-1/2 -bottom-[6px] bg-[rgba(14,20,22,0.7)]")}></span>
                        </div>
                      );
                  }
                })()}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
