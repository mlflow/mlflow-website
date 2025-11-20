import LinuxFoundationLogo from "@site/static/img/linux-foundation.svg";
import { motion } from "motion/react";
import { Section } from "../Section/Section";

type Stat = {
  value: string;
  label: string;
  type?: "github";
  repo?: string;
};

export const StatsBand = () => {
  return (
    <Section
      title="The Most Used Open-Source MLOps Platform"
      body="Backed by Linux Foundation, MLflow is fully committed to open-source for 6 years."
    >
      <div className="flex w-full flex-col items-center gap-16 relative">
        {/* Animated Background - Representing Openness */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          {/* Expanding circles - representing openness and transparency */}
          {[...Array(3)].map((_, i) => (
            <motion.div
              key={`circle-${i}`}
              className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border border-blue-500/20"
              style={{
                width: "200px",
                height: "200px",
              }}
              animate={{
                width: ["200px", "800px", "200px"],
                height: ["200px", "800px", "200px"],
                opacity: [0.3, 0, 0.3],
              }}
              transition={{
                duration: 6,
                repeat: Infinity,
                delay: i * 2,
                ease: "easeInOut",
              }}
            />
          ))}
          
          {/* Moving light orbs */}
          {[...Array(5)].map((_, i) => (
            <motion.div
              key={`orb-${i}`}
              className="absolute w-32 h-32 rounded-full blur-2xl"
              style={{
                background: `radial-gradient(circle, ${
                  i % 2 === 0 ? "rgba(59, 130, 246, 0.15)" : "rgba(147, 197, 253, 0.12)"
                }, transparent 70%)`,
              }}
              animate={{
                x: [
                  `${Math.random() * 100}%`,
                  `${Math.random() * 100}%`,
                  `${Math.random() * 100}%`,
                ],
                y: [
                  `${Math.random() * 100}%`,
                  `${Math.random() * 100}%`,
                  `${Math.random() * 100}%`,
                ],
                scale: [1, 1.5, 1],
                opacity: [0.3, 0.6, 0.3],
              }}
              transition={{
                duration: 15 + i * 2,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            />
          ))}
          
          {/* Radiating particles - symbolizing open-source spreading */}
          {[...Array(8)].map((_, i) => {
            const angle = (i / 8) * Math.PI * 2;
            return (
              <motion.div
                key={`particle-${i}`}
                className="absolute left-1/2 top-1/2 w-2 h-2 bg-blue-400/40 rounded-full"
                animate={{
                  x: [0, Math.cos(angle) * 300, 0],
                  y: [0, Math.sin(angle) * 300, 0],
                  opacity: [0, 0.8, 0],
                  scale: [0, 1, 0],
                }}
                transition={{
                  duration: 4,
                  repeat: Infinity,
                  delay: i * 0.5,
                  ease: "easeOut",
                }}
              />
            );
          })}
        </div>
        <motion.a
          href="https://github.com/mlflow/mlflow"
          className="github-stats-card relative flex items-center gap-4 rounded-2xl bg-[linear-gradient(135deg,#1f2f63,#1b2342)] px-6 py-4 text-white shadow-xl overflow-hidden z-10"
          target="_blank"
          rel="noreferrer noopener"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          whileHover={{ 
            scale: 1.02,
            y: -4,
            transition: { duration: 0.2 }
          }}
        >
          <div className="github-stats-icon">
            <div className="github-stats-icon-inner">
              <div className="github-stats-face github-stats-face-front">
                <span className="inline-flex h-11 w-11 items-center justify-center rounded-full bg-white/10">
                  <svg
                    width="24"
                    height="24"
                    viewBox="0 0 48 48"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                    className="text-white"
                  >
                    <path
                      d="M24 3.9C35.05 3.9 44 12.85 44 23.9c0 4.19-1.315 8.275-3.759 11.68-2.444 3.404-5.894 5.955-9.864 7.296-.996.2-1.371-.425-1.371-.95 0-.675.025-2.825.025-5.5 0-1.875-.625-3.075-1.35-3.7 4.45-.5 9.125-2.2 9.125-9.875 0-2.2-.775-3.975-2.05-5.375.2-.5.9-2.55-.2-5.3 0 0-1.675-.55-5.5 2.05-1.6-.45-3.3-.675-5-.675s-3.7.225-5.3.675c-3.825-2.575-5.5-2.05-5.5-2.05-1.1 2.75-.4 4.8-.2 5.3-1.275 1.4-2.05 3.2-2.05 5.375 0 7.65 4.65 9.4 9.1 9.9-.575.5-1.1 1.375-1.275 2.675-1.15.525-4.025 1.375-5.825-1.65-.375-.6-1.5-2.075-3.075-2.05-1.675.025-.675.95.025 1.325.85.475 1.825 2.25 2.05 2.825.4 1.125 1.7 3.275 6.725 2.35 0 1.675.025 3.25.025 3.725 0 .525-.375 1.125-1.375.95-3.983-1.326-7.447-3.872-9.902-7.278C5.318 32.192 3.998 28.1 4 23.901 4 12.851 12.95 3.9 24 3.9Z"
                      fill="currentColor"
                    />
                  </svg>
                </span>
              </div>
              <div className="github-stats-face github-stats-face-back" aria-hidden>
                <span className="github-stats-back-icon">
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
                      d="M12 5H8a3 3 0 0 0-3 3v8a3 3 0 0 0 3 3h8a3 3 3 0 0 0 3-3v-4"
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

          <div className="flex items-center gap-4">
            <span className="text-base font-semibold">mlflow/mlflow</span>
            <div className="flex items-center gap-2 text-base font-semibold">
              23K
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="currentColor"
                xmlns="http://www.w3.org/2000/svg"
                className="text-white"
              >
                <path d="M12 2l2.89 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l7.11-1.01L12 2z" />
              </svg>
            </div>
          </div>
        </motion.a>
        <motion.div 
          className="grid w-full max-w-4xl grid-cols-1 gap-6 text-center sm:grid-cols-3 relative z-10"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ duration: 0.6, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
          >
            <LinuxFoundationLogo className="h-20 w-auto text-white" />
          </motion.div>
          
          <motion.div 
            className="flex flex-col items-center gap-1.5"
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5, delay: 0.4, ease: [0.16, 1, 0.3, 1] }}
          >
            <motion.span 
              className="text-3xl font-semibold leading-tight text-white sm:text-4xl"
              whileInView={{
                backgroundImage: [
                  "linear-gradient(135deg, #ffffff, #e0e7ff)",
                  "linear-gradient(135deg, #e0e7ff, #ffffff)",
                  "linear-gradient(135deg, #ffffff, #e0e7ff)",
                ],
              }}
              transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
              style={{
                backgroundClip: "text",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}
            >
              25 Million+
            </motion.span>
            <span className="text-xs text-white/70 sm:text-sm">Package Downloads / Month</span>
          </motion.div>
          
          <motion.div 
            className="flex items-center gap-6 rounded-xl px-6 py-8 h-20 border border-gray-200 text-[22px] font-black font-weight-black uppercase text-white leading-tight"
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5, delay: 0.5, ease: [0.16, 1, 0.3, 1] }}
          >
            <span className="flex h-8 w-8 items-center justify-center text-white/80">
              <svg
                width="26"
                height="24"
                viewBox="0 0 16 16"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden="true"
              >
                <path d="M8.75.75V2h.985c.304 0 .603.08.867.231l1.29.736c.038.022.08.033.124.033h2.234a.75.75 0 0 1 0 1.5h-.427l2.111 4.692a.75.75 0 0 1-.154.838l-.53-.53.529.531-.001.002-.002.002-.006.006-.006.005-.01.01-.045.04c-.21.176-.441.327-.686.45C14.556 10.78 13.88 11 13 11a4.498 4.498 0 0 1-2.023-.454 3.544 3.544 0 0 1-.686-.45l-.045-.04-.016-.015-.006-.006-.004-.004v-.001a.75.75 0 0 1-.154-.838L12.178 4.5h-.162c-.305 0-.604-.079-.868-.231l-1.29-.736a.245.245 0 0 0-.124-.033H8.75V13h2.5a.75.75 0 0 1 0 1.5h-6.5a.75.75 0 0 1 0-1.5h2.5V3.5h-.984a.245.245 0 0 0-.124.033l-1.289.737c-.265.15-.564.23-.869.23h-.162l2.112 4.692a.75.75 0 0 1-.154.838l-.53-.53.529.531-.001.002-.002.002-.006.006-.016.015-.045.04c-.21.176-.441.327-.686.45C4.556 10.78 3.88 11 3 11a4.498 4.498 0 0 1-2.023-.454 3.544 3.544 0 0 1-.686-.45l-.045-.04-.016-.015-.006-.006-.004-.004v-.001a.75.75 0 0 1-.154-.838L2.178 4.5H1.75a.75.75 0 0 1 0-1.5h2.234a.249.249 0 0 0 .125-.033l1.288-.737c.265-.15.564-.23.869-.23h.984V.75a.75.75 0 0 1 1.5 0Zm2.945 8.477c.285.135.718.273 1.305.273s1.02-.138 1.305-.273L13 6.327Zm-10 0c.285.135.718.273 1.305.273s1.02-.138 1.305-.273L3 6.327Z" />
              </svg>
            </span>
            <div className="flex flex-col items-start gap-1">
              <span>Apache-2.0</span>
              <span>License</span>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </Section>
  );
};
