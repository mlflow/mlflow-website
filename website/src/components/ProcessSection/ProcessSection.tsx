import { motion, AnimatePresence } from "motion/react";
import { useState, useCallback } from "react";
import { Section } from "../Section/Section";

// Copy button component for code snippets
const CopyButton = ({ code }: { code: string }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }, [code]);

  return (
    <button
      onClick={handleCopy}
      className="p-1.5 rounded bg-white/10 hover:bg-white/20 transition-colors"
      aria-label={copied ? "Copied!" : "Copy code"}
    >
      <AnimatePresence mode="wait" initial={false}>
        {copied ? (
          <motion.svg
            key="check"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            exit={{ scale: 0 }}
            className="w-3.5 h-3.5 text-green-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </motion.svg>
        ) : (
          <motion.svg
            key="copy"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            exit={{ scale: 0 }}
            className="w-3.5 h-3.5 text-white/50 hover:text-white/80"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </motion.svg>
        )}
      </AnimatePresence>
    </button>
  );
};

const steps = [
  {
    number: "1",
    title: "Install MLflow",
    description: "One command to get started. Docker setup is also available.",
    time: "~1 minute",
    code: "pip install mlflow",
    language: "bash",
  },
  {
    number: "2",
    title: "Enable Auto-Logging",
    description: "Add 1 line of code to start capturing traces, metrics, and parameters",
    time: "~30 seconds",
    code: "mlflow.openai.autolog()",
    language: "python",
  },
  {
    number: "3",
    title: "View UI",
    description: "Start the MLflow server and explore the web UI from your browser.",
    time: "~1 minute",
    code: "mlflow server --port 5000",
    language: "bash",
  },
];

export function ProcessSection() {
  return (
    <Section
      title="Get Started in 3 Simple Steps"
      body="From zero to production-ready AI in minutes. No complex setup or major code changes required."
      align="center"
    >
      <div className="max-w-5xl mx-auto">
        {/* Steps container */}
        <div className="relative">
          {/* Steps grid */}
          <motion.div
            className="grid grid-cols-1 md:grid-cols-3 gap-8"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            {steps.map((step, index) => (
              <motion.div
                key={step.number}
                className="relative flex flex-col items-center text-center"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.2 }}
              >
                {/* Step number */}
                <motion.div
                  className="relative z-10 w-16 h-16 rounded-full bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-2xl font-bold text-white shadow-lg shadow-blue-500/30 mb-6"
                  whileHover={{ scale: 1.1 }}
                  transition={{ type: "spring", stiffness: 400 }}
                >
                  {step.number}
                  {/* Pulse animation */}
                  <motion.div
                    className="absolute inset-0 rounded-full bg-blue-500/50"
                    animate={{
                      scale: [1, 1.3, 1],
                      opacity: [0.5, 0, 0.5],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      delay: index * 0.3,
                    }}
                  />
                </motion.div>

                {/* Content card */}
                <div className="flex-1 w-full p-6 rounded-xl border border-white/10 bg-white/5">
                  {/* Title */}
                  <h3 className="text-xl font-semibold text-white !mb-6">
                    {step.title}
                  </h3>

                  {/* Description */}
                  <p className="text-sm text-white/60 mb-4">
                    {step.description}
                  </p>

                  {/* Code snippet */}
                  <div className="rounded-lg bg-[#0d1117] border border-white/10 overflow-hidden mb-3">
                    <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
                      <span className="text-xs text-white/50 font-mono">{step.language}</span>
                      <CopyButton code={step.code} />
                    </div>
                    <div className="p-3">
                      <code className="text-xs text-green-400 font-mono whitespace-pre !bg-transparent">
                        {step.code}
                      </code>
                    </div>
                  </div>

                  {/* Time estimate */}
                  <div className="flex items-center justify-center gap-2 text-xs text-white/40">
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                    <span>{step.time}</span>
                  </div>
                </div>

                {/* Arrow to next step (hidden on last item and mobile) */}
                {index < steps.length - 1 && (
                  <div className="hidden md:block absolute top-6 -right-4 z-20">
                    <motion.svg
                      className="w-8 h-8 text-white/30"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      initial={{ x: 0 }}
                      animate={{ x: [0, 5, 0] }}
                      transition={{ duration: 1.5, repeat: Infinity }}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M13 7l5 5m0 0l-5 5m5-5H6"
                      />
                    </motion.svg>
                  </div>
                )}
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
    </Section>
  );
}
