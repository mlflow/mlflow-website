import { motion } from "motion/react";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Section } from "../Section/Section";
import { CopyButton } from "../CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../CodeSnippet/codeTheme";
import { Clock, ArrowRight } from "lucide-react";

const steps = [
  {
    number: "1",
    title: "Start MLflow Server",
    description: "One command to get started. Docker setup is also available.",
    time: "~30 seconds",
    code: `uvx mlflow server`,
    language: "bash",
  },
  {
    number: "2",
    title: "Enable Logging",
    description:
      "Add minimal code to start capturing traces, metrics, and parameters",
    time: "~30 seconds",
    code: `import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.openai.autolog()`,
    language: "python",
  },
  {
    number: "3",
    title: "Run your code",
    description:
      "Run your ML/LLM code as usual. MLflow logs traces and you can explore them in the MLflow UI.",
    time: "~1 minute",
    code: `from openai import OpenAI

client = OpenAI()
client.responses.create(
    model="gpt-5-mini",
    input="Hello!",
)`,
    language: "python",
  },
];

export function ProcessSection() {
  return (
    <Section
      id="get-started"
      title="Get Started in 3 Simple Steps"
      body={
        <div className="flex flex-col items-center gap-4">
          <span>
            From zero to full-stack LLMOps in minutes. No complex setup or major
            code changes required.
          </span>
          <Link
            to="https://mlflow.org/docs/latest/genai/tracing/quickstart/"
            style={{ textDecoration: "underline" }}
            className="hover:opacity-80"
          >
            Get Started →
          </Link>
        </div>
      }
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
            {steps.map((step, index) => {
              // Red → Purple → Blue gradient theme (darkened) - each step gets a gradient segment
              const stepColors = [
                {
                  bg: "linear-gradient(135deg, #8a2a4a 0%, #6a2850 50%, #4a3366 100%)",
                  shadow: "rgba(138, 42, 74, 0.4)",
                  pulse: "rgba(138, 42, 74, 0.5)",
                },
                {
                  bg: "linear-gradient(135deg, #702848 0%, #4a3366 50%, #3a4080 100%)",
                  shadow: "rgba(74, 51, 102, 0.4)",
                  pulse: "rgba(74, 51, 102, 0.5)",
                },
                {
                  bg: "linear-gradient(135deg, #4a3366 0%, #3a4580 50%, #2a4a7a 100%)",
                  shadow: "rgba(42, 74, 122, 0.4)",
                  pulse: "rgba(42, 74, 122, 0.5)",
                },
              ];
              const colors = stepColors[index] || stepColors[0];

              return (
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
                    className="relative z-10 w-16 h-16 rounded-full flex items-center justify-center text-2xl font-bold text-white mb-6"
                    style={{
                      background: colors.bg,
                      boxShadow: `0 10px 15px -3px ${colors.shadow}, 0 4px 6px -4px ${colors.shadow}`,
                    }}
                    whileHover={{ scale: 1.1 }}
                    transition={{ type: "spring", stiffness: 400 }}
                  >
                    {step.number}
                    {/* Pulse animation */}
                    <motion.div
                      className="absolute inset-0 rounded-full"
                      style={{ backgroundColor: colors.pulse }}
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
                    <div
                      className={`rounded-lg border border-white/10 overflow-hidden mb-3`}
                      style={{ backgroundColor: CODE_BG }}
                    >
                      <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
                        <span className="text-xs text-white/50 font-mono">
                          {step.language}
                        </span>
                        <CopyButton code={step.code} />
                      </div>
                      <div className="p-3 overflow-x-auto hidden-scrollbar">
                        <Highlight
                          theme={customNightOwl}
                          code={step.code.trim()}
                          language={
                            step.language === "bash" ? "bash" : "python"
                          }
                        >
                          {({ style, tokens, getLineProps, getTokenProps }) => (
                            <pre
                              className="text-xs font-mono !m-0 !p-0 text-left"
                              style={{
                                ...style,
                                backgroundColor: "transparent",
                              }}
                            >
                              {tokens.map((line, i) => (
                                <div key={i} {...getLineProps({ line })}>
                                  {line.map((token, key) => (
                                    <span
                                      key={key}
                                      {...getTokenProps({ token })}
                                    />
                                  ))}
                                </div>
                              ))}
                            </pre>
                          )}
                        </Highlight>
                      </div>
                    </div>

                    {/* Time estimate */}
                    <div className="flex items-center justify-center gap-2 text-xs text-white/40">
                      <Clock className="w-4 h-4" />
                      <span>{step.time}</span>
                    </div>
                  </div>

                  {/* Arrow to next step (hidden on last item and mobile) */}
                  {index < steps.length - 1 && (
                    <motion.div
                      className="hidden md:block absolute top-6 -right-4 z-20 text-white/30"
                      initial={{ x: 0 }}
                      animate={{ x: [0, 5, 0] }}
                      transition={{ duration: 1.5, repeat: Infinity }}
                    >
                      <ArrowRight className="w-8 h-8" />
                    </motion.div>
                  )}
                </motion.div>
              );
            })}
          </motion.div>
        </div>
      </div>
    </Section>
  );
}
