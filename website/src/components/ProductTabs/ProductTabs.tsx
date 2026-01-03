import clsx from "clsx";
import { ReactNode, useMemo, useState, useCallback } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Highlight, themes } from "prism-react-renderer";
import styles from "./ProductTabs.module.css";
import TracingTabImg from "@site/static/img/GenAI_home/GenAI_trace_darkmode.png";
import EvaluationTabImg from "@site/static/img/GenAI_home/GenAI_evaluation_darkmode.png";
import PromptTabImg from "@site/static/img/GenAI_home/GenAI_prompts_darkmode.png";
import ModelTrainingTabImg from "@site/static/img/GenAI_home/model_training_darkmode.png";

// Icons for tabs
const TracingIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M3 12h4l3-9 4 18 3-9h4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const PromptIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M14 2v6h6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M8 13h8M8 17h5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const EvaluationIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M3 3v18h18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M7 16l4-4 4 4 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const GatewayIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M4 6h16M4 6v12a2 2 0 002 2h12a2 2 0 002-2V6M4 6l2-4h12l2 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <circle cx="9" cy="12" r="1.5" fill="currentColor"/>
    <circle cx="15" cy="12" r="1.5" fill="currentColor"/>
  </svg>
);

const TrainingIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

// Homepage tabs - the 5 core features shown on the homepage
export const homepageTabs: Tab[] = [
  {
    id: "observability",
    label: "Observability",
    icon: <TracingIcon />,
    imageSrc: TracingTabImg,
    imageZoom: 160,
    title: "Observability",
    description: "Capture complete traces of your LLM applications/agents. Use traces to inspect failures and build eval datasets. Built on OpenTelemetry and support for 30+ popular LLM providers and agent frameworks.",
    docLink: "https://mlflow.org/docs/latest/genai/tracing/",
    quickstartLink: "https://mlflow.org/docs/latest/genai/tracing/quickstart/",
    codeSnippets: {
      python: `import mlflow
import openai

# Enable auto-tracing for OpenAI - just 1 line!
mlflow.openai.autolog()

# All OpenAI calls are now automatically traced
client = openai.OpenAI()
client.responses.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is MLflow?"},
    ],
)`,
      typescript: `import mlflow from "mlflow";
import OpenAI from "openai";

// Enable auto-tracing for OpenAI
mlflow.openai.autolog();

const client = new OpenAI();

// All OpenAI calls are now automatically traced
async function handleRequest(text: string): Promise<string> {
  const res = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [
      { role: "system", content: "Summarize in one sentence." },
      { role: "user", content: text },
    ],
  });
  return res.choices[0].message.content ?? "";
}`,
    },
  },
  {
    id: "evaluation",
    label: "Evaluation",
    icon: <EvaluationIcon />,
    imageSrc: EvaluationTabImg,
    title: "Evaluation",
    description: "Run systematic evaluations using LLM-as-judge, custom scorers, and human feedback. Track quality metrics over time and catch regressions.",
    docLink: "https://mlflow.org/docs/latest/genai/eval/",
    quickstartLink: "https://mlflow.org/docs/latest/genai/eval/",
    codeSnippets: {
      python: `import mlflow
from mlflow.genai.judges import Correctness, Guidelines

# Define a simple Q&A dataset with questions and expected answers
eval_dataset = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "outputs": "Paris",
        "expectations": {"expected_response": "Paris"},
    },
    {
        "inputs": {"question": "Who was the first person to build an airplane?"},
        "outputs": "I don't know",
        "expectations": {"expected_response": "Wright Brothers"},
    },
]

# Define evaluation criteria
correctness = Correctness(model="openai:/gpt-5")
is_english = Guidelines(guidelines="Response should be in English.", model="openai:/gpt-5", )

# Run evaluation
results = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[correctness, is_english]
)`,
      typescriptComingSoon: true,
    },
  },
  {
    id: "prompt",
    label: "Prompt Management",
    icon: <PromptIcon />,
    imageSrc: PromptTabImg,
    imageZoom: 150,
    title: "Prompt Management",
    description: "Version, test, and deploy prompts with full lineage tracking. Compare prompt performance across versions and collaborate with your team.",
    docLink: "https://mlflow.org/docs/latest/genai/prompt-engineering/",
    quickstartLink: "https://mlflow.org/docs/latest/genai/prompt-engineering/",
    codeSnippets: {
      python: `import mlflow

# Use double curly braces for variables in the template
initial_template = """
Summarize content you are provided with in {{ num_sentences }} sentences.
Sentences: {{ sentences }}
"""

# Register a new prompt
mlflow.genai.register_prompt(
    name="summarization",
    template=initial_template,
    # Optional: Provide a commit message to describe the changes
    commit_message="Initial commit",
    # Optional: Set tags applies to the prompt (across versions)
    tags={"task": "summarization", "language": "en"},
)

# Load and use prompts in your app
loaded = mlflow.genai.load_prompt("prompts:/summarization@latest")
`,
      typescriptComingSoon: true,
    },
  },
  {
    id: "gateway",
    label: "AI Gateway",
    icon: <GatewayIcon />,
    imageSrc: "/img/GenAI_home/GenAI_trace_darkmode.png",
    title: "AI Gateway",
    description: "Unified API gateway for all your LLM providers. Route requests, manage rate limits, handle fallbacks, and control costs through a single interface.",
    docLink: "https://mlflow.org/docs/latest/genai/gateway/",
    quickstartLink: "https://mlflow.org/docs/latest/genai/gateway/",
    codeSnippets: {
      python: `from openai import OpenAI

# Set the base URL to the MLflow Gateway URL
client = OpenAI(
    base_url="<Your MLflow Server URL>/gateway/mlflow/v1",
    api_key="dummy",
)

# Query an endpoint via the MLflow Gateway in the same way as you would with OpenAI
# Gateway handles rate limiting, fallback, payload transformation, etc.
client.chat.completions.create(
    model="claude-opus-4-5",
    messages=[{"role": "user", "content": "Hi, how are you?"}]
)`,
      typescript: `import OpenAI from "openai";

// Set the base URL to the MLflow Gateway URL
const client = new OpenAI(
    base_url="<Your MLflow Server URL>/gateway/mlflow/v1",
    api_key="dummy",
)

//Query an endpoint via the MLflow Gateway in the same way as you would with OpenAI
// Gateway handles rate limiting, fallback, payload transformation, etc.
const response = await client.chat.completions.create({
    model="claude-opus-4-5",
    messages=[{"role": "user", "content": "Hi, how are you?"}]
})`,
    },
  },
  {
    id: "training",
    label: "Model Training",
    icon: <TrainingIcon />,
    imageSrc: ModelTrainingTabImg,
    imageZoom: 150,
    title: "Model Training",
    description: "Track experiments, log parameters, metrics, and artifacts. Compare runs, reproduce results, and manage the full ML lifecycle from training to deployment.",
    docLink: "https://mlflow.org/docs/latest/ml/index.html",
    quickstartLink: "https://mlflow.org/docs/latest/getting-started/index.html",
    codeSnippets: {
      python: `import mlflow

# Enable autologging for training
mlflow.sklearn.autolog()

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Just train the model normally. MLflow will automatically
# log the parameters, metrics, and model.
lr = LogisticRegression(solver="lbfgs", max_iter=100)
lr.fit(X_train, y_train)`,
    },
  },
];

export type Tab = {
  id: string;
  label: string;
  imageSrc: string;
  icon?: ReactNode;
  title?: string;
  description?: string;
  docLink?: string;
  quickstartLink?: string;
  codeSnippets?: {
    python: string;
    typescript?: string;
    /** Show "Coming Soon!" for TypeScript SDK */
    typescriptComingSoon?: boolean;
  };
  hotspots?: Hotspot[];
  link?: string;
  /** Zoom level for the screenshot image (e.g., 115 = 115%). Default: 115 */
  imageZoom?: number;
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

// Copy button component
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
      className="absolute top-3 right-3 p-2 rounded-md bg-white/10 hover:bg-white/20 transition-colors z-10 group"
      aria-label={copied ? "Copied!" : "Copy code"}
    >
      <AnimatePresence mode="wait" initial={false}>
        {copied ? (
          <motion.svg
            key="check"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            exit={{ scale: 0 }}
            className="w-4 h-4 text-green-400"
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
            className="w-4 h-4 text-white/60 group-hover:text-white/90"
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

// Custom Night Owl theme with modified string color
const customNightOwl = {
  ...themes.nightOwl,
  styles: themes.nightOwl.styles.map((style) => {
    // Change string color from light green to a different color
    if (style.types.includes('string')) {
      return { ...style, style: { ...style.style, color: '#58a6ff' } }; // Vivid blue
    }
    return style;
  }),
};

// Color scheme for code blocks
const codeColorScheme = { theme: customNightOwl, bg: '#0d1117', headerBg: '#0d1117' };

// Code block component with prism-react-renderer syntax highlighting
const CodeBlock = ({ code, language }: {
  code: string;
  language: "python" | "typescript";
}) => {
  const prismLanguage = language === "typescript" ? "tsx" : "python";

  return (
    <Highlight theme={codeColorScheme.theme} code={code.trim()} language={prismLanguage}>
      {({ className, style, tokens, getLineProps, getTokenProps }) => (
        <div className="relative h-full" style={{ backgroundColor: codeColorScheme.bg }}>
          <CopyButton code={code.trim()} />
          <pre
            className="h-full overflow-auto leading-snug font-mono p-4 m-0 dark-scrollbar"
            style={{ ...style, backgroundColor: codeColorScheme.bg, fontSize: '14px', lineHeight: '1.5' }}
          >
            {tokens.map((line, i) => (
              <div key={i} {...getLineProps({ line })}>
                {line.map((token, key) => (
                  <span key={key} {...getTokenProps({ token })} />
                ))}
              </div>
            ))}
          </pre>
        </div>
      )}
    </Highlight>
  );
};

// QuickstartLink component with animated glow effect
const QuickstartLink = ({ href }: { href: string }) => {
  return (
    <motion.a
      href={href}
      target="_blank"
      rel="noreferrer noopener"
      className="relative inline-flex items-center gap-1 text-sm font-medium"
      whileHover="hover"
    >
      {/* Pulsing glow background */}
      <motion.span
        className="absolute inset-0 rounded-md -z-10"
        style={{
          background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(168, 85, 247, 0.3))',
          filter: 'blur(8px)',
        }}
        animate={{
          opacity: [0.4, 0.8, 0.4],
          scale: [1, 1.1, 1],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
      <span className="relative z-10 text-white px-3 py-1">Quickstart</span>
      <motion.span
        className="relative z-10 text-white"
        variants={{
          hover: { x: 4 },
        }}
        animate={{ x: [0, 4, 0] }}
        transition={{
          duration: 1,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      >
        →
      </motion.span>
    </motion.a>
  );
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
  const [codeLanguage, setCodeLanguage] = useState<"python" | "typescript">("python");

  const activeTab = useMemo(
    () => tabs.find((tab) => tab.id === activeTabId) ?? tabs[0],
    [activeTabId, tabs],
  );

  if (!activeTab) {
    return null;
  }

  return (
    <div className="w-full flex flex-col gap-6 p-8 rounded-2xl border border-white/10 bg-white/[0.01]">
      {/* Tab Navigation */}
      <div className="flex flex-wrap justify-center gap-6 md:gap-8 border-b border-white/10 pb-0">
        {tabs.map((tab) => {
          const isActive = tab.id === activeTab.id;
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTabId(tab.id)}
              className="group relative flex items-center gap-2 text-base font-medium focus:outline-none px-1 py-3 transition-all"
            >
              {/* Icon */}
              {tab.icon && (
                <span
                  className={clsx(
                    "transition-colors",
                    !isActive && "text-white/40 group-hover:text-white/60"
                  )}
                  style={isActive ? { color: '#9066cc' } : undefined}
                >
                  {tab.icon}
                </span>
              )}

              {/* Label */}
              <span
                className={clsx(
                  "transition-colors font-semibold",
                  isActive
                    ? "text-white"
                    : "text-white/50 group-hover:text-white/70",
                )}
              >
                {tab.label}
              </span>

              {/* Active indicator line */}
              {isActive && (
                <motion.div
                  layoutId="activeTab"
                  className="absolute bottom-0 left-0 right-0 h-[2px]"
                  style={{
                    background: 'linear-gradient(90deg, #e05585, #9066cc, #5a8fd4)',
                  }}
                  transition={{ type: "spring", stiffness: 500, damping: 30 }}
                />
              )}
            </button>
          );
        })}
      </div>

      {/* Title and Description with Left/Right Navigation */}
      <div className="flex items-center gap-4 py-2">
          {/* Left Arrow */}
          <button
            onClick={() => {
              const currentIndex = tabs.findIndex(t => t.id === activeTab.id);
              const prevIndex = currentIndex > 0 ? currentIndex - 1 : tabs.length - 1;
              setActiveTabId(tabs[prevIndex].id);
            }}
            className="shrink-0 w-10 h-10 flex items-center justify-center text-white/40 hover:text-white transition-all"
            aria-label="Previous tab"
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M15 18l-6-6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>

          {/* Title and Description */}
          <div className="flex-1 min-w-0">
            <h3 className="text-2xl md:text-3xl font-bold text-white mb-2">
              {activeTab.title || activeTab.label}
            </h3>
            <p className="text-gray-400 leading-relaxed mb-3" style={{ marginTop: '4px' }}>
              {activeTab.description}
            </p>
            {activeTab.quickstartLink && (
              <QuickstartLink href={activeTab.quickstartLink} />
            )}
          </div>

          {/* Right Arrow */}
          <button
            onClick={() => {
              const currentIndex = tabs.findIndex(t => t.id === activeTab.id);
              const nextIndex = currentIndex < tabs.length - 1 ? currentIndex + 1 : 0;
              setActiveTabId(tabs[nextIndex].id);
            }}
            className="shrink-0 w-10 h-10 flex items-center justify-center text-white/40 hover:text-white transition-all"
            aria-label="Next tab"
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M9 18l6-6-6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </div>

      {/* Main Content: Code Left, Screenshot Right */}
      <div className="grid grid-cols-1 lg:grid-cols-2 px-4">
          {/* Left: Code Snippet */}
          <div className="relative">
            <div
              className="overflow-hidden h-[400px] flex flex-col"
              style={{ backgroundColor: codeColorScheme.bg }}
            >
              {/* Code language tabs */}
              <div
                className="flex border-b border-white/10"
                style={{ backgroundColor: codeColorScheme.headerBg }}
              >
                <button
                  onClick={() => setCodeLanguage("python")}
                  className={clsx(
                    "px-4 py-2.5 text-sm font-medium transition-colors",
                    codeLanguage === "python"
                      ? "text-white bg-white/10 border-b-2 border-blue-400"
                      : "text-white/60 hover:text-white/80"
                  )}
                >
                  Python SDK
                </button>
                {(activeTab.codeSnippets?.typescript || activeTab.codeSnippets?.typescriptComingSoon) && (
                  <button
                    onClick={() => setCodeLanguage("typescript")}
                    className={clsx(
                      "px-4 py-2.5 text-sm font-medium transition-colors",
                      codeLanguage === "typescript"
                        ? "text-white bg-white/10 border-b-2 border-blue-400"
                        : "text-white/60 hover:text-white/80"
                    )}
                  >
                    JS/TS SDK
                  </button>
                )}
              </div>

              {/* Code content */}
              <div className="flex-1 overflow-auto">
                {activeTab.codeSnippets && (
                  codeLanguage === "typescript" && activeTab.codeSnippets.typescriptComingSoon && !activeTab.codeSnippets.typescript ? (
                    <div className="flex items-center justify-center h-full text-white/60">
                      <div className="text-center">
                        <div className="text-2xl mb-2">Coming Soon!</div>
                        <div className="text-sm text-white/40">JS/TS SDK support is in development</div>
                      </div>
                    </div>
                  ) : (
                    <CodeBlock
                      code={codeLanguage === "typescript" && activeTab.codeSnippets.typescript
                        ? activeTab.codeSnippets.typescript
                        : activeTab.codeSnippets.python}
                      language={codeLanguage === "typescript" && activeTab.codeSnippets.typescript ? "typescript" : "python"}
                    />
                  )
                )}
              </div>

            </div>
          </div>

          {/* Right: Screenshot with gradient background */}
          <div className="relative overflow-hidden h-[400px]">
            {/* Dark gradient background with vivid red-to-blue theme */}
            <div
              className="absolute inset-0"
              style={{
                background: 'linear-gradient(135deg, #2a1020 0%, #251535 25%, #152040 50%, #102545 100%)',
              }}
            />

            {/* Subtle grid pattern */}
            <div
              className="absolute inset-0 opacity-40"
              style={{
                backgroundImage: `
                  linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
                  linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px)
                `,
                backgroundSize: '50px 50px',
              }}
            />

            {/* Blurred orbs for visual interest - more vivid */}
            <div className="absolute -top-20 -left-20 w-72 h-72 rounded-full opacity-50" style={{ background: 'radial-gradient(circle, rgba(224,85,133,0.5) 0%, transparent 70%)' }} />
            <div className="absolute -bottom-20 -right-20 w-64 h-64 rounded-full opacity-45" style={{ background: 'radial-gradient(circle, rgba(79,172,254,0.5) 0%, transparent 70%)' }} />
            <div className="absolute top-1/3 -right-10 w-56 h-56 rounded-full opacity-40" style={{ background: 'radial-gradient(circle, rgba(168,85,247,0.4) 0%, transparent 70%)' }} />

            {/* Screenshot image with gradient border (top-left only) */}
            <div
              className="absolute bottom-0 right-0 w-[93%] h-[93%] z-10 rounded-tl-lg pt-[1px] pl-[1px]"
              style={{
                background: 'linear-gradient(135deg, rgba(255,255,255,0.25) 0%, rgba(255,255,255,0.05) 50%, transparent 100%)',
              }}
            >
              <div className="w-full h-full rounded-tl-lg pt-[4px] pl-[4px] overflow-hidden" style={{ backgroundColor: '#11171d' }}>
                <img
                  src={activeTab.imageSrc}
                  alt={`${activeTab.label} screenshot`}
                  className="rounded-tl object-cover object-left-top"
                  style={{
                    width: `${activeTab.imageZoom ?? 115}%`,
                    height: `${activeTab.imageZoom ?? 115}%`,
                  }}
                  loading="lazy"
                />
              </div>
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
