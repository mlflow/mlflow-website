import clsx from "clsx";
import { useState, useRef } from "react";
import {
  motion,
  AnimatePresence,
  useScroll,
  useMotionValueEvent,
} from "motion/react";
import { Highlight } from "prism-react-renderer";
import { CopyButton } from "../CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../CodeSnippet/codeTheme";
import TracingTabImg from "@site/static/img/GenAI_home/GenAI_trace_darkmode.png";
import EvaluationTabImg from "@site/static/img/GenAI_home/GenAI_evaluation_darkmode.png";
import GatewayTabImg from "@site/static/img/GenAI_home/GenAI_gateway_darkmode.png";
import PromptTabImg from "@site/static/img/GenAI_home/GenAI_prompts_darkmode.png";
import ExperimentTrackingImg from "@site/static/img/GenAI_home/model_training_darkmode.png";
import ModelRegistryImg from "@site/static/img/GenAI_home/model_registry_darkmode.png";
import DeploymentImg from "@site/static/img/GenAI_home/deployment.png";

// Feature type definition
type Feature = {
  id: string;
  title: string;
  description: string;
  imageSrc?: string;
  imageZoom?: number;
  imagePosition?: string; // Custom object-position value (e.g., "30% top")
  quickstartLink?: string;
  codeSnippet: string;
  codeLanguage?: "python" | "typescript";
};

// LLMs & Agents features
const llmAgentFeatures: Feature[] = [
  {
    id: "observability",
    title: "Observability",
    description:
      "Capture complete traces of your LLM applications and agents to get deep insights into their behavior. Built on OpenTelemetry and supports any LLM provider and agent framework.",
    imageSrc: TracingTabImg,
    imageZoom: 160,
    quickstartLink: "https://mlflow.org/docs/latest/genai/tracing/quickstart/",
    codeSnippet: `import mlflow
import openai

# Enable auto-tracing for OpenAI - just 1 line!
mlflow.openai.autolog()

# All OpenAI calls are now automatically traced
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "What is MLflow?"},
    ],
)`,
  },
  {
    id: "evaluation",
    title: "Evaluation",
    description:
      "Run systematic evaluations, track quality metrics over time, and catch regressions before they reach production. Choose from 50+ built-in metrics and LLM judges, or define your own with highly flexible APIs.",
    imageSrc: EvaluationTabImg,
    quickstartLink:
      "https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/",
    codeSnippet: `import mlflow
from mlflow.genai.scorers import Correctness

# Define evaluation dataset
eval_data = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "outputs": {"response": "Paris"},
        "expectations": {"expected_response": "Paris"},
    },
]

# Run evaluation with LLM-as-judge
results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[Correctness()]
)`,
  },
  {
    id: "prompt",
    title: "Prompt & Optimization",
    description:
      "Version, test, and deploy prompts with full lineage tracking. Automatically optimize prompts with state-of-the-art algorithms to improve performance.",
    imageSrc: PromptTabImg,
    imageZoom: 150,
    quickstartLink: "https://mlflow.org/docs/latest/genai/prompt-registry/",
    codeSnippet: `import mlflow

# Register a prompt template
mlflow.genai.register_prompt(
    name="summarization",
    template="""
Summarize the following content in {{ num_sentences }} sentences.
Content: {{ content }}
""",
    commit_message="Initial version",
)

# Load and use prompts in your app
prompt = mlflow.genai.load_prompt("prompts:/summarization@latest")
formatted = prompt.format(num_sentences=2, content="...")`,
  },
  {
    id: "gateway",
    title: "AI Gateway",
    description:
      "Unified API gateway for all LLM providers. Route requests, manage rate limits, handle fallbacks, and control costs through a unified OpenAI-compatible interface.",
    imageSrc: GatewayTabImg,
    imagePosition: "0% top",
    quickstartLink:
      "https://mlflow.org/docs/latest/genai/governance/ai-gateway/quickstart/",
    codeSnippet: `from openai import OpenAI

# Point to MLflow AI Gateway - OpenAI compatible API
client = OpenAI(
    base_url="http://localhost:5000/gateway/v1",
    api_key="mlflow",  # Gateway handles auth
)

# Gateway routes to configured providers
# with rate limiting, fallbacks, and cost tracking
response = client.chat.completions.create(
    model="gpt-5.2",  # or "claude-opus-4.5", "gemini-3-flash", etc.
    messages=[{"role": "user", "content": "Hello!"}]
)`,
  },
];

// Model Training features
const modelTrainingFeatures: Feature[] = [
  {
    id: "experiment-tracking",
    title: "Experiment Tracking",
    description:
      "Track experiments, log parameters, metrics, and artifacts. Compare runs side-by-side, reproduce results, and collaborate with your team on ML experiments.",
    imageSrc: ExperimentTrackingImg,
    imageZoom: 150,
    quickstartLink:
      "https://mlflow.org/docs/latest/ml/getting-started/quickstart/",
    codeSnippet: `import mlflow

# Enable autologging for your ML framework
mlflow.sklearn.autolog()

# Train your model - MLflow automatically logs everything
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)  # Parameters, metrics, model logged automatically`,
  },
  {
    id: "model-registry",
    title: "Model Registry",
    description:
      "Central hub to manage the full lifecycle of ML models. Version models, track lineage, manage stage transitions, and collaborate on model development.",
    imageSrc: ModelRegistryImg,
    imageZoom: 120,
    quickstartLink:
      "https://mlflow.org/docs/latest/ml/model-registry/tutorial/",
    codeSnippet: `import mlflow

# Register a model from a run
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "fraud-detection-model")

# Load a registered model for inference
model = mlflow.pyfunc.load_model(
    "models:/fraud-detection-model@champion"
)

# Make predictions
predictions = model.predict(new_data)`,
  },
  {
    id: "model-deployment",
    title: "Deployment",
    description:
      "Deploy models to production with a single command. Serve models as REST APIs, batch inference jobs, or integrate with cloud platforms like AWS, Azure, and Databricks.",
    imageSrc: DeploymentImg,
    imageZoom: 110,
    quickstartLink: "https://mlflow.org/docs/latest/ml/deployment/",
    codeSnippet: `# Serve model as REST API
mlflow models serve -m "models:/my-model@champion" -p 5000

# Or deploy to cloud platforms
mlflow deployments create -t sagemaker \\
    -m "models:/my-model@champion" \\
    --name my-deployment

# Query the deployed model
import requests
response = requests.post(
    "http://localhost:5000/invocations",
    json={"inputs": [[1, 2, 3, 4]]}
)`,
    codeLanguage: "python",
  },
];

// Category tabs
const categories = [
  {
    id: "llm-agents",
    label: "LLMs & Agents",
    features: llmAgentFeatures,
  },
  {
    id: "model-training",
    label: "Model Training",
    features: modelTrainingFeatures,
  },
];

// Code block component
const CodeBlock = ({
  code,
  language = "python",
}: {
  code: string;
  language?: "python" | "typescript";
}) => {
  const prismLanguage = language === "typescript" ? "tsx" : "python";

  return (
    <Highlight
      theme={customNightOwl}
      code={code.trim()}
      language={prismLanguage}
    >
      {({ style, tokens, getLineProps, getTokenProps }) => (
        <div className="relative h-full" style={{ backgroundColor: CODE_BG }}>
          <CopyButton
            code={code.trim()}
            className="absolute top-3 right-3 p-2 rounded-md z-10"
          />
          <pre
            className="h-full overflow-auto leading-snug font-mono p-4 m-0 dark-scrollbar"
            style={{
              ...style,
              backgroundColor: CODE_BG,
              fontSize: "13px",
              lineHeight: "1.5",
            }}
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
      className="relative inline-flex items-center gap-1 text-sm font-medium w-fit"
      whileHover="hover"
    >
      {/* Pulsing glow background - sized to fit content */}
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
};

// Code icon component
const CodeIcon = () => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <polyline points="16 18 22 12 16 6" />
    <polyline points="8 6 2 12 8 18" />
  </svg>
);

// Screenshot icon component
const ScreenshotIcon = () => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
    <circle cx="8.5" cy="8.5" r="1.5" />
    <polyline points="21 15 16 10 5 21" />
  </svg>
);

// Feature Media Card with hover toggle button
const FeatureMediaCard = ({ feature }: { feature: Feature }) => {
  const [showCode, setShowCode] = useState(false);

  return (
    <div className="relative h-[350px] rounded-xl overflow-hidden border border-white/10 group">
      {/* Content area */}
      <AnimatePresence mode="wait">
        {!showCode ? (
          <motion.div
            key="screenshot"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="absolute inset-0"
          >
            {/* Dark gradient background with vivid red-to-blue theme */}
            <div
              className="absolute inset-0"
              style={{
                background:
                  "linear-gradient(135deg, #2a1020 0%, #251535 25%, #152040 50%, #102545 100%)",
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
                backgroundSize: "50px 50px",
              }}
            />

            {/* Blurred orbs for visual interest - more vivid */}
            <div
              className="absolute -top-20 -left-20 w-72 h-72 rounded-full opacity-50"
              style={{
                background:
                  "radial-gradient(circle, rgba(224,85,133,0.5) 0%, transparent 70%)",
              }}
            />
            <div
              className="absolute -bottom-20 -right-20 w-64 h-64 rounded-full opacity-45"
              style={{
                background:
                  "radial-gradient(circle, rgba(79,172,254,0.5) 0%, transparent 70%)",
              }}
            />
            <div
              className="absolute top-1/3 -right-10 w-56 h-56 rounded-full opacity-40"
              style={{
                background:
                  "radial-gradient(circle, rgba(168,85,247,0.4) 0%, transparent 70%)",
              }}
            />

            {/* Screenshot image */}
            <div
              className="absolute bottom-0 right-0 w-[93%] h-[93%] z-10 pt-[1px] rounded-tl-lg pl-[1px]"
              style={{
                background:
                  "linear-gradient(135deg, rgba(255,255,255,0.25) 0%, rgba(255,255,255,0.05) 50%, transparent 100%)",
              }}
            >
              <div
                className="w-full h-full pt-[4px] overflow-hidden rounded-tl-lg pl-[4px]"
                style={{ backgroundColor: "#11171d" }}
              >
                <img
                  src={feature.imageSrc}
                  alt={`${feature.title} screenshot`}
                  className="object-cover rounded-tl"
                  style={{
                    width: `${feature.imageZoom ?? 115}%`,
                    height: `${feature.imageZoom ?? 115}%`,
                    objectPosition: feature.imagePosition ?? "left top",
                  }}
                  loading="lazy"
                />
              </div>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="code"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="absolute inset-0"
          >
            <CodeBlock
              code={feature.codeSnippet}
              language={feature.codeLanguage}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Toggle button - bottom right with glow effect */}
      <motion.button
        onClick={() => setShowCode(!showCode)}
        className="absolute bottom-3 right-3 z-20 px-3 py-1.5 rounded-lg bg-black/80 hover:bg-black/90 text-white/90 hover:text-white transition-all backdrop-blur-sm border border-white/20 flex items-center gap-1.5 text-xs font-medium"
        style={{
          boxShadow: "0 0 8px rgba(99, 102, 241, 0.2)",
        }}
        whileHover={{
          boxShadow: "0 0 12px rgba(99, 102, 241, 0.35)",
        }}
        aria-label={showCode ? "Show screenshot" : "Show code"}
      >
        {showCode ? (
          <>
            <ScreenshotIcon />
            <span>Screenshot</span>
          </>
        ) : (
          <>
            <CodeIcon />
            <span>Code</span>
          </>
        )}
      </motion.button>
    </div>
  );
};

// Helper to interpolate between two hex colors
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

// Feature text section component - sticky card with QuickstartLink
const FeatureTextSection = ({
  feature,
  visibility = 1, // 0 = fully dark, 1 = fully visible
}: {
  feature: Feature;
  visibility?: number;
}) => {
  // Interpolate colors based on visibility (0 to 1)
  const titleColor = interpolateColor("#1a1a1a", "#ffffff", visibility);
  const descColor = interpolateColor("#1a1a1a", "#9ca3af", visibility);

  return (
    <div className="border-[rgba(255,255,255,0.08)] border-t border-b min-h-[350px] w-full lg:sticky top-24 bg-brand-black flex flex-col justify-center gap-y-8 py-10">
      {/* Text content */}
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

// Sticky features grid component (like StickyGrid with original spacing)
const StickyFeaturesGrid = ({ features }: { features: Feature[] }) => {
  const [scrollProgress, setScrollProgress] = useState(0);
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start 40vh", "end 40vh"],
  });

  useMotionValueEvent(scrollYProgress, "change", (latest) => {
    setScrollProgress(latest);
  });

  // Calculate visibility for each card based on distance from current scroll position
  // Uses a trapezoid shape: fade in -> plateau at 1.0 -> fade out
  const getVisibility = (index: number) => {
    const cardPosition = index / features.length;
    const distance = Math.abs(scrollProgress - cardPosition);

    // Plateau range where visibility stays at 1.0 (when card matches right image)
    const plateauRange = 0.08;
    // Fade range for transitioning in/out
    const fadeRange = 0.25;

    if (distance <= plateauRange) {
      // In the plateau zone - full visibility
      return 1;
    } else {
      // Fading zone - calculate based on distance from plateau edge
      const distanceFromPlateau = distance - plateauRange;
      const visibility = Math.max(0, 1 - distanceFromPlateau / fadeRange);
      return visibility;
    }
  };

  // Determine active feature for the image panel
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

      {/* Right: Sticky image panel (desktop only) - positioned at 40% viewport height */}
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

// Main component
export function ProductTabs() {
  const [activeCategory, setActiveCategory] = useState(categories[0].id);
  const activeFeatures =
    categories.find((c) => c.id === activeCategory)?.features ?? [];

  return (
    <div className="w-full flex flex-col gap-12">
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
    </div>
  );
}
