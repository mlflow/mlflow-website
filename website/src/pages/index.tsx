import {
  LatestNews,
  Layout,
  GlossyCard,
  GetStartedTagline,
  LogosCarousel,
  AboveTheFold,
  BelowTheFold,
  Card,
  GlossyCardContainer,
  EcosystemList,
  ProductTabs,
  Section,
  StatsBand,
} from "../components";
import GenAI from "@site/static/img/Home_page_hybrid/GenAI Apps & Agents.png";
import ModelTraining from "@site/static/img/Home_page_hybrid/Model Training.png";
import EvaluationTabImg from "@site/static/img/GenAI_home/GenAI_evaluation_darkmode.png";
import MonitoringTabImg from "@site/static/img/GenAI_home/GenAI_monitor_darkmode.png";
import AnnotationTabImg from "@site/static/img/GenAI_home/GenAI_annotation_darkmode.png";
import PromptTabImg from "@site/static/img/GenAI_home/GenAI_prompts_darkmode.png";
import OptimizeTabImg from "@site/static/img/GenAI_home/GenAI_optimize_darkmode.png";

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
    <path
      d="M8 15h4"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
    />
    <circle cx="10" cy="9" r="1" fill="currentColor" />
  </svg>
);

const defaultTabImage = "/img/GenAI_home/GenAI_trace_darkmode.png";

const productTabs = [
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
        description: "MLflow visualized the execution flow of your GenAI applications, including LLM calls, tool invocations, retrieval steps, and more.",
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
        description: "Each span represents a single step in the execution flow. They capture the inputs, outputs, token usage, latency, and many more metadata about the step.",
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
        description: "MLflow provides an UI and APIs for you to collect feedback from your users or domain experts on the quality of the application's output.",
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
        description: "The trace header panel provides a summary of the trace, including the the latency, token usage, session ID, and more.",
        direction: "bottom",
        link: "https://mlflow.org/docs/latest/genai/tracing/",
      }
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

export default function Home(): JSX.Element {
  return (
    <Layout>
      <AboveTheFold
        title="Deliver production-ready AI"
        body="The open source developer platform to build AI applications and models with confidence."
        minHeight="small"
      />
      <LogosCarousel />

      <Section
        title="End-to-End AIOps Platform"
        body="One MLflow platform to design, ship, and operate GenAI/LLM apps and agents. From experimentation through observability, optimization, and production."
      >
        <ProductTabs tabs={productTabs} />
      </Section>

      <EcosystemList />

      <StatsBand />

      <Section
        label="For every team"
        title="One Platform, Every Role"
        body="Choose the MLflow workspace that matches how you build, whether you're shipping GenAI apps or training models."
        headingLevel={2}
      >
        <GlossyCardContainer className="max-w-6xl">
          <GlossyCard className="github-stats-card">
            <div className="flex h-full flex-col gap-6 p-6 md:p-8">
              <div className="inline-flex items-center gap-2 self-start rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.12em] text-white/80">
                <span className="h-2 w-2 rounded-full bg-sky-300 shadow-[0_0_12px_rgba(56,189,248,0.65)]" />
                GenAI builders
              </div>
              <Card
                title="GenAI Apps & Agents"
                bodySize="m"
                body="Enhance GenAI applications with end-to-end observability, evaluations, gateway routing, and tracking in a single integrated platform."
                padded={false}
                rounded={false}
                image={<img src={GenAI} alt="" className="hidden md:block" />}
              />
            </div>
          </GlossyCard>
          <GlossyCard className="github-stats-card">
            <div className="flex h-full flex-col gap-6 p-6 md:p-8">
              <div className="inline-flex items-center gap-2 self-start rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.12em] text-white/80">
                <span className="h-2 w-2 rounded-full bg-emerald-300 shadow-[0_0_12px_rgba(16,185,129,0.7)]" />
                Model owners
              </div>
              <Card
                title="Model Training"
                bodySize="m"
                body="Streamline machine learning workflows with experiment tracking, model management, deployment, and monitoring built-in."
                padded={false}
                rounded={false}
                image={
                  <img src={ModelTraining} alt="" className="hidden md:block" />
                }
              />
            </div>
          </GlossyCard>
        </GlossyCardContainer>
      </Section>

      <BelowTheFold>
        <LatestNews />
        <GetStartedTagline />
      </BelowTheFold>
    </Layout>
  );
}
