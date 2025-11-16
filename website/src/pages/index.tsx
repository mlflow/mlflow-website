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
    hotspots: [
      {
        id: "trace-breakdown",
        left: "0%",
        top: "22%",
        width: "25%",
        height: "80%",
        label: "Trace breakdown: spans & tool calls",
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
        body="Toggle through the core product areas to preview the experience."
      >
        <ProductTabs tabs={productTabs} />
      </Section>

      <EcosystemList />

      <BelowTheFold>
        <LatestNews />
        <GetStartedTagline />
      </BelowTheFold>

      <GlossyCardContainer>
          <GlossyCard>
            <Card
              title="GenAI Apps & Agents"
              bodySize="m"
              body="Enhance your GenAI applications with end-to-end observability, evaluations, AI gateway and tracking all in one integrated platform."
              padded
              rounded={false}
              cta={{
                href: "/genai",
                text: "Learn more",
                prominent: true,
              }}
              image={<img src={GenAI} alt="" className="hidden md:block" />}
            />
          </GlossyCard>
          <GlossyCard>
            <Card
              title="Model Training"
              bodySize="m"
              body="Streamline your machine learning workflows with end-to-end tracking, model management, and deployment."
              padded
              rounded={false}
              cta={{
                href: "/classical-ml",
                text: "Learn more",
                prominent: true,
              }}
              image={
                <img src={ModelTraining} alt="" className="hidden md:block" />
              }
            />
          </GlossyCard>
        </GlossyCardContainer>
    </Layout>
  );
}
