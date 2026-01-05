import {
  LatestNews,
  Layout,
  GlossyCard,
  BelowTheFold,
  Card,
  GlossyCardContainer,
  EcosystemList,
  ProductTabs,
  Section,
  StatsBand,
  RunningAnywhere,
  HeroSection,
  HighlightedKeyword,
  BenefitsSection,
  ProcessSection,
  FAQSection,
} from "../components";
import { MLFLOW_DOCS_URL, MLFLOW_DBX_TRIAL_URL } from "../constants";
import GenAI from "@site/static/img/Home_page_hybrid/GenAI Apps & Agents.png";
import ModelTraining from "@site/static/img/Home_page_hybrid/Model Training.png";
import { homepageTabs } from "../components/ProductTabs/ProductTabs";

export default function Home(): JSX.Element {
  return (
    <Layout>
      {/* 1. HERO SECTION */}
      <HeroSection
        title="Deliver High-Quality AI, Fast"
        subtitle={
          <>
            Building AI products is all about iteration.
            <br />
            MLflow lets you move 10x faster by simplifying how you <br />
            <HighlightedKeyword>debug</HighlightedKeyword>,{" "}
            <HighlightedKeyword>test</HighlightedKeyword>, and{" "}
            <HighlightedKeyword>evaluate</HighlightedKeyword> your LLM
            applications and Agents.
          </>
        }
        primaryCTA={{
          label: "Get Started",
          href: `${MLFLOW_DOCS_URL}genai/index.html`,
        }}
        secondaryCTA={{
          label: "Try in Cloud",
          href: MLFLOW_DBX_TRIAL_URL,
        }}
      />

      {/* 2. FEATURES SECTION - 5 core capabilities with tabs */}
      <div className="w-full px-4 md:px-8 lg:px-16 pb-36">
        <div className="max-w-7xl mx-auto">
          <ProductTabs tabs={homepageTabs} />
        </div>
      </div>

      {/* 3. TRUST LOGOS */}
      <StatsBand />

      {/* 4. OPEN AND NEUTRAL - Vendor lock-in free, integrations (kept as is per user request) */}
      <div className="flex flex-col gap-16"></div>
      <EcosystemList />

      {/* 5. BENEFITS SECTION - Why teams choose MLflow */}
      <div className="flex flex-col gap-16"></div>
      <BenefitsSection />

      {/* 6. PROCESS SECTION - 1-2-3 getting started steps */}
      <div className="flex flex-col gap-16"></div>
      <ProcessSection />

      {/* 7. RUNNING ANYWHERE - Cloud platform logos
      <div className="flex flex-col gap-16"></div>
      <RunningAnywhere /> */}

      {/* 10. FAQ SECTION - Common questions */}
      <div className="flex flex-col gap-16"></div>
      <FAQSection />

      {/* 12. COMMUNITY & LATEST NEWS */}
      <BelowTheFold hideGetStarted>
        <LatestNews />
      </BelowTheFold>
    </Layout>
  );
}
