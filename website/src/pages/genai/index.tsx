import {
  LatestNews,
  Layout,
  BelowTheFold,
  EcosystemList,
  ProductTabs,
  StatsBand,
  HeroSection,
  HighlightedKeyword,
  BenefitsSection,
  ProcessSection,
  FAQSection,
} from "../../components";
import { MLFLOW_GENAI_DOCS_URL } from "../../constants";

export default function GenAi(): JSX.Element {
  return (
    <Layout>
      <HeroSection
        title="Ship High-Quality GenAI, Fast"
        subtitle={
          <>
            Building GenAI products is all about iteration.
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
          href: "#get-started",
        }}
        secondaryCTA={{
          label: "View Docs",
          href: MLFLOW_GENAI_DOCS_URL,
        }}
      />

      <div className="w-full px-4 md:px-8 lg:px-16 pb-36">
        <div className="max-w-7xl mx-auto">
          <ProductTabs />
        </div>
      </div>

      <StatsBand />
      <EcosystemList />
      <BenefitsSection />
      <ProcessSection />
      <FAQSection />

      <BelowTheFold hideGetStarted>
        <LatestNews />
      </BelowTheFold>
    </Layout>
  );
}
