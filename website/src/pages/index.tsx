import {
  LatestNews,
  Layout,
  GlossyCard,
  LogosCarousel,
  AboveTheFold,
  BelowTheFold,
  Card,
  GlossyCardContainer,
  EcosystemList,
  ProductTabs,
  Section,
  StatsBand,
  Challenges,
} from "../components";
import GenAI from "@site/static/img/Home_page_hybrid/GenAI Apps & Agents.png";
import ModelTraining from "@site/static/img/Home_page_hybrid/Model Training.png";
import { defaultProductTabs as productTabs } from "../components/ProductTabs/ProductTabs";


export default function Home(): JSX.Element {
  return (
    <Layout>
      <AboveTheFold
        title="Deliver production-ready AI"
        body="The open source developer platform to build AI applications and models with confidence."
        minHeight="small"
      >
      </AboveTheFold>
      <LogosCarousel />

      <Challenges />
      <Section
        title="Solution: End-to-End AIOps"
        body="One MLflow platform to build, test, and monitor GenAI/LLM apps and agents. From experimentation to production."
      >
        <ProductTabs tabs={productTabs} />
      </Section>

      <EcosystemList />
      <StatsBand />

      <Section
        label="For every team"
        title="One Platform, Every Role"
        body="Choose the MLflow workspace that matches how you build, whether you're shipping GenAI apps or training models."
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
      </BelowTheFold>
    </Layout>
  );
}
