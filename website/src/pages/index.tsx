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
  GetStartedButton,
  Button,
} from "../components";
import Link from "@docusaurus/Link";
import { MLFLOW_DOCS_URL, MLFLOW_DBX_TRIAL_URL } from "../constants";
import GenAI from "@site/static/img/Home_page_hybrid/GenAI Apps & Agents.png";
import ModelTraining from "@site/static/img/Home_page_hybrid/Model Training.png";
import { defaultProductTabs as productTabs } from "../components/ProductTabs/ProductTabs";


export default function Home(): JSX.Element {
  return (
    <Layout>
      <AboveTheFold
        title="Deliver Production-Ready AI"
        body={
          <>
            The open source platform for <span className="text-white">developing</span>,{" "}
            <span className="text-white">testing</span>, and{" "}
            <span className="text-white">governing</span> AI applications and
            agents.
          </>
        }
        minHeight="small"
        actions={
          <>
            <GetStartedButton
              link={`${MLFLOW_DOCS_URL}genai/index.html`}
            />
            <Link to={MLFLOW_DBX_TRIAL_URL}>
              <Button variant="outline" size="medium">
                Try in Cloud
              </Button>
            </Link>
          </>
        }
      >
      </AboveTheFold>

      <Section
        title="One Platform for AI Engineering"
        headingLevel={2}
        body="Build, test, and monitor LLM apps and agents in a single platform. MLflow accelerates your end-to-end AI projects from experimentation to production. You don't need to switch between tools."
        align="center"
      >
        <ProductTabs tabs={productTabs} />
      </Section>

      <div className="flex flex-col gap-32"></div>
      <EcosystemList />
      <div className="flex flex-col gap-32"></div>
      <StatsBand />

      <div className="flex flex-col gap-32"></div>
      <Section
        label="For every team"
        title="No More Tool Silos"
        body="Choose the MLflow workspace that matches how you build, whether you're shipping GenAI apps or training models."
        align="center"
      >
        <GlossyCardContainer className="max-w-6xl">
          <GlossyCard className="github-stats-card">
            <div className="flex h-full flex-col gap-6 p-6 md:p-8">
              <div className="inline-flex items-center gap-2 self-start rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.12em] text-white/80">
                <span className="h-2 w-2 rounded-full bg-sky-300 shadow-[0_0_12px_rgba(56,189,248,0.65)]" />
                GenAI builders
              </div>
              <Card
                title="LLM Apps & Agents"
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
      <div className="flex flex-col gap-16"></div>

      <BelowTheFold>
        <LatestNews />
      </BelowTheFold>
    </Layout>
  );
}
