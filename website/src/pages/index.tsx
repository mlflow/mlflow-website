import {
  LatestNews,
  Layout,
  GlossyCard,
  GetStartedTagline,
  Customers,
  AboveTheFold,
  BelowTheFold,
  Card,
  GlossyCardContainer,
} from "../components";

export default function Home(): JSX.Element {
  return (
    <Layout>
      <AboveTheFold
        title="Deliver production-ready AI"
        body="The developer platform to build AI applications and models with confidence"
      >
        <GlossyCardContainer>
          <GlossyCard>
            <Card
              title="GenAI Apps & Agents"
              bodySize="m"
              body="Enhance your GenAI applications with end-to-end observability, monitoring, and enterprise governance, all in one integrated platform."
              padded
              rounded={false}
              cta={{
                href: "/genai",
                text: "Learn more >",
                prominent: true,
              }}
              image={
                <div className="w-full bg-brand-black/60 min-h-[270px] rounded-b-4xl hidden md:block" />
              }
            />
          </GlossyCard>
          <GlossyCard>
            <Card
              title="Model Training"
              bodySize="m"
              body="Streamline your machine learning workflows with enterprise-grade tracking, model management, and deployment."
              padded
              rounded={false}
              cta={{
                href: "/classical-ml",
                text: "Learn more >",
                prominent: true,
              }}
              image={
                <div className="w-full bg-brand-black/60 min-h-[270px] rounded-b-4xl hidden md:block" />
              }
            />
          </GlossyCard>
        </GlossyCardContainer>
      </AboveTheFold>

      <Customers />

      <BelowTheFold>
        <LatestNews />
        <GetStartedTagline />
      </BelowTheFold>
    </Layout>
  );
}
