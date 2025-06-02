import {
  LatestNews,
  Layout,
  GlossyCard,
  GetStartedTagline,
  Testimonials,
  LogosCarousel,
  AboveTheFold,
  BelowTheFold,
  Card,
  GlossyCardContainer,
} from "../components";

export default function Home(): JSX.Element {
  return (
    <Layout variant="colorful">
      <AboveTheFold
        title="AI and ML made simple"
        body="The AI developer platform to build AI applications and models with confidence"
      >
        <GlossyCardContainer>
          <GlossyCard>
            <Card
              title="GenAI Apps & Agents"
              bodySize="m"
              body="Enhance your GenAI applications with end-to-end observability, monitoring, and enterprise governance, all in one integrated platform."
              padded
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
        </GlossyCardContainer>
      </AboveTheFold>

      <Testimonials />

      <BelowTheFold>
        <LatestNews />
        <GetStartedTagline />
      </BelowTheFold>
    </Layout>
  );
}
