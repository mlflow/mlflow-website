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
  EcosystemList,
} from "../components";
import GenAI from "@site/static/img/Home_page_hybrid/GenAI Apps & Agents.png";
import ModelTraining from "@site/static/img/Home_page_hybrid/Model Training.png";

export default function Home(): JSX.Element {
  return (
    <Layout>
      <AboveTheFold
        title="Deliver production-ready AI"
        body="The open source developer platform to build AI applications and models with confidence"
      >
        <GlossyCardContainer>
          <GlossyCard>
            <Card
              title="GenAI Apps & Agents"
              bodySize="m"
              body="Enhance your GenAI applications with end-to-end tracking, observability, and evaluations, all in one integrated platform."
              padded
              rounded={false}
              cta={{
                href: "/genai",
                text: "Learn more >",
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
                text: "Learn more >",
                prominent: true,
              }}
              image={
                <img src={ModelTraining} alt="" className="hidden md:block" />
              }
            />
          </GlossyCard>
        </GlossyCardContainer>
      </AboveTheFold>

      <Customers />
      <EcosystemList />

      <BelowTheFold>
        <LatestNews />
        <GetStartedTagline />
      </BelowTheFold>
    </Layout>
  );
}
