import Link from "@docusaurus/Link";

import {
  LatestNews,
  Layout,
  GlossyCard,
  Button,
  GetStartedTagline,
  Testimonials,
  LogosCarousel,
  Body,
  AboveTheFold,
  BelowTheFold,
} from "../components";

export default function Home(): JSX.Element {
  return (
    <Layout variant="colorful">
      <AboveTheFold
        title="AI and ML made simple"
        body="The AI developer platform to build AI applications and models with confidence"
      >
        <div className="flex flex-col md:flex-row gap-10">
          <GlossyCard image={null}>
            <h3 className="text-white">GenAI Apps & Agents</h3>
            <Body size="m">
              Enhance your GenAI applications with end-to-end observability,
              monitoring, and enterprise governance, all in one integrated
              platform.
            </Body>
            <Link href="/genai">
              <Button variant="primary">Learn more &gt;</Button>
            </Link>
          </GlossyCard>
          <GlossyCard image={null}>
            <h3 className="text-white">Model Training</h3>
            <Body size="m">
              Streamline your machine learning workflows with enterprise-grade
              tracking, model management, and deployment.
            </Body>
            <Link href="/classical-ml">
              <Button variant="primary">Learn more &gt;</Button>
            </Link>
          </GlossyCard>
        </div>
      </AboveTheFold>

      <Testimonials />

      <LogosCarousel />

      <BelowTheFold>
        <LatestNews />
        <GetStartedTagline />
      </BelowTheFold>
    </Layout>
  );
}
