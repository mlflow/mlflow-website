import Link from "@docusaurus/Link";

import {
  SocialWidget,
  LatestNews,
  Layout,
  GetStartedWithMLflow,
  GlossyCard,
  Button,
  GetStartedTagline,
  Testimonials,
  LogosCarousel,
  Heading,
  Body,
} from "../components";

export default function Home(): JSX.Element {
  return (
    <Layout variant="colorful">
      <div
        className="flex flex-col bg-[linear-gradient(to_bottom,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-1.png')]
 bg-top bg-no-repeat bg-cover w-full pt-42 pb-20 py-20"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-container">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <Heading level={1}>AI and ML made simple</Heading>
            <Body size="l">
              The AI developer platform to build AI applications and models with
              confidence
            </Body>
          </div>
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
        </div>
      </div>
      <div className="flex flex-col px-6 md:px-20 gap-40 mt-20 max-w-container">
        <div className="flex flex-col gap-16">
          <Testimonials />
          <LogosCarousel />
        </div>
        <GetStartedWithMLflow />
        <LatestNews variant="red" />
        <GetStartedTagline />
        <SocialWidget variant="red" />
      </div>
    </Layout>
  );
}
