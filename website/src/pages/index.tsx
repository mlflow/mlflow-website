import useDocusaurusContext from "@docusaurus/useDocusaurusContext";

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
} from "../components";

export default function Home(): JSX.Element {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout variant="colorful">
      <div
        className="flex flex-col bg-[linear-gradient(to_bottom,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-1.png')]
 bg-top bg-no-repeat bg-cover w-full pt-42 pb-20 py-20"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <h1 className="text-center text-wrap">GenAI and ML made simple</h1>
            <p className="text-center text-wrap text-lg">
              The AI developer platform to build AI applications and models with
              confidence
            </p>
          </div>
          <div className="flex flex-col md:flex-row gap-10">
            <GlossyCard image={null}>
              <h3>GenAI Apps & Agents</h3>
              <p>
                Enhance your GenAI applications with end-to-end observability,
                monitoring, and enterprise governance, all in one integrated
                platform.
              </p>
              <Button variant="primary">Learn more &gt;</Button>
            </GlossyCard>
            <GlossyCard image={null}>
              <h3>Model Training</h3>
              <p>
                Streamline your machine learning workflows with enterprise-grade
                tracking, model management, and deployment.
              </p>
              <Button variant="primary">Learn more &gt;</Button>
            </GlossyCard>
          </div>
        </div>
      </div>
      <div className="flex flex-col px-6 md:px-20 gap-40 mt-20">
        <Testimonials />
        <LogosCarousel>
          <img
            className="mx-4 inline h-16"
            src="img/companies/databricks.svg"
          />
          <img className="mx-4 inline h-16" src="img/companies/microsoft.svg" />
          <img className="mx-4 inline h-16" src="img/companies/meta.svg" />
          <img className="mx-4 inline h-16" src="img/companies/mosaicml.svg" />
          <img className="mx-4 inline h-16" src="img/companies/zillow.svg" />
          <img className="mx-4 inline h-16" src="img/companies/toyota.svg" />
          <img className="mx-4 inline h-16" src="img/companies/booking.svg" />
          <img className="mx-4 inline h-16" src="img/companies/wix.svg" />
          <img className="mx-4 inline h-16" src="img/companies/accenture.svg" />
          <img className="mx-4 inline h-16" src="img/companies/asml.svg" />
        </LogosCarousel>
        <GetStartedWithMLflow />
        <LatestNews variant="red" />
        <GetStartedTagline />
        <SocialWidget variant="red" />
      </div>
    </Layout>
  );
}
