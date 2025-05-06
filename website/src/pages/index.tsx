import useDocusaurusContext from "@docusaurus/useDocusaurusContext";

import { SocialWidget, LatestNews } from "../components";
import { GetStartedWithMLflow } from "../components";

export default function Home(): JSX.Element {
  const { siteConfig } = useDocusaurusContext();

  return (
    <div className="flex flex-col items-center justify-center bg-[#0E1416] gap-8 min-h-screen">
      <div className="flex flex-col w-full px-20 mb-10 gap-10">
        <SocialWidget variant="red" />
        <LatestNews variant="red" />
      </div>
    </div>
  );
}
