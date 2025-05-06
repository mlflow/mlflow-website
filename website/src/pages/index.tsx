import useDocusaurusContext from "@docusaurus/useDocusaurusContext";

import { SocialWidget, LatestNews, Header } from "../components";

export default function Home(): JSX.Element {
  const { siteConfig } = useDocusaurusContext();

  return (
    <div>
      <div className="flex flex-col items-center justify-center bg-[#0E1416] gap-8 min-h-screen">
        <Header />
        <div className="flex flex-col w-full px-20 mb-10 gap-10 mt-20">
          <SocialWidget variant="red" />
          <LatestNews variant="red" />
        </div>
      </div>
    </div>
  );
}
