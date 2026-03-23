import Link from "@docusaurus/Link";
import { cva } from "class-variance-authority";

const wrapper = cva(
  "flex flex-col md:flex-row md:max-w-2xl mx-auto gap-6 md:gap-8 lg:gap-10 px-1 md:px-4 lg:pl-0 overflow-x-hidden",
);

const linkItem = cva("flex flex-col gap-1 md:gap-2 pb-4 flex-1 group");

const linkTitle = cva(
  "text-white transition-colors duration-200 group-hover:!text-white/60",
);

const linkSubtitle = cva("text-[#F7F8F8]/60 m-0");

export const HeaderResourcesSubmenu = () => {
  return (
    <div className={wrapper()}>
      <div className={linkItem()}>
        <Link to="/cookbook">
          <h3 className={linkTitle()}>Cookbook</h3>
          <p className={linkSubtitle()}>
            Hands-on guides and code examples for building GenAI applications
            with MLflow.
          </p>
        </Link>
      </div>
      <div className={linkItem()}>
        <Link to="/ambassadors">
          <h3 className={linkTitle()}>Ambassador Program</h3>
          <p className={linkSubtitle()}>
            Join the MLflow community as an ambassador and help shape the future
            of ML tooling.
          </p>
        </Link>
      </div>
    </div>
  );
};
