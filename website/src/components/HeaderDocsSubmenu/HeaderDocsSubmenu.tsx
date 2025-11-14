import Link from "@docusaurus/Link";
import { cva } from "class-variance-authority";
import { MLFLOW_ML_DOCS_URL, MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";

const wrapper = cva(
  "flex flex-col md:flex-row md:max-w-2xl mx-auto gap-6 md:gap-8 lg:gap-10 px-1 md:px-4 lg:pl-0 docs-submenu overflow-x-hidden",
);

const linkItem = cva(
  "flex flex-col gap-1 md:gap-2 border-b border-[#F7F8F8]/8 pb-4 last:border-b-0",
);

const linkTitle = cva("text-white text-lg");

const linkSubtitle = cva("text-[#F7F8F8]/60 text-sm m-0");

export const HeaderDocsSubmenu = () => {
  return (
    <div className={wrapper()}>
      <div className={linkItem()}>
        <Link to={MLFLOW_ML_DOCS_URL}>
          <h4 className={linkTitle()}>ML</h4>
          <p className={linkSubtitle()}>Traditional ML documentation</p>
        </Link>
      </div>
      <div className={linkItem()}>
        <Link to={MLFLOW_GENAI_DOCS_URL}>
          <h4 className={linkTitle()}>GenAI</h4>
          <p className={linkSubtitle()}>Generative AI documentation</p>
        </Link>
      </div>
    </div>
  );
};
