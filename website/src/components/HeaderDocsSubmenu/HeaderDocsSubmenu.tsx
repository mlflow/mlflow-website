import Link from "@docusaurus/Link";
import { cva } from "class-variance-authority";
import { MLFLOW_ML_DOCS_URL, MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";

const wrapper = cva(
  "flex flex-col md:flex-row md:max-w-2xl mx-auto gap-6 md:gap-8 lg:gap-10 px-1 md:px-4 lg:pl-0 docs-submenu overflow-x-hidden",
);

const linkItem = cva("flex flex-col gap-1 md:gap-2 pb-4 flex-1 group");

const linkTitle = cva(
  "text-white transition-colors duration-200 group-hover:!text-white/60",
);

const linkSubtitle = cva("text-[#F7F8F8]/60 m-0");

export const HeaderDocsSubmenu = () => {
  return (
    <div className={wrapper()}>
      <div className={linkItem()}>
        <Link to={MLFLOW_GENAI_DOCS_URL}>
          <h3 className={linkTitle()}>LLMs & Agents</h3>
          <p className={linkSubtitle()}>
            Debug, evaluate, monitor, and optimize your AI agents and LLM
            applications, with production-grade tracing, evaluation, prompt
            management, and much more.
          </p>
        </Link>
      </div>
      <div className={linkItem()}>
        <Link to={MLFLOW_ML_DOCS_URL}>
          <h3 className={linkTitle()}>Model Training</h3>
          <p className={linkSubtitle()}>
            Manage the full machine learning and deep learning model lifecycle,
            with experiment tracking, hyperparameter tuning, and beyond.
          </p>
        </Link>
      </div>
    </div>
  );
};
