import Link from "@docusaurus/Link";
import { HeaderMenuItem } from "../HeaderMenuItem/HeaderMenuItem";
import { cva } from "class-variance-authority";
import {
  MLFLOW_ML_DOCS_URL,
  MLFLOW_GENAI_DOCS_URL,
  MLFLOW_3_DOCS_URL,
  MLFLOW_3_DEEP_LEARNING_DOCS_URL,
  MLFLOW_3_BREAKING_CHANGES_DOCS_URL,
  MLFLOW_3_FAQ_DOCS_URL,
  MLFLOW_3_GENAI_AGENT_DOCS_URL,
} from "@site/src/constants";

const wrapper = cva(
  "flex flex-col md:flex-row md:max-w-4xl mx-auto gap-6 md:gap-8 lg:gap-10 px-1 md:px-4 lg:pl-0 docs-submenu overflow-x-hidden",
);

const component = cva("flex flex-col gap-4");

const titleContainer = cva(
  "flex flex-col gap-1 md:gap-4 border-b border-[#F7F8F8]/8 pb-4",
);

const title = cva("text-white");

const subtitle = cva("text-[#F7F8F8]/60 m-0");

const feature = cva("flex flex-col gap-3");

const featureTitle = cva("text-[#F7F8F8]/60 text-sm");

const columns = cva("flex flex-row gap-6 xxs:gap-8");

const column = cva("min-w-30 xxs:min-w-40 flex flex-col md:gap-1");

export const HeaderDocsSubmenu = () => {
  return (
    <div className={wrapper()}>
      <div className={component()}>
        <div className={titleContainer()}>
          <Link to={MLFLOW_ML_DOCS_URL}>
            <h3 className={title()}>MLflow for Model Training</h3>
            <p className={subtitle()}>
              Get started with the core functionality for traditional machine
              learning workflows, hyperparameter tuning, and model lifecycle
              management.
            </p>
          </Link>
        </div>
        <div className={feature()}>
          <span className={featureTitle()}>Get Started</span>
          <div className={columns()}>
            <div className={column()}>
              <HeaderMenuItem href={MLFLOW_3_DOCS_URL} label="MLflow 3.0" />
              <HeaderMenuItem
                href={MLFLOW_3_DEEP_LEARNING_DOCS_URL}
                label="Deep Learning with MLflow 3"
              />
            </div>
            <div className={column()}>
              <HeaderMenuItem
                href={MLFLOW_3_BREAKING_CHANGES_DOCS_URL}
                label="Breaking Changes in MLflow 3"
              />
              <HeaderMenuItem href={MLFLOW_3_FAQ_DOCS_URL} label="FAQs" />
            </div>
          </div>
        </div>
      </div>
      <div className={component()}>
        <div className={titleContainer()}>
          <Link to={MLFLOW_GENAI_DOCS_URL}>
            <h3 className={title()}>MLflow for GenAI</h3>
            <p className={subtitle()}>
              Learn how to track, evaluate, and optimize your GenAI applications
              and agent workflows.
            </p>
          </Link>
        </div>
        <div className={feature()}>
          <span className={featureTitle()}>Get Started</span>
          <div className={columns()}>
            <div className={column()}>
              <HeaderMenuItem href={MLFLOW_3_DOCS_URL} label="MLflow 3.0" />
              <HeaderMenuItem
                href={MLFLOW_3_GENAI_AGENT_DOCS_URL}
                label="GenAI Agent with MLflow 3"
              />
            </div>
            <div className={column()}>
              <HeaderMenuItem
                href={MLFLOW_3_BREAKING_CHANGES_DOCS_URL}
                label="Breaking Changes in MLflow 3"
              />
              <HeaderMenuItem href={MLFLOW_3_FAQ_DOCS_URL} label="FAQs" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
