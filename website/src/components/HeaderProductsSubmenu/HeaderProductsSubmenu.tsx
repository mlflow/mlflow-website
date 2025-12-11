import Link from "@docusaurus/Link";
import { HeaderMenuItem } from "../HeaderMenuItem/HeaderMenuItem";
import { cva } from "class-variance-authority";

const wrapper = cva(
  "flex flex-col md:flex-row md:max-w-4xl mx-auto gap-6 md:gap-8 lg:gap-10 px-1 md:px-4 lg:pl-0 products-submenu overflow-x-hidden",
);

const component = cva("flex flex-col gap-4");

const titleContainer = cva(
  "flex flex-col gap-1 md:gap-4 border-b border-[#F7F8F8]/8 pb-4",
);

const title = cva(
  "text-white transition-colors duration-200 hover:text-white/80",
);

const subtitle = cva("text-[#F7F8F8]/60 m-0");

const feature = cva("flex flex-col gap-3");

const featureTitle = cva("text-[#F7F8F8]/60 text-sm");

const columns = cva("flex flex-row gap-6 xxs:gap-8");

const column = cva("min-w-30 xxs:min-w-40 flex flex-col md:gap-1");

export const HeaderProductsSubmenu = () => {
  return (
    <div className={wrapper()}>
      <div className={component()}>
        <div className={titleContainer()}>
          <Link to="/genai">
            <h3 className={title()}>Gen AI</h3>
            <p className={subtitle()}>Ship high-quality GenAI, fast</p>
          </Link>
        </div>
        <div className={feature()}>
          <span className={featureTitle()}>Features</span>
          <div className={columns()}>
            <div className={column()}>
              <HeaderMenuItem
                href="/genai/observability"
                label="Observability"
              />
              <HeaderMenuItem href="/genai/evaluations" label="Evaluations" />
              <HeaderMenuItem
                href="/genai/prompt-registry"
                label="Prompt Registry"
              />
            </div>
            <div className={column()}>
              <HeaderMenuItem
                href="/genai/app-versioning"
                label="App versioning"
              />
              <HeaderMenuItem href="/genai/ai-gateway" label="AI Gateway" />
            </div>
          </div>
        </div>
      </div>
      <div className={component()}>
        <div className={titleContainer()}>
          <Link to="/classical-ml">
            <h3 className={title()}>Model training</h3>
            <p className={subtitle()}>Mastering the ML lifecycle</p>
          </Link>
        </div>
        <div className={feature()}>
          <span className={featureTitle()}>Features</span>
          <div className={columns()}>
            <div className={column()}>
              <HeaderMenuItem
                href="/classical-ml/experiment-tracking"
                label="Experiment tracking"
              />
              <HeaderMenuItem
                href="/classical-ml/model-evaluation"
                label="Model evaluation"
              />
            </div>
            <div className={column()}>
              <HeaderMenuItem
                href="/classical-ml/models"
                label="MLflow models"
              />
              <HeaderMenuItem
                href="/classical-ml/model-registry"
                label="Model Registry & deployment"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
