import Link from "@docusaurus/Link";
import { HeaderMenuItem } from "../HeaderMenuItem/HeaderMenuItem";

export const HeaderProductsSubmenu = () => {
  return (
    <div className="flex flex-col md:flex-row md:max-w-4xl mx-auto gap-6 md:gap-8 lg:gap-10 px-4 lg:pl-0 products-submenu">
      <div className="flex flex-col gap-1 md:gap-4">
        <div className="flex flex-col gap-1 md:gap-4 md:border-b border-[#F7F8F8]/8 pb-4">
          <Link to="/genai">
            <h3 className="text-white">Gen AI</h3>
            <p className="text-[#F7F8F8]/60 m-0">
              Ship high-quality GenAI, fast
            </p>
          </Link>
        </div>
        <div className="flex flex-col gap-3">
          <span className="text-[#F7F8F8]/60 text-sm">Features</span>
          <div className="flex flex-row gap-4">
            <div className="min-w-50 flex flex-col gap-4 md:gap-1">
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
            <div className="min-w-40 flex flex-col gap-4 md:gap-1">
              <HeaderMenuItem
                href="/genai/app-versioning"
                label="App versioning"
              />
              <HeaderMenuItem href="/genai/ai-gateway" label="AI Gateway" />
            </div>
          </div>
        </div>
      </div>
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-1 md:gap-4 md:border-b border-[#F7F8F8]/8 pb-4">
          <Link to="/classical-ml">
            <h3 className="text-white">Model training</h3>
            <p className="text-[#F7F8F8]/60">Mastering the ML lifecycle</p>
          </Link>
        </div>
        <div className="flex flex-col gap-3">
          <span className="text-[#F7F8F8]/60 text-sm">Features</span>
          <div className="flex flex-row gap-8">
            <div className="min-w-40 flex flex-col gap-4 md:gap-1">
              <HeaderMenuItem
                href="/classical-ml/experiment-tracking"
                label="Experiment tracking"
              />
              <HeaderMenuItem
                href="/classical-ml/hyperparam-tuning"
                label="Hyperparameter tuning"
              />
            </div>
            <div className="min-w-40 flex flex-col gap-4 md:gap-1">
              <HeaderMenuItem
                href="/classical-ml/model-registry"
                label="Model Registry & deployment"
              />
              <HeaderMenuItem
                href="/classical-ml/serving"
                label="Model evaluation"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
