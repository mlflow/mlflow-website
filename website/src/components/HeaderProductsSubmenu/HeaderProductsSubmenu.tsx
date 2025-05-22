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
                href="/genai/prompt-registry-versioning"
                label="Prompt registry & versioning"
              />
            </div>
            <div className="min-w-40 flex flex-col gap-4 md:gap-1">
              <HeaderMenuItem
                href="/genai/human-feedback"
                label="Human feedback"
              />
              <HeaderMenuItem href="/genai/governance" label="Governance" />
              <HeaderMenuItem href="/genai/ai-gateway" label="AI gateway" />
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
              <HeaderMenuItem href="/classical-ml/tracking" label="Tracking" />
              <HeaderMenuItem
                href="/classical-ml/hyperparam-tuning"
                label="Hyperparameter tuning"
              />
              <HeaderMenuItem href="/classical-ml/models" label="Models" />
            </div>
            <div className="min-w-40 flex flex-col gap-4 md:gap-1">
              <HeaderMenuItem
                href="/classical-ml/unified-registry"
                label="Unified registry"
              />
              <HeaderMenuItem href="/classical-ml/serving" label="Serving" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
