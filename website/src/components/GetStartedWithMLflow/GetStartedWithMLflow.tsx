import DatabricksLogo from "@site/static/img/databricks-logo.svg";
import Checkmark from "@site/static/img/checkmark.svg";

import { Button } from "../Button/Button";

export const GetStartedWithMLflow = () => {
  return (
    <div className="flex flex-col lg:flex-row w-full justify-between gap-6">
      <div className="flex flex-col gap-6 w-full lg:w-1/2">
        <h1>Get started with MLflow</h1>
        <span className="text-white/60 font-light text-lg">
          Choose from two options depending on your needs
        </span>
      </div>
      <div className="flex flex-col w-full lg:w-1/2 gap-6">
        <div className="flex flex-col gap-8 p-8 bg-[#fff]/4 rounded-2xl">
          <div className="flex flex-row justify-between items-center gap-4">
            <div className="flex flex-row justify-center items-end gap-3 flex-wrap">
              <h3 className="m-0">Managed </h3>
              <span className="text-white/50 text-sm">WITH</span>
              <DatabricksLogo />
            </div>
            <div className="hidden lg:block rounded-full uppercase px-4 py-2 bg-[#EB1700] text-xs font-semibold whitespace-nowrap">
              <span>MOST POPULAR</span>
            </div>
          </div>
          <div className="flex flex-col gap-4">
            <div className="flex flex-row items-center gap-4">
              <Checkmark />
              <span className="text-md font-light text-white/60">
                Production-ready
              </span>
            </div>
            <div className="flex flex-row items-center gap-4">
              <Checkmark />
              <span className="text-md font-light text-white/60">
                Secure & scalable
              </span>
            </div>
            <div className="flex flex-row items-center gap-4">
              <Checkmark />
              <span className="text-md font-light text-white/60">
                24/7 support
              </span>
            </div>
          </div>
          <Button size="large" width="full">
            Get started
          </Button>
        </div>
        <div className="flex flex-col gap-8 p-8 bg-[#fff]/4 rounded-2xl">
          <h3 className="m-0">Self-Hosting</h3>
          <Button size="large" width="full" variant="dark">
            Get started
          </Button>
        </div>
      </div>
    </div>
  );
};
