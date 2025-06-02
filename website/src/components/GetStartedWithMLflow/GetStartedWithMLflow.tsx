import DatabricksLogo from "@site/static/img/databricks-logo.svg";
import Checkmark from "@site/static/img/checkmark.svg";

import { GetStartedButton } from "../GetStartedButton/GetStartedButton";
import { cn } from "../../utils";
import { Heading } from "../Typography/Heading";
import { Body } from "../Typography/Body";
import { useLayoutVariant } from "../Layout/Layout";

export const GetStartedWithMLflow = () => {
  const variant = useLayoutVariant();
  return (
    <div className="flex flex-col lg:flex-row w-full justify-between gap-6">
      <div className="flex flex-col gap-6 w-full lg:w-1/2 items-start">
        <Heading level={2}>Get started with MLflow</Heading>
        <Body size="l">Choose from two options depending on your needs</Body>
      </div>
      <div className="flex flex-col w-full lg:w-1/2 gap-6">
        <div className="flex flex-col gap-8 p-8 bg-[#fff]/4 rounded-2xl @container">
          <div className="flex flex-row justify-between items-center gap-4">
            <div className="flex flex-row justify-center items-end gap-3 flex-wrap">
              <h3 className="m-0 text-white">Managed </h3>
              <span className="text-gray-500 text-sm">WITH</span>
              <DatabricksLogo />
            </div>
            <div
              className={cn(
                "hidden @md:block rounded-full uppercase px-4 py-2 text-xs font-semibold whitespace-nowrap",
                variant === "blue"
                  ? "bg-brand-teal text-black"
                  : "bg-brand-red text-white",
              )}
            >
              <span>MOST POPULAR</span>
            </div>
          </div>
          <div className="flex flex-col gap-4">
            {[
              "Access to all platform features",
              "Unlimited users",
              "Unlimited data access",
              "No charge up to 50K traces, covered by free credits upon signup",
              "Pay-as-you-go billing with credit card",
              "Enterprise support available",
            ].map((bulletPoint, index) => (
              <div key={index} className="flex flex-row items-center gap-4">
                <Checkmark className="shrink-0" />
                <span className="text-md font-light text-gray-600">
                  {bulletPoint}
                </span>
              </div>
            ))}
          </div>
          <GetStartedButton
            size="large"
            width="full"
            variant={variant === "blue" ? "blue" : "primary"}
          />
        </div>
        <div className="flex flex-col gap-8 p-8 bg-[#fff]/4 rounded-2xl">
          <h3 className="m-0 text-white">Self-Hosting</h3>
          <GetStartedButton
            size="large"
            width="full"
            variant="dark"
            link="http://mlflow.org/docs/latest/getting-started/intro-quickstart/"
          />
        </div>
      </div>
    </div>
  );
};
