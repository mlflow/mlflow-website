import DatabricksLogo from "@site/static/img/databricks-logo.svg";
import Checkmark from "@site/static/img/checkmark.svg";

import { GetStartedButton } from "../GetStartedButton/GetStartedButton";
import { cn } from "../../utils";
import { Heading } from "../Typography/Heading";
import { Body } from "../Typography/Body";
import { useLayoutVariant } from "../Layout/Layout";
import { MLFLOW_GET_STARTED_URL } from "@site/src/constants";

export const GetStartedWithMLflow = () => {
  const variant = useLayoutVariant();

  return (
    <div className={cn("grid grid-cols-1 lg:grid-cols-2 gap-8")}>
      <div
        className={cn(
          "flex flex-col gap-6 items-start",
          variant !== "blue" ? "lg:col-span-2" : "",
        )}
      >
        <Heading level={2}>Get started with MLflow</Heading>
        {variant !== "blue" ? (
          <Body size="l">Choose from two options depending on your needs</Body>
        ) : null}
      </div>
      {variant !== "blue" ? (
        <div className="flex flex-col gap-8 p-8 bg-[#fff]/4 rounded-2xl @container justify-between">
          <div className="flex flex-col gap-8">
            <div className="flex flex-row justify-between items-center gap-4">
              <div className="flex flex-row justify-center items-end gap-3 flex-wrap">
                <h3 className="m-0 text-white">Managed </h3>
                <span className="text-gray-500 text-sm">WITH</span>
                <DatabricksLogo />
              </div>
              <div className="hidden @md:block rounded-full uppercase px-4 py-2 text-xs font-semibold whitespace-nowrap bg-brand-red text-white">
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
          </div>
          <GetStartedButton size="large" width="full" variant="primary" />
        </div>
      ) : null}
      <div className="flex flex-col gap-8 p-8 bg-[#fff]/4 rounded-2xl justify-between">
        <div className="flex flex-col gap-8">
          <h3 className="m-0 text-white">Self-Hosted Open Source</h3>
          <div className="flex flex-col gap-4">
            {[
              "Apache-2.0 license",
              "Access to all core platform features",
              "Full control over your own infrastructure",
              "Ability to customize MLflow to fit your specific needs",
              "Community support",
            ].map((bulletPoint, index) => (
              <div key={index} className="flex flex-row items-center gap-4">
                <Checkmark className="shrink-0" />
                <span className="text-md font-light text-gray-600">
                  {bulletPoint}
                </span>
              </div>
            ))}
          </div>
        </div>
        <GetStartedButton
          size="large"
          width="full"
          variant={variant === "blue" ? "blue" : "dark"}
          link={MLFLOW_GET_STARTED_URL}
        />
      </div>
    </div>
  );
};
