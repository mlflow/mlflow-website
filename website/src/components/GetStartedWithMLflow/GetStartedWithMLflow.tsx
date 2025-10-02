import DatabricksLogo from "@site/static/img/databricks-logo.svg";
import Checkmark from "@site/static/img/checkmark.svg";
import { useLocation } from "@docusaurus/router";
import useBaseUrl from "@docusaurus/useBaseUrl";
import useBrokenLinks from "@docusaurus/useBrokenLinks";

import { GetStartedButton } from "../GetStartedButton/GetStartedButton";
import { cn, isClassicalMLPage } from "../../utils";
import { Heading } from "../Typography/Heading";
import { Body } from "../Typography/Body";
import { useLayoutVariant } from "../Layout/Layout";
import {
  MLFLOW_DOCS_URL,
  MLFLOW_GENAI_DOCS_URL,
  MLFLOW_DBX_TRIAL_URL,
  MLFLOW_DBX_INSTALL_URL,
} from "@site/src/constants";

type ContentType = "genai" | "classical-ml";

interface GetStartedWithMLflowProps {
  contentType?: ContentType;
}

export const GetStartedWithMLflow = ({ contentType }: GetStartedWithMLflowProps = {}) => {
  const variant = useLayoutVariant();
  const location = useLocation();
  const classicalMLPath = useBaseUrl("/classical-ml");
  const isClassicalML = isClassicalMLPage(location.pathname, classicalMLPath);
  const databricksUrl = isClassicalML
    ? MLFLOW_DBX_INSTALL_URL
    : MLFLOW_DBX_TRIAL_URL;
  const databricksButtonText = isClassicalML
    ? "Get started"
    : "Get started for free";

  // Determine the appropriate docs URL based on content type
  const getDocsUrl = () => {
    if (contentType === "genai") {
      return MLFLOW_GENAI_DOCS_URL;
    }
    return MLFLOW_DOCS_URL;
  };

  useBrokenLinks().collectAnchor("get-started");

  return (
    <div
      id="get-started"
      className={cn("grid grid-cols-1 lg:grid-cols-2 gap-8")}
    >
      <div className="flex flex-col gap-6 items-start lg:col-span-2">
        <Heading level={2}>Get started with MLflow</Heading>
        <Body size="l">Choose from two options depending on your needs</Body>
      </div>
      <div className="flex flex-col gap-8 p-8 bg-[#fff]/4 rounded-2xl justify-between">
        <div className="flex flex-col gap-8">
          <h3 className="m-0 text-white">Self-hosted Open Source</h3>
          <div className="flex flex-col gap-4">
            {[
              "Apache-2.0 license",
              "Full control over your own infrastructure",
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
          variant="primary"
          link={getDocsUrl()}
        />
      </div>
      <div className="flex flex-col gap-8 p-8 bg-[#fff]/4 rounded-2xl @container justify-between">
        <div className="flex flex-col gap-8">
          <div className="flex flex-row justify-between items-center gap-4">
            <div className="flex flex-row justify-center items-end gap-3 flex-wrap">
              <h3 className="m-0 text-white">Managed hosting </h3>
              <span className="text-gray-500 text-sm">ON</span>
              <DatabricksLogo />
            </div>
          </div>
          <div className="flex flex-col gap-4">
            {[
              "Free and fully managed — experience MLflow without the setup hassle",
              "Built and maintained by the original creators of MLflow",
              "Full OSS compatibility",
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
          body={databricksButtonText}
          size="large"
          width="full"
          variant="dark"
          link={databricksUrl}
        />
      </div>
    </div>
  );
};
