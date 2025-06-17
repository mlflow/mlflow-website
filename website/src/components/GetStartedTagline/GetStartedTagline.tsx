import { MLFLOW_DOCS_URL } from "@site/src/constants";
import { GetStartedButton } from "../GetStartedButton/GetStartedButton";

export const GetStartedTagline = () => {
  return (
    <div className="flex flex-col md:flex-row justify-center items-center gap-6 md:gap-16">
      <span className="text-lg text-gray-600 text-center">
        Join the industry leading companies building with MLflow
      </span>
      <GetStartedButton variant="primary" link="#get-started" />
    </div>
  );
};
