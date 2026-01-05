import { Section } from "../Section/Section";
import MiniLogoCard from "../MiniLogoCard/MiniLogoCard";

const platforms = [
  {
    title: "Self-Hosted",
    src: "/img/cloud.svg",
    href: "https://mlflow.org/docs/latest/tracking.html",
  },
  {
    title: "Databricks",
    src: "/img/databricks-symbol-color.svg",
    href: "https://www.databricks.com/product/managed-mlflow",
  },
  {
    title: "AWS SageMaker",
    src: "/img/bedrock.png",
    href: "https://aws.amazon.com/jp/sagemaker/ai/experiments/",
  },
  {
    title: "Azure ML",
    src: "/img/azureml.svg",
    href: "https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow",
  },
  {
    title: "Google Cloud",
    src: "/img/vertexai.svg",
    href: "https://docs.cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments",
  },
  {
    title: "Nebius",
    src: "/img/nebius-logo.jpeg",
    href: "https://nebius.com/docs/ml-platform/mlflow",
  },
];

export const RunningAnywhere = () => {
  return (
    <Section
      title="Running Everywhere"
      body="Local development, on-premises clusters, cloud platforms, or managed services. Being open-source, MLflow is vendor-neutral and runs wherever you need it."
      align="center"
    >
      <div
        className="grid grid-cols-2 md:grid-cols-6 gap-6 justify-items-center mx-auto"
        style={{ maxWidth: 1200 }}
      >
        {platforms.map((platform) => (
          <div key={platform.title} style={{ width: 160, height: 160 }}>
            <MiniLogoCard
              title={platform.title}
              src={platform.src}
              href={platform.href}
            />
          </div>
        ))}
      </div>
    </Section>
  );
};
