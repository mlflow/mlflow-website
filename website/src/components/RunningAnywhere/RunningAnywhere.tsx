import { Section } from "../Section/Section";
import { motion } from "motion/react";

const platforms = [
  {
    name: "Self-Hosted",
    logo: "/img/cloud.svg",
    href: "https://mlflow.org/docs/latest/tracking.html",
  },
  {
    name: "Databricks",
    logo: "/img/databricks-symbol-color.svg",
    href: "https://www.databricks.com/product/managed-mlflow",
    iconClass: "scale-75",
  },
  {
    name: "AWS SageMaker",
    logo: "/img/bedrock.png",
    href: "https://aws.amazon.com/jp/sagemaker/ai/experiments/",
  },
  {
    name: "Azure ML",
    logo: "/img/azureml.svg",
    href: "https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow",
  },
  {
    name: "Google Cloud",
    logo: "/img/vertexai.svg",
    href: "https://docs.cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments",
  },
  {
    name: "Nebius",
    logo: "/img/nebius-logo.jpeg",
    href: "https://nebius.com/docs/ml-platform/mlflow",
    iconClass: "scale-90",
  },
];

export const RunningAnywhere = () => {
  return (
    <Section
      title="Running Everywhere"
      body="Local development, on-premises clusters, cloud platforms, or managed services. Being open-source, MLflow is vendor-neutral and runs wherever you need it."
      align="center"
    >
      <motion.div
        className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-6 max-w-6xl mx-auto"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      >
        {platforms.map((platform, index) => (
          <motion.a
            key={platform.name}
            href={platform.href}
            target="_blank"
            rel="noreferrer noopener"
            className="group flex flex-col items-center gap-4 p-6 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 hover:border-white/20 transition-all"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
            whileHover={{ y: -4 }}
          >
            <div className="w-16 h-16 flex items-center justify-center">
              <img
                src={platform.logo}
                alt={platform.name}
                className={`max-w-full max-h-full object-contain ${platform.iconClass || ""}`}
              />
            </div>
            <span className="text-sm font-medium text-white/70 group-hover:text-white transition-colors">
              {platform.name}
            </span>
          </motion.a>
        ))}
      </motion.div>
    </Section>
  );
};
