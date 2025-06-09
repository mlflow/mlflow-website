import { Section } from "..";
import BrowserOnly from "@docusaurus/BrowserOnly";
import ExpandableGrid from "../ExpandableGrid/ExpandableGrid";
import MiniLogoCard from "../MiniLogoCard/MiniLogoCard";

export const EcosystemList = () => {
  return (
    <div className="w-full px-4 md:px-8 lg:px-16">
      <Section title="Integrates with 25+ apps and frameworks">
        <BrowserOnly>
          {() => (
            <ExpandableGrid
              items={[
                {
                  title: "PyTorch",
                  src: "img/pytorch.svg",
                  href: "https://mlflow.org/docs/latest/ml/deep-learning/pytorch/index.html",
                },
                {
                  title: "OpenAI",
                  src: "img/openai.svg",
                  href: "https://mlflow.org/docs/latest/genai/flavors/openai/index.html",
                },
                {
                  title: "HuggingFace",
                  src: "img/huggingface.svg",
                  href: "https://mlflow.org/docs/latest/ml/deep-learning/transformers/index.html",
                },
                {
                  title: "LangChain",
                  src: "img/langchain.svg",
                  href: "https://mlflow.org/docs/latest/genai/flavors/langchain/index.html",
                },
                {
                  title: "Anthropic",
                  src: "img/anthropic.svg",
                  href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/anthropic.html",
                },
                {
                  title: "Gemini",
                  src: "img/google-gemini.svg",
                  href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/gemini.html",
                },
                {
                  title: "AutoGen",
                  src: "img/autogen.jpeg",
                  href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/autogen.html",
                },
                {
                  title: "LlamaIndex",
                  src: "img/llamaindex.svg",
                  href: "https://mlflow.org/docs/latest/genai/flavors/llama-index/index.html",
                },
                {
                  title: "Spark",
                  src: "img/spark.svg",
                  href: "https://mlflow.org/docs/latest/ml/traditional-ml/sparkml/index.html",
                },
                {
                  title: "Keras",
                  src: "img/keras.svg",
                  href: "https://mlflow.org/docs/latest/ml/deep-learning/keras/index.html",
                },
                {
                  title: "TensorFlow",
                  src: "img/tensorflow.svg",
                  href: "https://mlflow.org/docs/latest/ml/deep-learning/tensorflow/index.html",
                },
                {
                  title: "scikit-learn",
                  src: "img/scikit-learn.svg",
                  href: "https://mlflow.org/docs/latest/ml/traditional-ml/sklearn/index.html",
                },
                {
                  title: "XGBoost",
                  src: "img/xgboost.svg",
                  href: "https://mlflow.org/docs/latest/ml/traditional-ml/xgboost/index.html",
                },
                {
                  title: "LightGBM",
                  src: "img/lightgbm.svg",
                  href: "https://mlflow.org/docs/latest/models.html#lightgbm-lightgbm",
                },
                {
                  title: "CatBoost",
                  src: "img/catboost.svg",
                  href: "https://mlflow.org/docs/latest/ml/model/index.html#catboost-catboost",
                },
                {
                  title: "CrewAI",
                  src: "img/crewai.svg",
                  href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/crewai.html",
                },

                {
                  title: "LiteLLM",
                  src: "img/litellm.png",
                  href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/litellm.html",
                },
                {
                  title: "Sentence Transformers",
                  src: "img/sentence-transformers.svg",
                  href: "https://mlflow.org/docs/latest/ml/deep-learning/sentence-transformers/index.html",
                },
                {
                  title: "ONNX",
                  src: "img/onnx.svg",
                  href: "https://mlflow.org/docs/latest/ml/model/inex.html#onnx-onnx",
                },
                {
                  title: "Spacy",
                  src: "img/spacy.svg",
                  href: "https://mlflow.org/docs/latest/ml/deep-learning/spacy/index.html",
                },
                {
                  title: "FastAI",
                  src: "img/fastai.png",
                  href: "https://mlflow.org/docs/latest/ml/model/index.html#fastai-fastai",
                },
                {
                  title: "StatsModels",
                  src: "img/statsmodels.svg",
                  href: "https://mlflow.org/docs/latest/ml/model/index.html#statsmodels-statsmodels",
                },
                {
                  title: "Prompt flow",
                  src: "img/promptflow.svg",
                  href: "https://mlflow.org/docs/latest/ml/model/index.html#promptflow-promptflow-experimental",
                },
                {
                  title: "JohnSnowLabs",
                  src: "img/johnsnowlab.svg",
                  href: "https://mlflow.org/docs/latest/ml/model/index.html#john-snow-labs-johnsnowlabs-experimental",
                },
                {
                  title: "H2O",
                  src: "img/h2o.svg",
                  href: "https://mlflow.org/docs/latest/ml/model/index.html#h2o-h2o",
                },
                {
                  title: "Prophet",
                  src: "img/prophet.svg",
                  href: "https://mlflow.org/docs/latest/ml/traditional-ml/prophet/index.html",
                },
              ]}
              defaultVisibleCount={window.innerWidth > 996 ? 16 : 8}
              renderItem={({ title, src, href }) => (
                <MiniLogoCard title={title} src={src} href={href} />
              )}
            />
          )}
        </BrowserOnly>
      </Section>
    </div>
  );
};
