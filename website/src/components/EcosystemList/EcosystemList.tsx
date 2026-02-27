import { Section } from "..";
import BrowserOnly from "@docusaurus/BrowserOnly";
import ExpandableGrid from "../ExpandableGrid/ExpandableGrid";
import MiniLogoCard from "../MiniLogoCard/MiniLogoCard";

export const EcosystemList = () => {
  return (
    <div className="w-full px-4 md:px-8 lg:px-16 flex justify-center relative overflow-hidden">
      <Section
        title="Works With Any Framework"
        body="From LLM agent frameworks to traditional ML libraries - MLflow integrates seamlessly with 100+ tools across the AI ecosystem. Supports Python, TypeScript/JavaScript, Java, R, and natively integrates with OpenTelemetry."
        align="center"
        ambient
      >
        <div style={{ maxWidth: 1000 }} className="w-full">
          <BrowserOnly>
            {() => (
              <ExpandableGrid
                items={[
                  {
                    title: "OpenAI",
                    src: "img/openai.svg",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/openai.html",
                  },
                  {
                    title: "Anthropic",
                    src: "img/anthropic.svg",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/anthropic.html",
                  },
                  {
                    title: "LangChain / LangGraph",
                    src: "img/langchain.svg",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langchain.html",
                  },
                  {
                    title: "Vercel AI",
                    src: "img/vercel.svg",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/vercelai.html",
                  },
                  {
                    title: "Amazon Bedrock",
                    src: "img/bedrock.png",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/bedrock.html",
                  },
                  {
                    title: "LiteLLM",
                    src: "img/litellm.png",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/litellm.html",
                  },
                  {
                    title: "Gemini",
                    src: "img/google-gemini.svg",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/gemini.html",
                  },
                  {
                    title: "ADK",
                    src: "img/google-adk.png",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/gemini.html",
                  },
                  {
                    title: "Strands Agent",
                    src: "img/strands-agents.svg",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/strands.html",
                  },
                  {
                    title: "DSPy",
                    src: "img/dspy.png",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/dspy.html",
                  },
                  {
                    title: "PydanticAI",
                    src: "img/pydantic-ai.png",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/pydantic_ai.html",
                  },
                  {
                    title: "LlamaIndex",
                    src: "img/llamaindex.svg",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/llamaindex.html",
                  },
                  {
                    title: "Agno",
                    src: "img/agno.jpeg",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/agno.html",
                  },
                  {
                    title: "Semantic Kernel",
                    src: "img/semantic-kernel.png",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/semantic_kernel.html",
                  },
                  {
                    title: "AutoGen",
                    src: "img/autogen.jpeg",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/autogen.html",
                  },
                  {
                    title: "CrewAI",
                    src: "img/crewai.svg",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/crewai.html",
                  },

                  {
                    title: "PyTorch",
                    src: "img/pytorch.svg",
                    href: "https://mlflow.org/docs/latest/ml/deep-learning/pytorch/index.html",
                  },
                  {
                    title: "HuggingFace",
                    src: "img/huggingface.svg",
                    href: "https://mlflow.org/docs/latest/ml/deep-learning/transformers/index.html",
                  },
                  {
                    title: "Ollama",
                    src: "img/ollama.png",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/ollama.html",
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
                    title: "Smolagents",
                    src: "img/smolagents.png",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/smolagents.html",
                  },
                  {
                    title: "Groq",
                    src: "img/groq.svg",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/groq.html",
                  },
                  {
                    title: "Mistral",
                    src: "img/mistral.png",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/mistral.html",
                  },
                  {
                    title: "DeepSeek",
                    src: "img/deepseek.png",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/deepseek.html",
                  },
                  {
                    title: "Haystack",
                    src: "img/haystack.png",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/haystack.html",
                  },
                  {
                    title: "Claude Code",
                    src: "img/claude.png",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/claude_code.html",
                  },
                  {
                    title: "AG2",
                    src: "img/ag2.svg",
                    href: "https://mlflow.org/docs/latest/genai/tracing/integrations/listing/ag2.html",
                  },
                  {
                    title: "Sentence Transformers",
                    src: "img/sentence-transformers.svg",
                    href: "https://mlflow.org/docs/latest/ml/deep-learning/sentence-transformers/index.html",
                  },
                  {
                    title: "ONNX",
                    src: "img/onnx.svg",
                    href: "https://mlflow.org/docs/latest/ml/model/index.html#onnx-onnx",
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
                defaultVisibleCount={window.innerWidth > 996 ? 18 : 6}
                renderItem={({ title, src, href }) => (
                  <MiniLogoCard title={title} src={src} href={href} />
                )}
              />
            )}
          </BrowserOnly>
        </div>
      </Section>
    </div>
  );
};
