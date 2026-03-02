import React from "react";
import TracingTabImg from "@site/static/img/GenAI_home/GenAI_trace_darkmode.png";
import EvaluationTabImg from "@site/static/img/GenAI_home/GenAI_evaluation_darkmode.png";
import GatewayTabImg from "@site/static/img/GenAI_home/GenAI_gateway_darkmode.png";
import PromptTabImg from "@site/static/img/GenAI_home/GenAI_prompts_darkmode.png";
import ExperimentTrackingImg from "@site/static/img/GenAI_home/model_training_darkmode.png";
import ModelRegistryImg from "@site/static/img/GenAI_home/model_registry_darkmode.png";
import DeploymentImg from "@site/static/img/GenAI_home/deployment.png";
import Link from "@docusaurus/Link";

export type Feature = {
  id: string;
  title: string;
  description: string | React.ReactElement;
  imageSrc?: string;
  imageZoom?: number;
  imagePosition?: string;
  quickstartLink?: string;
  codeSnippet: string;
  codeLanguage?: "python" | "typescript";
  fullBleedImage?: boolean;
  imageFit?: "cover" | "contain";
};

export type Category = {
  id: string;
  label: string;
  features: Feature[];
};

const llmAgentFeatures: Feature[] = [
  {
    id: "observability",
    title: "Observability",
    description: (
      <>
        Capture complete{" "}
        <Link
          href="/faq/llm-tracing"
          className="text-white/90 hover:text-white underline"
        >
          traces
        </Link>{" "}
        of your LLM applications and agents to get deep{" "}
        <Link
          href="/faq/ai-observability"
          className="text-white/90 hover:text-white underline"
        >
          insights
        </Link>{" "}
        into their behavior. Built on OpenTelemetry and supports any LLM
        provider and agent framework.
      </>
    ),
    imageSrc: TracingTabImg,
    imageZoom: 160,
    quickstartLink: "https://mlflow.org/docs/latest/genai/tracing/quickstart/",
    codeSnippet: `import mlflow
import openai

# Enable auto-tracing for OpenAI - just 1 line!
mlflow.openai.autolog()

# All OpenAI calls are now automatically traced
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "What is MLflow?"},
    ],
)`,
  },
  {
    id: "evaluation",
    title: "Evaluation",
    description:
      "Run systematic evaluations, track quality metrics over time, and catch regressions before they reach production. Choose from 50+ built-in metrics and LLM judges, or define your own with highly flexible APIs.",
    imageSrc: EvaluationTabImg,
    quickstartLink:
      "https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/",
    codeSnippet: `import mlflow
from mlflow.genai.scorers import Correctness

# Define evaluation dataset
eval_data = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "outputs": {"response": "Paris"},
        "expectations": {"expected_response": "Paris"},
    },
]

# Run evaluation with LLM-as-judge
results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[Correctness()]
)`,
  },
  {
    id: "prompt",
    title: "Prompts & Optimization",
    description:
      "Version, test, and deploy prompts with full lineage tracking. Automatically optimize prompts with state-of-the-art algorithms to improve performance.",
    imageSrc: PromptTabImg,
    imageZoom: 150,
    quickstartLink: "https://mlflow.org/docs/latest/genai/prompt-registry/",
    codeSnippet: `import mlflow

# Register a prompt template
mlflow.genai.register_prompt(
    name="summarization",
    template="""
Summarize the following content in {{ num_sentences }} sentences.
Content: {{ content }}
""",
    commit_message="Initial version",
)

# Load and use prompts in your app
prompt = mlflow.genai.load_prompt("prompts:/summarization@latest")
formatted = prompt.format(num_sentences=2, content="...")`,
  },
  {
    id: "gateway",
    title: "AI Gateway",
    description: (
      <>
        Unified API{" "}
        <Link
          href="/faq/ai-gateway"
          className="text-white/90 hover:text-white underline"
        >
          gateway
        </Link>{" "}
        for all LLM providers. Route requests, manage rate limits, handle
        fallbacks, and control costs through a unified OpenAI-compatible
        interface.
      </>
    ),
    imageSrc: GatewayTabImg,
    imagePosition: "0% top",
    quickstartLink:
      "https://mlflow.org/docs/latest/genai/governance/ai-gateway/quickstart/",
    codeSnippet: `from openai import OpenAI

# Point to MLflow AI Gateway - OpenAI compatible API
client = OpenAI(
    base_url="http://localhost:5000/gateway/v1",
    api_key="mlflow",  # Gateway handles auth
)

# Gateway routes to configured providers
# with rate limiting, fallbacks, and cost tracking
response = client.chat.completions.create(
    model="gpt-5.2",  # or "claude-opus-4.5", "gemini-3-flash", etc.
    messages=[{"role": "user", "content": "Hello!"}]
)`,
  },
  {
    id: "agent-server",
    title: "Agent Server",
    description:
      "Deploy agents to production with a single command. The MLflow Agent Server provides a FastAPI-based hosting solution with automatic request validation, streaming support, and built-in tracing — so you can go from prototype to production endpoint in minutes.",
    quickstartLink:
      "https://mlflow.org/docs/latest/genai/serving/agent-server/",
    codeSnippet: `from mlflow.agent_server import AgentServer, invoke, stream
from mlflow.types.agent import ResponsesAgentRequest, ResponsesAgentResponse

@invoke()
async def run_agent(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    msgs = [i.model_dump() for i in request.input]
    result = await Runner.run(agent, msgs)
    return ResponsesAgentResponse(
        output=[item.to_input_item() for item in result.new_items]
    )

# Start the server
agent_server = AgentServer("MyAgent")
agent_server.run(app_import_string="server:app")`,
  },
];

const modelTrainingFeatures: Feature[] = [
  {
    id: "experiment-tracking",
    title: "Experiment Tracking",
    description:
      "Track experiments, log parameters, metrics, and artifacts. Compare runs side-by-side, reproduce results, and collaborate with your team on ML experiments.",
    imageSrc: ExperimentTrackingImg,
    imageZoom: 150,
    quickstartLink:
      "https://mlflow.org/docs/latest/ml/getting-started/quickstart/",
    codeSnippet: `import mlflow

# Enable autologging for your ML framework
mlflow.sklearn.autolog()

# Train your model - MLflow automatically logs everything
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)  # Parameters, metrics, model logged automatically`,
  },
  {
    id: "model-registry",
    title: "Model Registry",
    description:
      "Central hub to manage the full lifecycle of ML models. Version models, track lineage, manage stage transitions, and collaborate on model development.",
    imageSrc: ModelRegistryImg,
    imageZoom: 120,
    quickstartLink:
      "https://mlflow.org/docs/latest/ml/model-registry/tutorial/",
    codeSnippet: `import mlflow

# Register a model from a run
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "fraud-detection-model")

# Load a registered model for inference
model = mlflow.pyfunc.load_model(
    "models:/fraud-detection-model@champion"
)

# Make predictions
predictions = model.predict(new_data)`,
  },
  {
    id: "model-deployment",
    title: "Deployment",
    description:
      "Deploy models to production with a single command. Serve models as REST APIs, batch inference jobs, or integrate with cloud platforms like AWS, Azure, and Databricks.",
    imageSrc: DeploymentImg,
    imageZoom: 110,
    quickstartLink: "https://mlflow.org/docs/latest/ml/deployment/",
    codeSnippet: `# Serve model as REST API
mlflow models serve -m "models:/my-model@champion" -p 5000

# Or deploy to cloud platforms
mlflow deployments create -t sagemaker \\
    -m "models:/my-model@champion" \\
    --name my-deployment

# Query the deployed model
import requests
response = requests.post(
    "http://localhost:5000/invocations",
    json={"inputs": [[1, 2, 3, 4]]}
)`,
    codeLanguage: "python",
  },
];

export const categories: Category[] = [
  {
    id: "llm-agents",
    label: "LLMs & Agents",
    features: llmAgentFeatures,
  },
  {
    id: "model-training",
    label: "Model Training",
    features: modelTrainingFeatures,
  },
];
