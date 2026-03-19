import { useState } from "react";
import {
  Layout,
  BelowTheFold,
  EcosystemList,
  LogosCarousel,
  StatsBand,
  HeroSection,
  HighlightedKeyword,
  ProcessSection,
  TrustPills,
} from "../components";
import { Section } from "../components/Section/Section";
import { StickyFeaturesGrid } from "../components/ProductTabs/StickyFeaturesGrid";
import type { Feature } from "../components/ProductTabs/features";
import Link from "@docusaurus/Link";
import { MLFLOW_DOCS_URL } from "../constants";
import Head from "@docusaurus/Head";
import clsx from "clsx";
import { motion, AnimatePresence } from "motion/react";
import {
  LockOpen,
  Link as LinkIcon,
  Zap,
  BarChart3,
  Users,
  Puzzle,
} from "lucide-react";
import GithubIcon from "@site/static/img/social/github.svg";
import YoutubeIcon from "@site/static/img/social/youtube.svg";
import BookIcon from "@site/static/img/social/book.svg";
import LinkedinIcon from "@site/static/img/social/linkedin.svg";
import XIcon from "@site/static/img/social/x.svg";
import SlackIcon from "@site/static/img/social/slack.svg";
import { SocialWidgetItem } from "../components/SocialWidgetItem/SocialWidgetItem";
import { Grid } from "../components/Grid/Grid";
import { SectionLabel } from "../components/Section/SectionLabel";
import { Heading } from "../components/Typography/Heading";
import { Body } from "../components/Typography/Body";
import { useGitHubStars } from "../hooks/useGitHubStars";
import TracingTabImg from "@site/static/img/GenAI_home/GenAI_trace_darkmode.png";
import EvaluationTabImg from "@site/static/img/GenAI_home/GenAI_evaluation_darkmode.png";
import GatewayTabImg from "@site/static/img/GenAI_home/GenAI_gateway_darkmode.png";
import PromptTabImg from "@site/static/img/GenAI_home/GenAI_prompts_darkmode.png";
import ExperimentTrackingImg from "@site/static/img/GenAI_home/model_training_darkmode.png";
import ModelRegistryImg from "@site/static/img/GenAI_home/model_registry_darkmode.png";
import DeploymentImg from "@site/static/img/GenAI_home/deployment.png";

const llmAgentFeatures: Feature[] = [
  {
    id: "observability",
    title: "可观测性",
    description:
      "捕获 LLM 应用程序和 Agent 的完整追踪，深入了解其行为。基于 OpenTelemetry 构建，支持任何 LLM 提供商和 Agent 框架。监控生产环境的质量、成本和安全性。",
    imageSrc: TracingTabImg,
    imageZoom: 160,
    quickstartLink: "https://mlflow.org/docs/latest/genai/tracing/quickstart/",
    codeSnippet: `import mlflow
import openai

# 启用 OpenAI 自动追踪 - 仅需一行代码！
mlflow.openai.autolog()

# 所有 OpenAI 调用将被自动追踪
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "什么是 MLflow？"},
    ],
)`,
  },
  {
    id: "evaluation",
    title: "评估",
    description:
      "运行系统化评估，跟踪质量指标的变化趋势，在问题到达生产环境之前检测回归。从 50 多个内置指标和 LLM 评判器中选择，或使用灵活的 API 定义自己的评估标准。",
    imageSrc: EvaluationTabImg,
    quickstartLink:
      "https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/",
    codeSnippet: `import mlflow
from mlflow.genai.scorers import Correctness

# 定义评估数据集
eval_data = [
    {
        "inputs": {"question": "法国的首都是哪里？"},
        "outputs": {"response": "巴黎"},
        "expectations": {"expected_response": "巴黎"},
    },
]

# 使用 LLM-as-Judge 运行评估
results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[Correctness()]
)`,
  },
  {
    id: "prompt",
    title: "提示词管理与优化",
    description:
      "通过完整的血缘追踪进行提示词的版本管理、测试和部署。使用最先进的算法自动优化提示词，提升性能表现。",
    imageSrc: PromptTabImg,
    imageZoom: 150,
    quickstartLink: "https://mlflow.org/docs/latest/genai/prompt-registry/",
    codeSnippet: `import mlflow

# 注册提示词模板
mlflow.genai.register_prompt(
    name="summarization",
    template="""
请用 {{ num_sentences }} 句话总结以下内容。
内容：{{ content }}
""",
    commit_message="初始版本",
)

# 在应用中加载和使用提示词
prompt = mlflow.genai.load_prompt("prompts:/summarization@latest")
formatted = prompt.format(num_sentences=2, content="...")`,
  },
  {
    id: "gateway",
    title: "AI 网关",
    description:
      "面向所有 LLM 提供商的统一 API 网关。通过 OpenAI 兼容接口实现请求路由、速率限制管理、故障转移处理和成本控制。",
    imageSrc: GatewayTabImg,
    imagePosition: "0% top",
    quickstartLink:
      "https://mlflow.org/docs/latest/genai/governance/ai-gateway/quickstart/",
    codeSnippet: `from openai import OpenAI

# 连接 MLflow AI 网关 - OpenAI 兼容 API
client = OpenAI(
    base_url="http://localhost:5000/gateway/v1",
    api_key="mlflow",  # 网关管理认证
)

# 网关路由到已配置的提供商
# 具备速率限制、故障转移和成本追踪
response = client.chat.completions.create(
    model="gpt-5.2",  # 也支持 "claude-opus-4.5"、"gemini-3-flash" 等
    messages=[{"role": "user", "content": "你好！"}]
)`,
  },
  {
    id: "agent-server",
    title: "Agent 服务器",
    description:
      "一条命令即可将 Agent 部署到生产环境。MLflow Agent 服务器提供基于 FastAPI 的托管解决方案，具备自动请求验证、流式支持和内置追踪功能。",
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

# 启动服务器
agent_server = AgentServer("MyAgent")
agent_server.run(app_import_string="server:app")`,
  },
];

const modelTrainingFeatures: Feature[] = [
  {
    id: "experiment-tracking",
    title: "实验追踪",
    description:
      "追踪实验，记录参数、指标和工件。并排比较不同运行结果，复现实验，与团队协作进行 ML 实验。",
    imageSrc: ExperimentTrackingImg,
    imageZoom: 150,
    quickstartLink:
      "https://mlflow.org/docs/latest/ml/getting-started/quickstart/",
    codeSnippet: `import mlflow

# 启用 ML 框架的自动日志记录
mlflow.sklearn.autolog()

# 训练模型 - MLflow 自动记录一切
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)  # 参数、指标、模型自动记录`,
  },
  {
    id: "model-registry",
    title: "模型注册中心",
    description:
      "管理 ML 模型全生命周期的中央枢纽。支持模型版本管理、血缘追踪、阶段转换管理和模型开发协作。",
    imageSrc: ModelRegistryImg,
    imageZoom: 120,
    quickstartLink:
      "https://mlflow.org/docs/latest/ml/model-registry/tutorial/",
    codeSnippet: `import mlflow

# 从运行中注册模型
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "fraud-detection-model")

# 加载已注册的模型用于推理
model = mlflow.pyfunc.load_model(
    "models:/fraud-detection-model@champion"
)

# 进行预测
predictions = model.predict(new_data)`,
  },
  {
    id: "model-deployment",
    title: "部署",
    description:
      "一条命令即可将模型部署到生产环境。可以将模型作为 REST API 提供服务、运行批量推理任务，或与 AWS、Azure、Databricks 等云平台集成。",
    imageSrc: DeploymentImg,
    imageZoom: 110,
    quickstartLink: "https://mlflow.org/docs/latest/ml/deployment/",
    codeSnippet: `# 将模型作为 REST API 提供服务
mlflow models serve -m "models:/my-model@champion" -p 5000

# 或部署到云平台
mlflow deployments create -t sagemaker \\
    -m "models:/my-model@champion" \\
    --name my-deployment

# 查询已部署的模型
import requests
response = requests.post(
    "http://localhost:5000/invocations",
    json={"inputs": [[1, 2, 3, 4]]}
)`,
    codeLanguage: "python",
  },
];

type CategoryCN = {
  id: string;
  label: string;
  features: Feature[];
};

const categoriesCN: CategoryCN[] = [
  {
    id: "llm-agents",
    label: "LLM 应用 & Agent",
    features: llmAgentFeatures,
  },
  {
    id: "model-training",
    label: "ML 模型训练",
    features: modelTrainingFeatures,
  },
];

const SEO_TITLE = "MLflow - 开源 AI 平台 | Agent・LLM・模型";
const SEO_DESCRIPTION =
  "MLflow 是面向 AI Agent、LLM 和机器学习模型的最大开源 AI 工程平台。实现 AI 应用的调试、评估、监控和优化。";

const GETTING_STARTED_STEPS = [
  {
    number: "1",
    title: "安装",
    description: "一条命令即可启动。",
    time: "~30秒",
    code: `pip install mlflow
mlflow server`,
    language: "bash",
  },
  {
    number: "2",
    title: "记录追踪",
    description: "几行代码即可开始自动记录追踪。",
    time: "~30秒",
    code: `import mlflow
from openai import OpenAI

mlflow.openai.autolog()

client = OpenAI()
client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "user",
     "content": "Hello!"}
  ],
)`,
    language: "python",
  },
  {
    number: "3",
    title: "在 UI 中查看",
    description: "在 MLflow UI 中查看追踪和指标。",
    time: "~1分钟",
    code: `# http://localhost:5000
# 查看追踪记录`,
    language: "bash",
  },
];

const BENEFITS = [
  {
    icon: <LockOpen className="w-6 h-6" />,
    title: "开源",
    description:
      "基于 Apache 2.0 许可证的 100% 开源项目。永久免费，无任何限制。",
    iconBg: "bg-blue-500/20",
    iconColor: "text-blue-400",
  },
  {
    icon: <LinkIcon className="w-6 h-6" />,
    title: "无供应商锁定",
    description: "兼容任何云平台、框架或工具。随时可以更换供应商。",
    iconBg: "bg-purple-500/20",
    iconColor: "text-purple-400",
  },
  {
    icon: <Zap className="w-6 h-6" />,
    title: "生产就绪",
    description: "经过世界 500 强企业和数千个团队的大规模验证。",
    iconBg: "bg-amber-500/20",
    iconColor: "text-amber-400",
  },
  {
    icon: <BarChart3 className="w-6 h-6" />,
    title: "全面可视化",
    description: "为所有 AI 应用和 Agent 提供完整的追踪和可观测性。",
    iconBg: "bg-cyan-500/20",
    iconColor: "text-cyan-400",
  },
  {
    icon: <Users className="w-6 h-6" />,
    title: "社区",
    description: "20K+ GitHub Stars，900+ 贡献者。加入增长最快的 AIOps 社区。",
    iconBg: "bg-green-500/20",
    iconColor: "text-green-400",
  },
  {
    icon: <Puzzle className="w-6 h-6" />,
    title: "丰富的集成",
    description:
      "开箱即用地支持 LangChain、OpenAI、PyTorch 等 100 多个 AI 框架。",
    iconBg: "bg-rose-500/20",
    iconColor: "text-rose-400",
  },
];

const socialsCN = [
  {
    key: "docs",
    icon: (
      <div>
        <BookIcon />
      </div>
    ),
    label: "文档",
    description: "阅读文档",
    href: MLFLOW_DOCS_URL,
  },
  {
    key: "github",
    icon: <GithubIcon />,
    label: "GitHub",
    description: "20k+ Stars",
    href: "https://github.com/mlflow/mlflow",
  },
  {
    key: "linkedin",
    icon: <LinkedinIcon />,
    label: "LinkedIn",
    description: "69k 关注者",
    href: "https://www.linkedin.com/company/mlflow-org",
  },
  {
    key: "youtube",
    icon: <YoutubeIcon />,
    label: "YouTube",
    description: "观看教程",
    href: "https://www.youtube.com/@mlflowoss",
  },
  {
    key: "x",
    icon: <XIcon />,
    label: "X",
    description: "关注我们",
    href: "https://x.com/mlflow",
  },
  {
    key: "slack",
    icon: <SlackIcon />,
    label: "Slack",
    description: "加入 Slack",
    href: "https://go.mlflow.org/slack",
  },
];

function SocialWidgetCN() {
  const stars = useGitHubStars();
  return (
    <div className="flex flex-col w-full gap-16">
      <div className="flex flex-col w-full gap-6 items-center justify-center text-center">
        <SectionLabel label="社区" />
        <Heading level={2}>加入开源社区</Heading>
        <Body size="l">与全球 MLflow 用户交流</Body>
      </div>
      <Grid className="px-10">
        {socialsCN.map((social) => (
          <SocialWidgetItem
            key={social.key}
            href={social.href}
            icon={social.icon}
            label={social.label}
            description={
              social.key === "github" && stars
                ? `${stars}+ Stars`
                : social.description
            }
          />
        ))}
      </Grid>
    </div>
  );
}

function ProductTabsCN() {
  const [activeCategory, setActiveCategory] = useState(categoriesCN[0].id);
  const activeFeatures =
    categoriesCN.find((c) => c.id === activeCategory)?.features ?? [];

  return (
    <div className="w-full px-4 md:px-8 lg:px-16 pb-36">
      <div className="max-w-7xl mx-auto">
        <motion.div
          className="w-full flex flex-col gap-12"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <div className="flex justify-center">
            <div className="flex gap-8">
              {categoriesCN.map((category) => {
                const isActive = category.id === activeCategory;
                return (
                  <button
                    key={category.id}
                    onClick={() => setActiveCategory(category.id)}
                    className={clsx(
                      "relative px-2 py-3 text-lg font-medium transition-colors",
                      isActive
                        ? "text-white"
                        : "text-white/50 hover:text-white/70",
                    )}
                  >
                    {category.label}
                    {isActive && (
                      <motion.div
                        layoutId="activeUnderlineCN"
                        className="absolute bottom-0 left-0 right-0 h-[2px]"
                        style={{
                          background:
                            "linear-gradient(90deg, #e05585, #9066cc, #5a8fd4)",
                        }}
                        transition={{
                          type: "spring",
                          stiffness: 400,
                          damping: 30,
                        }}
                      />
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="px-4">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeCategory}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3 }}
              >
                <StickyFeaturesGrid features={activeFeatures} />
              </motion.div>
            </AnimatePresence>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

export default function ChinesePage(): JSX.Element {
  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://mlflow.org/cn" />
        <meta property="og:locale" content="zh_CN" />
        <link rel="canonical" href="https://mlflow.org/cn" />
        <html lang="zh-CN" />
      </Head>

      <Layout>
        {/* 1. HERO SECTION */}
        <HeroSection
          title="更快交付高质量 AI"
          subtitle={
            <>
              AI 产品开发的核心在于快速迭代。
              <br />
              MLflow 让您的 LLM 应用、Agent 和模型的
              <br />
              <HighlightedKeyword href="https://mlflow.org/docs/latest/genai/tracing/">
                调试
              </HighlightedKeyword>
              、
              <HighlightedKeyword href="https://mlflow.org/docs/latest/genai/eval-monitor/">
                评估
              </HighlightedKeyword>
              、
              <HighlightedKeyword href="https://mlflow.org/docs/latest/genai/tracing/prod-tracing/">
                监控
              </HighlightedKeyword>
              快 10 倍。
            </>
          }
          primaryCTA={{
            label: "开始使用",
            href: "#get-started",
          }}
          secondaryCTA={{
            label: "查看文档",
            href: MLFLOW_DOCS_URL,
          }}
        >
          <TrustPills />
        </HeroSection>

        {/* WELCOME MESSAGE */}
        <div className="w-full px-4 md:px-8 lg:px-16 pt-16 pb-8">
          <div className="max-w-3xl mx-auto text-center">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-8 md:p-12">
              <h2 className="text-2xl md:text-3xl font-semibold text-white mb-4">
                致中国用户
              </h2>
              <p className="text-white/70 text-lg leading-relaxed mb-4">
                本页面旨在以中文向您介绍
                MLflow。衷心感谢中国的用户和企业对我们的支持。
              </p>
              <p className="text-white/50 text-sm">
                请注意，除本页面外，MLflow
                网站的其他内容均以英文提供。您可以使用 Google Chrome
                的翻译功能以中文浏览。
              </p>
            </div>
          </div>
        </div>

        {/* 2. COMPANY LOGOS */}
        <LogosCarousel />

        {/* 3. FEATURES SECTION */}
        <ProductTabsCN />

        {/* 4. TRUST LOGOS */}
        <StatsBand
          title="最广泛采用的开源 AIOps 平台"
          body={
            <>
              在 Linux Foundation 的支持下，MLflow 已坚持开源 5
              年以上。目前，全球数千个组织和研究团队正在使用 MLflow 来驱动其{" "}
              <Link
                href="/llmops"
                style={{ color: "inherit", textDecoration: "underline" }}
              >
                LLMOps
              </Link>{" "}
              和{" "}
              <Link
                href="/classical-ml"
                style={{ color: "inherit", textDecoration: "underline" }}
              >
                MLOps
              </Link>{" "}
              工作流。
            </>
          }
        />

        {/* 5. INTEGRATIONS */}
        <EcosystemList
          title="兼容所有框架"
          body="从 LLM Agent 框架到传统 ML 库，MLflow 与 100 多个工具无缝集成。支持 Python、TypeScript/JavaScript、Java、R，并原生支持 OpenTelemetry。"
          seeMoreLabel="查看更多 ∨"
          seeLessLabel="收起 ∧"
        />

        {/* 6. BENEFITS SECTION */}
        <Section
          title="团队选择 MLflow 的原因"
          body="专注于构建优秀的 AI。MLflow 处理复杂性，让您更快地交付产品。"
          align="center"
        >
          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-px max-w-6xl mx-auto bg-white/10"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            {BENEFITS.map((benefit, index) => (
              <motion.div
                key={benefit.title}
                className="relative p-6 bg-[#0E1416]"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <div
                  className={`w-12 h-12 ${benefit.iconBg} ${benefit.iconColor} flex items-center justify-center mb-4`}
                >
                  {benefit.icon}
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  {benefit.title}
                </h3>
                <p className="text-sm text-white/60 leading-relaxed">
                  {benefit.description}
                </p>
              </motion.div>
            ))}
          </motion.div>
        </Section>

        {/* 7. PROCESS SECTION */}
        <ProcessSection
          title="3 步开始"
          subtitle="几分钟即可开始 LLMOps。无需复杂设置。"
          steps={GETTING_STARTED_STEPS}
          getStartedLink="https://mlflow.org/docs/latest/genai/tracing/quickstart/"
          getStartedLabel="开始使用 →"
        />

        {/* 8. COMMUNITY */}
        <SocialWidgetCN />

        {/* 9. FOOTER */}
        <BelowTheFold hideGetStarted hideSocialWidget />
      </Layout>
    </>
  );
}
