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
    title: "オブザーバビリティ",
    description:
      "LLMアプリケーションやエージェントの完全なトレースをキャプチャし、動作を深く理解できます。OpenTelemetryベースで、あらゆるLLMプロバイダーやエージェントフレームワークに対応。本番環境の品質、コスト、安全性を監視します。",
    imageSrc: TracingTabImg,
    imageZoom: 160,
    quickstartLink: "https://mlflow.org/docs/latest/genai/tracing/quickstart/",
    codeSnippet: `import mlflow
import openai

# OpenAIの自動トレースを有効化 - たった1行！
mlflow.openai.autolog()

# すべてのOpenAI呼び出しが自動的にトレースされます
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "MLflowとは？"},
    ],
)`,
  },
  {
    id: "evaluation",
    title: "評価",
    description:
      "体系的な評価を実行し、品質メトリクスを経時的に追跡し、本番環境に到達する前にリグレッションを検出します。50以上の組み込みメトリクスとLLMジャッジから選択するか、柔軟なAPIで独自の評価を定義できます。",
    imageSrc: EvaluationTabImg,
    quickstartLink:
      "https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/",
    codeSnippet: `import mlflow
from mlflow.genai.scorers import Correctness

# 評価データセットを定義
eval_data = [
    {
        "inputs": {"question": "フランスの首都は？"},
        "outputs": {"response": "パリ"},
        "expectations": {"expected_response": "パリ"},
    },
]

# LLM-as-Judgeで評価を実行
results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[Correctness()]
)`,
  },
  {
    id: "prompt",
    title: "プロンプト管理・最適化",
    description:
      "プロンプトのバージョン管理、テスト、デプロイを完全なリネージ追跡付きで実行。最先端のアルゴリズムでプロンプトを自動最適化し、パフォーマンスを向上させます。",
    imageSrc: PromptTabImg,
    imageZoom: 150,
    quickstartLink: "https://mlflow.org/docs/latest/genai/prompt-registry/",
    codeSnippet: `import mlflow

# プロンプトテンプレートを登録
mlflow.genai.register_prompt(
    name="summarization",
    template="""
{{ num_sentences }}文で以下の内容を要約してください。
内容: {{ content }}
""",
    commit_message="初期バージョン",
)

# アプリでプロンプトを読み込んで使用
prompt = mlflow.genai.load_prompt("prompts:/summarization@latest")
formatted = prompt.format(num_sentences=2, content="...")`,
  },
  {
    id: "gateway",
    title: "AIゲートウェイ",
    description:
      "すべてのLLMプロバイダーへの統一APIゲートウェイ。リクエストのルーティング、レート制限の管理、フォールバック処理、コスト管理をOpenAI互換インターフェースで実現します。",
    imageSrc: GatewayTabImg,
    imagePosition: "0% top",
    quickstartLink:
      "https://mlflow.org/docs/latest/genai/governance/ai-gateway/quickstart/",
    codeSnippet: `from openai import OpenAI

# MLflow AIゲートウェイに接続 - OpenAI互換API
client = OpenAI(
    base_url="http://localhost:5000/gateway/v1",
    api_key="mlflow",  # ゲートウェイが認証を管理
)

# ゲートウェイが設定済みプロバイダーにルーティング
# レート制限、フォールバック、コスト追跡付き
response = client.chat.completions.create(
    model="gpt-5.2",  # "claude-opus-4.5", "gemini-3-flash" なども可
    messages=[{"role": "user", "content": "こんにちは！"}]
)`,
  },
  {
    id: "agent-server",
    title: "エージェントサーバー",
    description:
      "1つのコマンドでエージェントを本番環境にデプロイ。MLflowエージェントサーバーは、自動リクエストバリデーション、ストリーミングサポート、組み込みトレースを備えたFastAPIベースのホスティングソリューションを提供します。",
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

# サーバーを起動
agent_server = AgentServer("MyAgent")
agent_server.run(app_import_string="server:app")`,
  },
];

const modelTrainingFeatures: Feature[] = [
  {
    id: "experiment-tracking",
    title: "実験トラッキング",
    description:
      "実験を追跡し、パラメータ、メトリクス、アーティファクトを記録。ランを並べて比較し、結果を再現し、チームでML実験を共同作業できます。",
    imageSrc: ExperimentTrackingImg,
    imageZoom: 150,
    quickstartLink:
      "https://mlflow.org/docs/latest/ml/getting-started/quickstart/",
    codeSnippet: `import mlflow

# MLフレームワークの自動ロギングを有効化
mlflow.sklearn.autolog()

# モデルをトレーニング - MLflowが全てを自動記録
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)  # パラメータ、メトリクス、モデルが自動記録`,
  },
  {
    id: "model-registry",
    title: "モデルレジストリ",
    description:
      "MLモデルの全ライフサイクルを管理する中央ハブ。モデルのバージョン管理、リネージ追跡、ステージ遷移管理、モデル開発のコラボレーションが可能です。",
    imageSrc: ModelRegistryImg,
    imageZoom: 120,
    quickstartLink:
      "https://mlflow.org/docs/latest/ml/model-registry/tutorial/",
    codeSnippet: `import mlflow

# ランからモデルを登録
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "fraud-detection-model")

# 登録済みモデルを推論用にロード
model = mlflow.pyfunc.load_model(
    "models:/fraud-detection-model@champion"
)

# 予測を実行
predictions = model.predict(new_data)`,
  },
  {
    id: "model-deployment",
    title: "デプロイメント",
    description:
      "1つのコマンドでモデルを本番環境にデプロイ。REST APIとしてモデルを提供したり、バッチ推論ジョブを実行したり、AWS、Azure、Databricksなどのクラウドプラットフォームと統合できます。",
    imageSrc: DeploymentImg,
    imageZoom: 110,
    quickstartLink: "https://mlflow.org/docs/latest/ml/deployment/",
    codeSnippet: `# モデルをREST APIとして提供
mlflow models serve -m "models:/my-model@champion" -p 5000

# またはクラウドプラットフォームにデプロイ
mlflow deployments create -t sagemaker \\
    -m "models:/my-model@champion" \\
    --name my-deployment

# デプロイ済みモデルにクエリ
import requests
response = requests.post(
    "http://localhost:5000/invocations",
    json={"inputs": [[1, 2, 3, 4]]}
)`,
    codeLanguage: "python",
  },
];

type CategoryJP = {
  id: string;
  label: string;
  features: Feature[];
};

const categoriesJP: CategoryJP[] = [
  {
    id: "llm-agents",
    label: "LLM アプリケーション & エージェント",
    features: llmAgentFeatures,
  },
  {
    id: "model-training",
    label: "ML モデルトレーニング",
    features: modelTrainingFeatures,
  },
];

const SEO_TITLE =
  "MLflow - オープンソースAIプラットフォーム | エージェント・LLM・モデル";
const SEO_DESCRIPTION =
  "MLflowは、AIエージェント、LLM、機械学習モデルのための最大級のオープンソースAIエンジニアリングプラットフォームです。AIアプリケーションのデバッグ、評価、モニタリング、最適化を実現します。";

const GETTING_STARTED_STEPS = [
  {
    number: "1",
    title: "インストール",
    description: "1つのコマンドで起動できます。",
    time: "〜30秒",
    code: `pip install mlflow
mlflow server`,
    language: "bash",
  },
  {
    number: "2",
    title: "トレースを記録",
    description: "数行のコードでトレースの自動記録を開始できます。",
    time: "〜30秒",
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
    title: "UIで確認",
    description: "MLflow UIでトレースやメトリクスを確認できます。",
    time: "〜1分",
    code: `# http://localhost:5000
# でトレースを確認`,
    language: "bash",
  },
];

const BENEFITS = [
  {
    icon: <LockOpen className="w-6 h-6" />,
    title: "オープンソース",
    description:
      "Apache 2.0ライセンスの100%オープンソース。永久無料、制限なし。",
    iconBg: "bg-blue-500/20",
    iconColor: "text-blue-400",
  },
  {
    icon: <LinkIcon className="w-6 h-6" />,
    title: "ベンダーロックインなし",
    description:
      "あらゆるクラウド、フレームワーク、ツールと連携。いつでもベンダーを変更可能。",
    iconBg: "bg-purple-500/20",
    iconColor: "text-purple-400",
  },
  {
    icon: <Zap className="w-6 h-6" />,
    title: "本番環境対応",
    description: "Fortune 500企業や数千のチームで大規模に実証済み。",
    iconBg: "bg-amber-500/20",
    iconColor: "text-amber-400",
  },
  {
    icon: <BarChart3 className="w-6 h-6" />,
    title: "完全な可視性",
    description:
      "すべてのAIアプリケーションとエージェントの完全なトラッキングとオブザーバビリティ。",
    iconBg: "bg-cyan-500/20",
    iconColor: "text-cyan-400",
  },
  {
    icon: <Users className="w-6 h-6" />,
    title: "コミュニティ",
    description:
      "20K+ GitHub Stars、900+コントリビューター。最も成長しているAIOpsコミュニティ。",
    iconBg: "bg-green-500/20",
    iconColor: "text-green-400",
  },
  {
    icon: <Puzzle className="w-6 h-6" />,
    title: "豊富なインテグレーション",
    description:
      "LangChain、OpenAI、PyTorchなど100以上のAIフレームワークとすぐに連携可能。",
    iconBg: "bg-rose-500/20",
    iconColor: "text-rose-400",
  },
];

const socialsJP = [
  {
    key: "docs",
    icon: (
      <div>
        <BookIcon />
      </div>
    ),
    label: "ドキュメント",
    description: "ドキュメントを読む",
    href: MLFLOW_DOCS_URL,
  },
  {
    key: "github",
    icon: <GithubIcon />,
    label: "GitHub",
    description: "20k+ スター",
    href: "https://github.com/mlflow/mlflow",
  },
  {
    key: "linkedin",
    icon: <LinkedinIcon />,
    label: "LinkedIn",
    description: "69k フォロワー",
    href: "https://www.linkedin.com/company/mlflow-org",
  },
  {
    key: "youtube",
    icon: <YoutubeIcon />,
    label: "YouTube",
    description: "チュートリアルを見る",
    href: "https://www.youtube.com/@mlflowoss",
  },
  {
    key: "x",
    icon: <XIcon />,
    label: "X",
    description: "フォローする",
    href: "https://x.com/mlflow",
  },
  {
    key: "slack",
    icon: <SlackIcon />,
    label: "Slack",
    description: "Slackに参加",
    href: "https://go.mlflow.org/slack",
  },
];

function SocialWidgetJP() {
  const stars = useGitHubStars();
  return (
    <div className="flex flex-col w-full gap-16">
      <div className="flex flex-col w-full gap-6 items-center justify-center text-center">
        <SectionLabel label="コミュニティ" />
        <Heading level={2}>オープンソースコミュニティに参加</Heading>
        <Body size="l">世界中のMLflowユーザーとつながりましょう</Body>
      </div>
      <Grid className="px-10">
        {socialsJP.map((social) => (
          <SocialWidgetItem
            key={social.key}
            href={social.href}
            icon={social.icon}
            label={social.label}
            description={
              social.key === "github" && stars
                ? `${stars}+ スター`
                : social.description
            }
          />
        ))}
      </Grid>
    </div>
  );
}

function ProductTabsJP() {
  const [activeCategory, setActiveCategory] = useState(categoriesJP[0].id);
  const activeFeatures =
    categoriesJP.find((c) => c.id === activeCategory)?.features ?? [];

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
              {categoriesJP.map((category) => {
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
                        layoutId="activeUnderlineJP"
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

export default function JapanesePage(): JSX.Element {
  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://mlflow.org/jp" />
        <meta property="og:locale" content="ja_JP" />
        <link rel="canonical" href="https://mlflow.org/jp" />
        <html lang="ja" />
      </Head>

      <Layout>
        {/* 1. HERO SECTION */}
        <HeroSection
          title="高品質なAIを、もっと速く"
          subtitle={
            <>
              AIプロダクト開発はイテレーションが全てです。
              <br />
              MLflowは、LLMアプリケーション、エージェント、モデルの
              <br />
              <HighlightedKeyword href="https://mlflow.org/docs/latest/genai/tracing/">
                デバッグ
              </HighlightedKeyword>
              ・
              <HighlightedKeyword href="https://mlflow.org/docs/latest/genai/eval-monitor/">
                評価
              </HighlightedKeyword>
              ・
              <HighlightedKeyword href="https://mlflow.org/docs/latest/genai/tracing/prod-tracing/">
                モニタリング
              </HighlightedKeyword>
              を10倍速くします。
            </>
          }
          primaryCTA={{
            label: "はじめる",
            href: "#get-started",
          }}
          secondaryCTA={{
            label: "ドキュメントを見る",
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
                日本の皆様へ
              </h2>
              <p className="text-white/70 text-lg leading-relaxed mb-4">
                このページは、MLflow
                を日本語で分かりやすくご紹介するために作成しました。日本のユーザーや企業の皆様に心より感謝申し上げます。
              </p>
              <p className="text-white/50 text-sm">
                なお、このページ以外の MLflow
                ウェブサイトは英語のみでの提供となります。Google Chrome
                の翻訳機能などをご利用いただくことで、日本語でもご覧いただけます。
              </p>
            </div>
          </div>
        </div>

        {/* 2. COMPANY LOGOS */}
        <LogosCarousel />

        {/* 3. FEATURES SECTION */}
        <ProductTabsJP />

        {/* 4. TRUST LOGOS */}
        <StatsBand
          title="最も採用されているオープンソースAIOpsプラットフォーム"
          body={
            <>
              Linux
              Foundationの支援のもと、MLflowは5年以上にわたりオープンソースに取り組んできました。現在、世界中の数千の組織や研究チームが
              <Link
                href="/llmops"
                style={{ color: "inherit", textDecoration: "underline" }}
              >
                LLMOps
              </Link>
              や
              <Link
                href="/classical-ml"
                style={{ color: "inherit", textDecoration: "underline" }}
              >
                MLOps
              </Link>
              のワークフローに活用しています。
            </>
          }
        />

        {/* 5. INTEGRATIONS */}
        <EcosystemList
          title="あらゆるフレームワークと連携"
          body="LLMエージェントフレームワークから従来のMLライブラリまで、MLflowは100以上のツールとシームレスに統合できます。Python、TypeScript/JavaScript、Java、Rをサポートし、OpenTelemetryにもネイティブ対応しています。"
          seeMoreLabel="もっと見る ∨"
          seeLessLabel="閉じる ∧"
        />

        {/* 6. BENEFITS SECTION */}
        <Section
          title="チームがMLflowを選ぶ理由"
          body="優れたAIの構築に集中できます。MLflowが複雑さを処理し、より速い開発を実現します。"
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
          title="3ステップで開始"
          subtitle="数分でLLMOpsを始められます。複雑なセットアップは不要です。"
          steps={GETTING_STARTED_STEPS}
          getStartedLink="https://mlflow.org/docs/latest/genai/tracing/quickstart/"
          getStartedLabel="はじめる →"
        />

        {/* 8. COMMUNITY */}
        <SocialWidgetJP />

        {/* 9. FOOTER */}
        <BelowTheFold hideGetStarted hideSocialWidget />
      </Layout>
    </>
  );
}
