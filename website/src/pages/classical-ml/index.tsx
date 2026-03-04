import { MLFLOW_ML_DOCS_URL } from "@site/src/constants";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import {
  Layout,
  LogosCarousel,
  LatestNews,
  AboveTheFold,
  BelowTheFold,
  StickyGrid,
  ValuePropWidget,
  HighlightedKeyword,
  Button,
  ProcessSection,
} from "../../components";
import { TrustPills } from "../../components/TrustPills/TrustPills";
import Card1 from "@site/static/img/Classical_home/Classical_home_1.png";
import Card2 from "@site/static/img/Classical_home/Classical_home_2.png";
import Card3 from "@site/static/img/Classical_home/Classical_home_3.png";
import Card4 from "@site/static/img/Classical_home/Classical_home_4.png";
import Card5 from "@site/static/img/Classical_home/Classical_home_5.png";

const SEO_TITLE = "MLflow for ML Models | MLflow AI Platform";
const SEO_DESCRIPTION =
  "MLflow for machine learning models. Manage the full ML and deep learning lifecycle with experiment tracking, hyperparameter tuning, model registry, and deployment.";

export default function ClassicalML(): JSX.Element {
  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://mlflow.org/classical-ml" />
        <link rel="canonical" href="https://mlflow.org/classical-ml" />
        <script type="application/ld+json">
          {JSON.stringify({
            "@context": "https://schema.org",
            "@type": "SoftwareApplication",
            name: "MLflow",
            applicationCategory: "DeveloperApplication",
            operatingSystem: "Cross-platform",
            offers: {
              "@type": "Offer",
              price: "0",
              priceCurrency: "USD",
            },
            description: SEO_DESCRIPTION,
            url: "https://mlflow.org/classical-ml",
            license: "https://www.apache.org/licenses/LICENSE-2.0",
          })}
        </script>
      </Head>
      <Layout>
        <AboveTheFold
          title={
            <span className="text-[48px] xxs:text-[64px] xs:text-[80px] leading-[110%]">
              Master the ML Lifecycle
            </span>
          }
          body={
            <>
              From experimentation to production, MLflow for machine learning
              models streamlines your complete ML journey, with comprehensive{" "}
              <HighlightedKeyword href="/classical-ml/experiment-tracking">
                experiment tracking
              </HighlightedKeyword>
              ,{" "}
              <HighlightedKeyword href="/classical-ml/model-registry">
                model management
              </HighlightedKeyword>
              , and{" "}
              <HighlightedKeyword href="/classical-ml/models">
                deployment
              </HighlightedKeyword>
              .
            </>
          }
          bodyColor="white"
          bodySize="xl"
          actions={
            <div className="flex flex-col items-center gap-4">
              <div className="flex flex-wrap justify-center items-center gap-4">
                <Link to="#get-started">
                  <Button variant="primary" size="medium">
                    Get Started
                  </Button>
                </Link>
                <Link to={MLFLOW_ML_DOCS_URL}>
                  <Button variant="outline" size="medium">
                    View Docs
                  </Button>
                </Link>
              </div>
              <TrustPills />
            </div>
          }
        />

        <LogosCarousel />

        <StickyGrid
          cards={[
            {
              title: "Build production quality models",
              body: "MLflow makes it easy to iterate toward production-ready models by organizing and comparing runs, helping teams refine training pipelines based on real performance insights.",
              cta: {
                text: "Learn more",
                href: "/classical-ml/experiment-tracking",
              },
              image: (
                <img
                  src={Card1}
                  alt="Building production quality ML models with MLflow"
                />
              ),
            },
            {
              title: "Framework neutral",
              body: "Works seamlessly with popular tools like scikit-learn, PyTorch, TensorFlow, and XGBoost without vendor lock-in, providing flexibility with a common interface.",
              cta: {
                text: "Learn more",
                href: "/classical-ml/experiment-tracking",
              },
              image: (
                <img
                  src={Card2}
                  alt="MLflow integrations with scikit-learn, PyTorch, TensorFlow, and XGBoost"
                />
              ),
            },
            {
              title: "Reliable reproducibility",
              body: "Automatically logs parameters, weights, artifacts, code, metrics, and dependencies to ensure experiments can be restored accurately, enabling confident governance for enterprise deployments.",
              cta: {
                text: "Learn more",
                href: "/classical-ml/experiment-tracking",
              },
              image: (
                <img
                  src={Card3}
                  alt="MLflow logging parameters, artifacts, and dependencies for reproducible experiments"
                />
              ),
            },
            {
              title: "Deployment ready",
              body: "Simplifies the path from experimentation to production with a built-in registry that gives you complete control over model states, whether sharing new approaches or deploying solutions.",
              cta: {
                text: "Learn more",
                href: "/classical-ml/model-registry",
              },
              image: (
                <img
                  src={Card4}
                  alt="MLflow model registry with lifecycle management and deployment controls"
                />
              ),
            },
            {
              title: "Unified workflow",
              body: "MLflow streamlines your entire ML process with tracking, packaging, and deployment capabilities, eliminating tool fragmentation so you can focus on model development rather than infrastructure",
              cta: {
                text: "Learn more",
                href: "/classical-ml/models",
              },
              image: (
                <img
                  src={Card5}
                  alt="Unified ML workflow from experimentation to production"
                />
              ),
            },
          ]}
        />

        <ValuePropWidget />

        <ProcessSection
          subtitle="From zero to full experiment tracking in minutes. No complex setup required."
          colorTheme="default"
          getStartedLink="https://mlflow.org/docs/latest/ml/"
          steps={[
            {
              number: "1",
              title: "Start MLflow Server",
              description:
                "One command to get started. Docker setup is also available.",
              time: "~30 seconds",
              code: "uvx mlflow server",
              language: "bash",
            },
            {
              number: "2",
              title: "Enable Autologging",
              description:
                "One line to automatically capture parameters, metrics, and models.",
              time: "~30 seconds",
              code: `import mlflow

mlflow.set_tracking_uri(
    "http://localhost:5000"
)
mlflow.sklearn.autolog()`,
              language: "python",
            },
            {
              number: "3",
              title: "Train Your Model",
              description:
                "Train as usual. Explore runs, metrics, and models in the MLflow UI.",
              time: "~1 minute",
              code: `from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.datasets import (
    load_iris,
)

X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier()
clf.fit(X, y)`,
              language: "python",
            },
          ]}
        />

        <BelowTheFold contentType="classical-ml" hideGetStarted>
          <LatestNews />
        </BelowTheFold>
      </Layout>
    </>
  );
}
