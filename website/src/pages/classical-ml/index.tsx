import { MLFLOW_DOCS_URL } from "@site/src/constants";
import {
  Layout,
  LogosCarousel,
  LatestNews,
  AboveTheFold,
  BelowTheFold,
  StickyGrid,
  ValuePropWidget,
} from "../../components";
import Card1 from "@site/static/img/Classical_home/Classical_home_1.png";
import Card2 from "@site/static/img/Classical_home/Classical_home_2.png";
import Card3 from "@site/static/img/Classical_home/Classical_home_3.png";
import Card4 from "@site/static/img/Classical_home/Classical_home_4.png";
import Card5 from "@site/static/img/Classical_home/Classical_home_5.png";

export default function GenAi(): JSX.Element {
  return (
    <Layout>
      <AboveTheFold
        title="Mastering the ML lifecycle"
        body="From experiment to production, MLflow streamlines your complete machine learning journey with end-to-end tracking, model management, and deployment."
        hasGetStartedButton="#get-started"
        bodyColor="white"
      />

      <StickyGrid
        cards={[
          {
            title: "Build production quality models",
            body: "MLflow makes it easy to iterate toward production-ready models by organizing and comparing runs, helping teams refine training pipelines based on real performance insights.",
            cta: {
              text: "Learn more",
              href: "/classical-ml/experiment-tracking",
            },
            image: <img src={Card1} alt="" />,
          },
          {
            title: "Framework neutral",
            body: "Works seamlessly with popular tools like scikit-learn, PyTorch, TensorFlow, and XGBoost without vendor lock-in, providing flexibility with a common interface.",
            cta: {
              text: "Learn more",
              href: "/classical-ml/experiment-tracking",
            },
            image: <img src={Card2} alt="" />,
          },
          {
            title: "Reliable reproducibility",
            body: "Automatically logs parameters, weights, artifacts, code, metrics, and dependencies to ensure experiments can be restored accurately, enabling confident governance for enterprise deployments.",
            cta: {
              text: "Learn more",
              href: "/classical-ml/experiment-tracking",
            },
            image: <img src={Card3} alt="" />,
          },
          {
            title: "Deployment ready",
            body: "Simplifies the path from experimentation to production with a built-in registry that gives you complete control over model states, whether sharing new approaches or deploying solutions.",
            cta: {
              text: "Learn more",
              href: "/classical-ml/model-registry",
            },
            image: <img src={Card4} alt="" />,
          },
          {
            title: "Unified workflow",
            body: "MLflow streamlines your entire ML process with tracking, packaging, and deployment capabilities, eliminating tool fragmentation so you can focus on model development rather than infrastructure",
            cta: {
              text: "Learn more",
              href: "/classical-ml/models",
            },
            image: <img src={Card5} alt="" />,
          },
        ]}
      />

      <LogosCarousel />
      <ValuePropWidget />

      <BelowTheFold contentType="classical-ml">
        <LatestNews />
      </BelowTheFold>
    </Layout>
  );
}
