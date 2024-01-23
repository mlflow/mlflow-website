import React, { useState, useEffect, useRef } from "react";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import AutoScroll from "@site/src/components/AutoScroll";
import FloatingNav from "../components/FloatingNav";
import { latestBlogs, latestReleases } from "../posts";
import MLflowLogoAndCards from "../components/community-section/MLflowLogoAndCards";
import Arrow from "../components/community-section/Arrow";
import BrowserOnly from "@docusaurus/BrowserOnly";
import ArrowText from "../components/ArrowText";
import HeroBadges from "../components/HeroBadges";
import GetStarted from "../components/GetStarted";
import { H1, H2 } from "../components/Header";
import Blog from "../components/Blog";
import LearnCard from "../components/LearnCard";
import Grid from "../components/Grid";
import { DoubleGrid } from "../components/Grid";
import BenefitCard from "../components/BenefitCard";
import FeatureCard from "../components/FeatureCard";
import ReleaseNote from "../components/ReleaseNote";
import LogoCard from "../components/LogoCard";
import MiniLogoCard from "../components/MiniLogoCard";
import Anchor from "../components/Anchor";
import Flow from "../components/Flow";
import ConceptCard from "../components/ConceptCard";
import CenterGrid from "../components/CenterGrid";

const IMAGE =
  // "https://images.unsplash.com/photo-1506624183912-c602f4a21ca7?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=800&q=60";
  "img/media.png";

function Spacer({ height }: { height?: number }): JSX.Element {
  return <div style={{ height: height || 32 }} />;
}

interface Card {
  title: string;
  content: string;
  href: string;
  img?: string;
}

interface Item {
  title: string;
  content: JSX.Element;
}

const LOREM =
  "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Quis ipsum suspendisse ultrices gravida.";

function Pills({
  title,
  items,
}: {
  title: React.ReactNode;
  items: Item[];
}): JSX.Element {
  const [active, setActive] = useState(0);
  const activeClass = "pills__item--active";
  return (
    <>
      <div
        style={{
          background:
            "radial-gradient(61.2% 156.44% at 48.65% 192.33%, #43C9ED 0%, rgba(67, 201, 237, 0.00) 100%)",
          paddingBottom: "32px",
        }}
      >
        {title}
        <ul
          className="pills"
          style={{
            display: "flex",
            justifyContent: "center",
            flexWrap: "wrap",
            marginBottom: "0",
            marginTop: "64px",
          }}
        >
          {items.map(({ title }, i) => (
            <li
              className={`pills__item ${active === i ? activeClass : ""}`}
              key={title}
              onClick={() => setActive(i)}
            >
              {title}
            </li>
          ))}
        </ul>
      </div>
      {items[active].content}
    </>
  );
}

function CardTile({ cards }: { cards: Card[] }): JSX.Element {
  if (cards.length === 2) {
    return (
      <>
        <Line />
        <div
          style={{
            marginTop: "32px",
          }}
        />
        <DoubleGrid>
          {cards.map(({ title, content, href, img }, index) => (
            <LearnCard
              title={title}
              content={content}
              href={href}
              img={img}
              key={index}
            />
          ))}
        </DoubleGrid>
      </>
    );
  }
  return (
    <>
      <Line />
      <div
        style={{
          marginTop: "32px",
        }}
      />
      <Grid>
        {cards.map(({ title, content, href, img }, index) => (
          <LearnCard
            title={title}
            content={content}
            href={href}
            img={img}
            key={index}
          />
        ))}
      </Grid>
    </>
  );
}

const SECTIONS = {
  coreConcepts: "core-concepts",
  benefits: "benefits",
  features: "features",
  integrations: "integrations",
  learningResources: "learning-resources",
};

function Line(): JSX.Element {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        margin: 0,
        height: 0,
        width: "100%",
      }}
    >
      <svg
        style={{
          fill: "var(--ifm-color-primary)",
          height: "8px",
          width: "8px",
          zIndex: 2,
        }}
        viewBox="0 0 10 10"
        xmlns="http://www.w3.org/2000/svg"
      >
        <circle cx="50%" cy="50%" r="4.5" />
      </svg>
      <svg
        width="100%"
        strokeWidth={1}
        viewBox="0 0 200 10"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          vectorEffect="non-scaling-stroke"
          className="lines"
          d="M0 5 H 200"
        />
      </svg>
      <svg
        style={{
          fill: "var(--ifm-color-primary)",
          height: "8px",
          width: "8px",
          zIndex: 2,
        }}
        viewBox="0 0 10 10"
        xmlns="http://www.w3.org/2000/svg"
      >
        <circle cx="50%" cy="50%" r="4.5" />
      </svg>
    </div>
  );
}

export default function Home(): JSX.Element {
  const { siteConfig } = useDocusaurusContext();
  const [active, setActive] = useState<string>();

  const sectionRefs = Object.keys(SECTIONS).reduce(
    (acc, key) => ({ ...acc, [key]: useRef<HTMLAnchorElement>(null) }),
    {}
  ) as Record<keyof typeof SECTIONS, React.RefObject<HTMLAnchorElement>>;

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const { id } = entry.target;
            setActive(id);
            window.history.replaceState(null, null, `#${id}`);
          }
        });
      },
      { threshold: 0.5, rootMargin: "0% 0px -80% 0px" }
    );

    Object.values(sectionRefs).forEach((value) => {
      observer.observe(value.current);
    });

    return () => observer.disconnect();
  }, []);

  return (
    <Layout
      title={siteConfig.title}
      description="Description will go into a meta tag in <head />"
    >
      <div className="container">
        <Spacer height={64} />
        <H1>
          ML and GenAI
          <br /> made simple
        </H1>
        <div
          style={{
            background:
              "radial-gradient(51.2% 156.44% at 48.65% 192.33%, #43C9ED 0%, rgba(67, 201, 237, 0.00) 100%)",
          }}
        >
          <div
            style={{
              textAlign: "center",
              fontSize: "1.5rem",
              width: "70%",
              margin: "auto",
              paddingTop: "32px",
              paddingBottom: "32px",
            }}
          >
            Build better models and generative AI apps on an unified,
            end-to-end,
            <br />
            open source MLOps platform
          </div>

          <HeroBadges />
        </div>
      </div>
      <Flow />
      <FloatingNav
        sections={Object.values(SECTIONS)}
        active={active}
        onClick={setActive}
      />
      <Spacer height={96} />

      <div className="container">
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: "3rem",
            gap: "1rem",
          }}
        >
          <div style={{ width: "100%" }}>
            <Line />
          </div>
          <div style={{ textAlign: "center", whiteSpace: "nowrap" }}>
            Join thousands of users worldwide
          </div>
          <div style={{ width: "100%" }}>
            <Line />
          </div>
        </div>
      </div>

      <AutoScroll
        images={[
          "img/companies/databricks.svg",
          "img/companies/microsoft.svg",
          "img/companies/meta.svg",
          "img/companies/mosaicml.svg",
          "img/companies/zillow.svg",
          "img/companies/toyota.svg",
          "img/companies/booking.svg",
          "img/companies/wix.svg",
          "img/companies/accenture.svg",
          "img/companies/asml.svg",
          // "img/companies/atlassian.svg",
          // "img/companies/samsara.svg",
          // "img/companies/scribd.svg",
          // "img/companies/headspace.svg",
          // "img/companies/grab.svg",
          // "img/companies/gousto.svg",
          // "img/companies/riot-games.svg",
          // "img/companies/nike.svg",
          // "img/companies/comcast.svg",
          // "img/companies/intuit.svg",
        ]}
      />

      <div className="container">
        <Spacer height={200} />
        <Anchor id={SECTIONS.coreConcepts} ref={sectionRefs.coreConcepts} />
        <H2>
          Run ML and generative AI projects that solve complex, real-world
          challenges
        </H2>
        <Spacer height={64} />
        <CenterGrid>
          <ConceptCard
            logo="img/concepts/experiment-tracking.svg"
            title="Experiment tracking"
            href="docs/latest/tracking.html"
          />
          <ConceptCard
            logo="img/concepts/visualization.svg"
            title="Visualization"
            href="/docs/latest/getting-started/quickstart-2/index.html#compare-the-results"
          />
          <ConceptCard
            logo="img/concepts/generative-ai.svg"
            title="Generative AI"
            href="docs/latest/llms/index.html"
          />
          <ConceptCard
            logo="img/concepts/evaluation.svg"
            title="Evaluation"
            href="docs/latest/model-evaluation/index.html"
          />
          <ConceptCard
            logo="img/concepts/models.svg"
            title="Models"
            href="docs/latest/models.html"
          />
          <ConceptCard
            logo="img/concepts/model-registry.svg"
            title="Model Registry"
            href="docs/latest/model-registry.html"
          />
          <ConceptCard
            logo="img/concepts/serving.svg"
            title="Serving"
            href="docs/latest/deployment/index.html"
          />
        </CenterGrid>

        <Spacer height={200} />
        <Anchor id={SECTIONS.benefits} ref={sectionRefs.benefits} />
        <H2>What makes MLflow different</H2>
        <Spacer height={64} />
        <Grid>
          <BenefitCard
            title="Open Source"
            body="Integrate with any ML library and platform"
          />
          <BenefitCard
            title="Comprehensive"
            body="Manage end-to-end ML and GenAI workflows, from development to production"
          />
          <BenefitCard
            title="Unified"
            body="Unified platform for both traditional ML and GenAI applications"
          />
        </Grid>

        <Spacer height={200} />
        <Anchor id={SECTIONS.features} ref={sectionRefs.features} />
        <Pills
          title={
            <H2>
              Streamline your entire ML and generative AI lifecycle in a dynamic
              landscape
            </H2>
          }
          items={[
            {
              title: "Generative AI",
              content: (
                <FeatureCard
                  items={[
                    "Improve generative AI quality",
                    "Build applications with prompt engineering",
                    "Track progress during fine tuning",
                    "Package and deploy models",
                    "Securely host LLMs at scale with MLflow Deployments",
                  ]}
                  img="img/media/generative-ai.png"
                  href="docs/latest/llms/index.html"
                />
              ),
            },
            {
              title: "Deep Learning",
              content: (
                <FeatureCard
                  items={[
                    "Native integrations with popular DL frameworks in use in industry (PyTorch, TensorFlow, Keras)",
                    "Simple, low-code performance tracking with autologging",
                    "State-of-the-art UI for deep learning model analysis and comparison",
                  ]}
                  img="img/media/deep-learning.png"
                  href="docs/latest/deep-learning/index.html"
                />
              ),
            },
            {
              title: "Traditional ML",
              content: (
                <FeatureCard
                  items={[
                    "End-to-end MLOps solution for traditional ML, including integrations with scikit-learn, XGBoost, PySpark, and more",
                    "Simple, low-code performance tracking with autologging",
                    "State-of-the-art UI for model analysis and comparison",
                  ]}
                  img="img/media/traditional-ml.png"
                  href="docs/latest/traditional-ml/index.html"
                />
              ),
            },
            {
              title: "Evaluation",
              content: (
                <FeatureCard
                  items={[
                    "Easily compare different ML models and GenAI application versions",
                    "Evaluate different prompts",
                    "Compare performance against a baseline to prevent regressions",
                    "Simplify and automate performance evaluation",
                  ]}
                  img="img/media/evaluation.png"
                  href="docs/latest/model-evaluation/index.html"
                />
              ),
            },
            {
              title: "Model Management",
              content: (
                <FeatureCard
                  items={[
                    "Package models for production, including code and dependencies",
                    "Catalog, govern, and manage model versions",
                    "Orchestrate model rollouts to staging and production",
                    "Deploy models for large scale batch and real-time inference",
                  ]}
                  img="img/media/model-management.png"
                  href="docs/latest/model-registry/index.html"
                />
              ),
            },
          ]}
        />

        <Spacer height={200} />
        <Anchor id={SECTIONS.integrations} ref={sectionRefs.integrations} />
        <H2>Run MLflow anywhere</H2>

        <div className="row">
          {[
            {
              title: "Databricks",
              src: "img/databricks.svg",
              href: "https://docs.databricks.com/en/mlflow/index.html",
            },
            {
              title: "Your cloud provider",
              src: "img/cloud.svg",
              href: "https://www.mlflow.org/docs/latest/tracking/tutorials/remote-server.html ",
            },
            {
              title: "Your datacenter",
              src: "img/datacenter.svg",
              href: "https://www.mlflow.org/docs/latest/tracking.html#common-setups ",
            },
            {
              title: "Your computer",
              src: "img/computer.svg",
              href: "https://www.mlflow.org/docs/latest/getting-started/intro-quickstart/index.html",
            },
          ].map(({ title, src, href }, index) => (
            <div className="col" key={index}>
              <LogoCard title={title} src={src} href={href} />
            </div>
          ))}
        </div>

        <Spacer height={96} />
        <div style={{ textAlign: "center" }}>
          <h2
            style={{
              color: "var(--ifm-color-success)",
              marginBottom: "48px",
            }}
          >
            MLflow integrates with these tools and platforms
          </h2>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
            gap: "24px",
          }}
        >
          {[
            {
              title: "PyTorch",
              src: "img/pytorch.svg",
              href: "docs/latest/python_api/mlflow.pytorch.html",
            },
            {
              title: "HuggingFace",
              src: "img/huggingface.svg",
              href: "docs/latest/python_api/mlflow.transformers.html",
            },
            {
              title: "OpenAI",
              src: "img/openai.svg",
              href: "docs/latest/python_api/openai/index.html",
            },
            {
              title: "LangChain",
              src: "img/langchain.svg",
              href: "docs/latest/python_api/mlflow.langchain.html",
            },
            {
              title: "Spark",
              src: "img/spark.svg",
              href: "docs/latest/python_api/mlflow.spark.html",
            },
            {
              title: "Keras",
              src: "img/keras.svg",
              href: "docs/latest/python_api/mlflow.keras_core.html",
            },
            {
              title: "TensorFlow",
              src: "img/tensorflow.svg",
              href: "docs/latest/python_api/mlflow.tensorflow.html",
            },
            {
              title: "Prophet",
              src: "img/prophet.svg",
              href: "docs/latest/python_api/mlflow.prophet.html",
            },
            {
              title: "scikit-learn",
              src: "img/scikit-learn.svg",
              href: "docs/latest/python_api/mlflow.sklearn.html",
            },
            {
              title: "XGBoost",
              src: "img/xgboost.svg",
              href: "docs/latest/python_api/mlflow.xgboost.html",
            },
            {
              title: "LightGBM",
              src: "img/lightgbm.svg",
              href: "docs/latest/python_api/mlflow.lightgbm.html",
            },
            {
              title: "CatBoost",
              src: "img/catboost.svg",
              href: "docs/latest/python_api/mlflow.catboost.html",
            },
          ].map(({ title, src, href }, index) => (
            <MiniLogoCard title={title} src={src} href={href} key={index} />
          ))}
        </div>

        <Spacer height={200} />
        <Anchor
          id={SECTIONS.learningResources}
          ref={sectionRefs.learningResources}
        />
        <Pills
          title={
            <H2>
              Get started with how-to guides, tutorials and everything you need
            </H2>
          }
          items={[
            {
              title: "LLMs",
              content: (
                <CardTile
                  cards={[
                    {
                      title: "Evaluating LLMs",
                      content: "Learn how to evaluate LLMs with MLflow",
                      href: "docs/latest/llms/llm-evaluate/index.html",
                      img: "img/learning/evaluating-llms_16_9.png",
                    },
                    {
                      title: "Using Custom PyFunc with LLMs",
                      content:
                        "Explore the nuances of packaging, customizing, and deploying advanced LLMs in MLflow using custom PyFuncs.",
                      href: "docs/latest/llms/custom-pyfunc-for-llms/index.html",
                      img: "img/learning/custom-pyfunc_16_9.png",
                    },
                    {
                      title: "Evaluation for RAG",
                      content:
                        "Learn how to evaluate Retrieval Augmented Generation applications by leveraging LLMs to generate a evaluation dataset and evaluate it using the built-in metrics in the MLflow Evaluate API.",
                      href: "docs/latest/llms/rag/index.html",
                      img: "img/learning/rag_16_9.png",
                    },
                  ]}
                />
              ),
            },
            {
              title: "Deep Learning",
              content: (
                <CardTile
                  cards={[
                    {
                      title: "Tensorflow",
                      content:
                        "Learn about MLflow's native integration with the Tensorflow library and see example notebooks that leverage MLflow and Tensorflow to build deep learning workflows.",
                      href: "docs/latest/deep-learning/tensorflow/index.html",
                      img: "img/learning/tensorflow_16_9.png",
                    },
                    {
                      title: "Keras",
                      content:
                        "Learn about MLflow's native integration with the Keras library and see example notebooks that leverage MLflow and Keras to build deep learning workflows.",
                      href: "docs/latest/deep-learning/keras/index.html",
                      img: "img/learning/keras_16_9.png",
                    },
                    {
                      title: "PyTorch",
                      content:
                        "Learn about MLflow's native integration with the PyTorch library and see example notebooks that leverage MLflow and PyTorch to build deep learning workflows.",
                      href: "docs/latest/deep-learning/pytorch/index.html",
                      img: "img/learning/pytorch_16_9.png",
                    },
                  ]}
                />
              ),
            },
            {
              title: "Traditional ML",
              content: (
                <CardTile
                  cards={[
                    {
                      title: "Hyperparameter Tuning with MLflow and Optuna",
                      content:
                        "Explore the integration of MLflow Tracking with Optuna for hyperparameter tuning. Dive into the capabilities of MLflow, understand parent-child run relationships, and compare different tuning runs to optimize model performance.",
                      href: "docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/index.html",
                      img: "img/learning/hyperparameter-tuning_16_9.png",
                    },
                    {
                      title: "Custom Pyfunc Models with MLflow",
                      content:
                        "Dive deep into the world of MLflow's Custom Pyfunc. Starting with basic model definitions, embark on a journey that showcases the versatility and power of Pyfunc.",
                      href: "docs/latest/traditional-ml/creating-custom-pyfunc/index.html",
                      img: "img/learning/custom-pyfunc-traditional_16_9.png",
                    },
                  ]}
                />
              ),
            },

            // {
            //   title: "MLflow UI",
            //   content: (
            //     <CardTile
            //       cards={[
            //         { title: "MLflow UI", content: "MLflow UI" },
            //         { title: "MLflow UI", content: "MLflow UI" },
            //         { title: "MLflow UI", content: "MLflow UI" },
            //       ]}
            //     />
            //   ),
            // },
            {
              title: "Tracking",
              content: (
                <CardTile
                  cards={[
                    {
                      title: "MLflow Tracking",
                      content:
                        "The MLflow Tracking is an API and UI for logging parameters, code versions, metrics, and output files when running your machine learning code and for later visualizing the results. MLflow Tracking provides Python, REST, R, and Java APIs.",
                      href: "docs/latest/tracking.html",
                      img: "img/learning/mlflow_16_9.png",
                    },
                    {
                      title: "MLflow Tracking Quickstart",
                      content:
                        "A great place to start to learn the fundamentals of MLflow Tracking! Learn in 5 minutes how to log, register, and load a model for inference.",
                      href: "docs/latest/getting-started/intro-quickstart/index.html",
                      img: "img/learning/mlflow_16_9.png",
                    },
                  ]}
                />
              ),
            },
            {
              title: "Deployment",
              content: (
                <CardTile
                  cards={[
                    {
                      title: "Deployment",
                      content:
                        "In the modern age of machine learning, deploying models effectively and consistently plays a pivotal role. The capability to serve predictions at scale, manage dependencies, and ensure reproducibility is paramount for businesses to derive actionable insights from their ML models.",
                      href: "docs/latest/deployment/index.html",
                      img: "img/learning/deployment_16_9.png",
                    },
                    {
                      title: "Deploying a Model to Kubernetes with MLflow",
                      content:
                        "Explore an end-to-end guide on using MLflow to train a linear regression model, package it, and deploy it to a Kubernetes cluster. Understand how MLflow simplifies the entire process, from training to serving.",
                      href: "docs/latest/deployment/deploy-model-to-kubernetes/index.html",
                      img: "img/learning/k8s_16_9.png",
                    },
                    // { title: "Deployment", content: "Deployments" },
                    // { title: "Deployment", content: "Deployments" },
                  ]}
                />
              ),
            },
          ]}
        />

        <Spacer height={200} />
        <div
          style={{
            background:
              "radial-gradient(27.46% 40.02% at 50% 63.2%, rgba(67, 201, 237, 0.30) 0%, rgba(67, 201, 237, 0.00) 100%)",
          }}
        >
          <H1>Join our growing community</H1>
          <div
            style={{
              textAlign: "center",
              fontSize: "1.5rem",
              width: "70%",
              margin: "auto",
            }}
          >
            14M+ monthly downloads
            <br />
            600+ contributors worldwide
          </div>
          <Spacer />
          <BrowserOnly>{() => <MLflowLogoAndCards />}</BrowserOnly>
        </div>

        <Spacer height={200} />
        <div style={{ textAlign: "center" }}>
          <h2
            style={{
              color: "var(--ifm-color-success)",
              marginBottom: "64px",
            }}
          >
            Latest release notes
          </h2>
        </div>
        <Grid>
          {latestReleases.map((release, idx) => (
            <ReleaseNote release={release} key={idx} />
          ))}
        </Grid>

        {/* Blog */}
        <Spacer height={200} />
        <div style={{ textAlign: "center" }}>
          <h2
            style={{
              color: "var(--ifm-color-primary)",
              marginBottom: "64px",
            }}
          >
            Latest blog posts
          </h2>
        </div>

        <Grid>
          {latestBlogs.map((blog, idx) => (
            <Blog blog={blog} key={idx} />
          ))}
        </Grid>

        <Spacer height={200} />
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
            gap: "16px",
          }}
        >
          <GetStarted text="Get Started with MLflow" />
          <ArrowText
            text={
              <a
                style={{ color: "inherit" }}
                href="https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md"
              >
                Learn how to contribute
              </a>
            }
          />
        </div>

        <Spacer height={200} />
      </div>

      <img src="img/prefooter-circles.svg" alt="" />
    </Layout>
  );
}
