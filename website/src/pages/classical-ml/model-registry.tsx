import Head from "@docusaurus/Head";
import {
  Layout,
  AboveTheFold,
  BelowTheFold,
  HeroImage,
  StickyGrid,
} from "../../components";
import CardHero from "@site/static/img/Classical_registry/classical_registry_hero.png";
import Card1 from "@site/static/img/Classical_registry/classical_registry_1.png";
import Card2 from "@site/static/img/Classical_registry/classical_registry_2.png";

const SEO_TITLE = "ML Model Registry | MLflow AI Platform";
const SEO_DESCRIPTION =
  "Manage your ML models in production with MLflow's model registry. Version control, approval workflows, and deployment management for machine learning models.";

export default function ModelRegistryAndDeployment() {
  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta
          property="og:url"
          content="https://mlflow.org/classical-ml/model-registry"
        />
        <link
          rel="canonical"
          href="https://mlflow.org/classical-ml/model-registry"
        />
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
            url: "https://mlflow.org/classical-ml/model-registry",
            license: "https://www.apache.org/licenses/LICENSE-2.0",
          })}
        </script>
      </Head>
      <Layout>
        <AboveTheFold
          sectionLabel="Model registry & deployment"
          title="Deploy and manage models in production"
          body="Streamline your ML workflows with MLflow's comprehensive model registry for version control, approvals, and deployment management."
          hasGetStartedButton="#get-started"
        >
          <HeroImage src={CardHero} alt="MLflow model registry screenshot" />
        </AboveTheFold>

        <StickyGrid
          cards={[
            {
              title: "Stage-based model lifecycle management",
              body: "Move models through customizable staging environments (Development, Staging, Production, or any stage alias you choose) with built-in approval workflow capabilities and automated notifications. Maintain complete audit trails of model transitions with detailed metadata about who approved changes and when they occurred.",
              image: <img src={Card1} alt="MLflow model registry screenshot" />,
            },
            {
              title: "Model deployment flexibility",
              body: "Deploy models as Docker containers, Python functions, REST endpoints, or directly to various serving platforms with MLflow's versatile deployment capabilities. Streamline the transition from development to production with consistent model behavior across any target environment, from local testing to cloud-based serving.",
              image: <img src={Card2} alt="MLflow model registry screenshot" />,
            },
          ]}
        />

        <BelowTheFold contentType="classical-ml" />
      </Layout>
    </>
  );
}
