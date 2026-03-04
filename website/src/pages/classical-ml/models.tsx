import Head from "@docusaurus/Head";
import {
  Layout,
  AboveTheFold,
  BelowTheFold,
  HeroImage,
  StickyGrid,
} from "../../components";
import CardHero from "@site/static/img/Classical_models/classical_models_hero.png";
import Card1 from "@site/static/img/Classical_models/classical_models_1.png";
import Card2 from "@site/static/img/Classical_models/classical_models_2.png";

const SEO_TITLE = "ML Model Packaging | MLflow AI Platform";
const SEO_DESCRIPTION =
  "Package, share, and deploy ML models across frameworks with MLflow's unified model format. Simplify machine learning model packaging and distribution.";

export default function Models() {
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
          content="https://mlflow.org/classical-ml/models"
        />
        <link rel="canonical" href="https://mlflow.org/classical-ml/models" />
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
            url: "https://mlflow.org/classical-ml/models",
            license: "https://www.apache.org/licenses/LICENSE-2.0",
          })}
        </script>
      </Head>
      <Layout>
        <AboveTheFold
          sectionLabel="MLflow Models"
          title="Unified model packaging"
          body="A unified format to package, share, and deploy models across frameworks."
          hasGetStartedButton="#get-started"
        >
          <HeroImage src={CardHero} alt="MLflow model packaging screenshot" />
        </AboveTheFold>

        <StickyGrid
          cards={[
            {
              title: "Unified model format",
              body: "MLflow's MLModel file provides a standardized structure for packaging models from any framework, capturing essential dependencies and input/output specifications. This consistent packaging approach eliminates integration friction while ensuring models can be reliably deployed across any environment.",
              image: (
                <img src={Card1} alt="MLflow model packaging screenshot" />
              ),
            },
            {
              title: "Comprehensive model metadata",
              body: "Track crucial model requirements and artifacts including data schemas, preprocessing steps, and environment dependencies automatically with MLflow's metadata system. Create fully reproducible model packages that document the complete model context for simplified governance and troubleshooting.",
              image: (
                <img src={Card2} alt="MLflow model packaging screenshot" />
              ),
            },
          ]}
        />

        <BelowTheFold contentType="classical-ml" />
      </Layout>
    </>
  );
}
