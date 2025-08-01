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

export default function Models() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="MLflow Models"
        title="Unified model packaging"
        body="A unified format to package, share, and deploy models across frameworks."
        hasGetStartedButton="#get-started"
      >
        <HeroImage src={CardHero} alt="" />
      </AboveTheFold>

      <StickyGrid
        cards={[
          {
            title: "Unified model format",
            body: "MLflow's MLModel file provides a standardized structure for packaging models from any framework, capturing essential dependencies and input/output specifications. This consistent packaging approach eliminates integration friction while ensuring models can be reliably deployed across any environment.",
            image: <img src={Card1} alt="" />,
          },
          {
            title: "Comprehensive model metadata",
            body: "Track crucial model requirements and artifacts including data schemas, preprocessing steps, and environment dependencies automatically with MLflow's metadata system. Create fully reproducible model packages that document the complete model context for simplified governance and troubleshooting.",
            image: <img src={Card2} alt="" />,
          },
        ]}
      />

      <BelowTheFold />
    </Layout>
  );
}
