import { MLFLOW_DOCS_URL } from "@site/src/constants";
import {
  Layout,
  Grid,
  GridItem,
  AboveTheFold,
  BelowTheFold,
  Card,
  HeroImage,
} from "../../components";
import CardHero from "@site/static/img/GenAI_app_versioning/GenAI_appversioning_hero.png";
import Card1 from "@site/static/img/GenAI_app_versioning/GenAI_appversioning_1.png";
import Card2 from "@site/static/img/GenAI_app_versioning/GenAI_appversioning_2.png";
import Card3 from "@site/static/img/GenAI_app_versioning/GenAI_appversioning_3.png";

export default function AppVersioning() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="App versioning"
        title="Manage app versions with ease"
        body="Track and compare different versions of GenAI applications to ensure quality and maintainability."
        hasGetStartedButton={MLFLOW_DOCS_URL}
      >
        <HeroImage src={CardHero} alt="" />
      </AboveTheFold>

      <Grid columns={2}>
        <GridItem width="wide">
          <Card
            title="Version tracking"
            body="Track different versions of your GenAI applications using LoggedModels. Link evaluation results, traces, and prompt versions to specific application versions. Optionally package application code for deployment and compare versions to understand performance impacts."
            image={<img src={Card1} alt="" />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Link evaluation results and traces to app versions"
            body="Automatically link evaluation metrics, outputs, and traces from `mlflow.genai.evaluate()` and autologging back to the specific LoggedModel version."
            image={<img src={Card2} alt="" />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Compare app versions"
            body="Compare different LoggedModel versions using metrics like performance, cost, and quality scores to make data-driven decisions."
            image={<img src={Card3} alt="" />}
          />
        </GridItem>
      </Grid>

      <BelowTheFold />
    </Layout>
  );
}
