import {
  Layout,
  Grid,
  GridItem,
  AboveTheFold,
  BelowTheFold,
  Card,
  HeroImage,
} from "../../components";
import CardHero from "@site/static/img/GenAI_governance/GenAI_governance_hero.png";
import Card1 from "@site/static/img/GenAI_governance/GenAI_governance_1.png";
import Card2 from "@site/static/img/GenAI_governance/GenAI_governance_2.png";

export default function AiGateway() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="AI gateway"
        title="Unified access to all AI models"
        body="Protects your data and GenAI deployments through centralized governance across all models."
        hasGetStartedButton
      >
        <HeroImage src={CardHero} alt="" />
      </AboveTheFold>

      <Grid columns={2}>
        <GridItem width="wide">
          <Card
            title="Improved model accuracy"
            body="Elevate your model quality with our robust observability tools that capture detailed request and response data. Payload logging enables you to debug, fine-tune and enhance models, improving accuracy and reducing latency."
            image={<img src={Card1} alt="Improved Model Accuracy" />}
          />
        </GridItem>
        <GridItem width="wide">
          <Card
            title="Spending oversight"
            body="With real-time insights into your AI operations, you can monitor expenses, optimize resource allocation and ensure efficient performance."
            image={<img src={Card2} alt="Spending Oversight" />}
          />
        </GridItem>
      </Grid>

      <BelowTheFold />
    </Layout>
  );
}
