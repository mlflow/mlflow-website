import {
  Layout,
  Grid,
  GridItem,
  AboveTheFold,
  BelowTheFold,
  Card,
} from "../../components";
import AiGatewayImage from "@site/static/img/ai-gateway.png";
import ImprovedModelAccuracyImage from "@site/static/img/improved-model-accuracy.png";
import SpendingOversightImage from "@site/static/img/spending-oversight.png";

const FakeImage = () => (
  <div className="w-full aspect-[3/2] bg-black rounded-lg border border-[rgba(255,255,255,0.08)]"></div>
);

export default function AiGateway() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="AI gateway"
        title="Unified access to all AI models"
        body="Protects your data and GenAI deployments through centralized governance across all models."
        hasGetStartedButton
      >
        <img
          src={AiGatewayImage}
          alt="AI Gateway"
          className="w-full max-w-[800px] rounded-lg mx-auto"
        />
      </AboveTheFold>

      <Grid columns={2}>
        <GridItem>
          <Card
            title="Improved model accuracy"
            body="Elevate your model quality with our robust observability tools that capture detailed request and response data. Payload logging enables you to debug, fine-tune and enhance models, improving accuracy and reducing latency."
            image={
              <img
                src={ImprovedModelAccuracyImage}
                alt="Improved Model Accuracy"
                className="aspect-[3/2] object-cover rounded-lg"
              />
            }
          />
        </GridItem>
        <GridItem>
          <Card
            title="Spending oversight"
            body="With real-time insights into your AI operations, you can monitor expenses, optimize resource allocation and ensure efficient performance."
            image={
              <img
                src={SpendingOversightImage}
                alt="Spending Oversight"
                className="aspect-[3/2] object-cover rounded-lg"
              />
            }
          />
        </GridItem>
      </Grid>

      <BelowTheFold />
    </Layout>
  );
}
