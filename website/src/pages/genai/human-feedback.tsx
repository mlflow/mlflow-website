import {
  Layout,
  AboveTheFold,
  BelowTheFold,
  HeroImage,
  StickyGrid,
} from "../../components";
import CardHero from "@site/static/img/GenAI_humanfeedback/GenAI_humanfeedback_hero.png";
import Card1 from "@site/static/img/GenAI_humanfeedback/GenAI_humanfeedback_1.png";
import Card2 from "@site/static/img/GenAI_humanfeedback/GenAI_humanfeedback_2.png";
import Card3 from "@site/static/img/GenAI_humanfeedback/GenAI_humanfeedback_3.png";
import Card4 from "@site/static/img/GenAI_humanfeedback/GenAI_humanfeedback_4.png";

export default function HumanFeedback() {
  return (
    <Layout>
      <AboveTheFold
        sectionLabel="Human feedback"
        title="Incorporate human insight to understand and improve quality"
        body={[
          "Capture domain expert feedback to understand how your app should behave and align your custom LLM-judge metrics with those expert's judgement.",
          "Capture end user feedback to quickly pinpoint quality issues in production.",
        ]}
        hasGetStartedButton="/docs/latest/"
      >
        <HeroImage src={CardHero} alt="" />
      </AboveTheFold>

      <StickyGrid
        cards={[
          {
            title: "Intuitive labeling UIs for business users",
            body: "MLflow's Review App enables busy domain experts to quickly provide feedback on production logs. Share logs for review and use predefined or custom questions.",
            image: <img src={Card1} alt="" />,
            // Hybrid animation / product GIF of showing a trace in the trace UI, then animating it to go to the review app, and then seeing the review app in action (provide feedback clicked) and then animation to see it on the trace UI
          },
          {
            title: "Track and visualize feedback",
            body: "MLflow replaces spreadsheets by attaching expert/user feedback to traces as versioned labels. Visualize this data in MLflow Trace UIs and dashboards to swiftly identify quality issues.",
            image: <img src={Card2} alt="" />,
            // Animation of an app executing, producing a trace, having feedback attached to it, and then seeing the feedback in the trace UI
          },
          {
            title: "Capture end-user feedback",
            body: "MLflow scalable feedback APIs allow you to attach end-user feedback from your deployed app to the source MLflow Trace, so you debug negative feedback with access to the step-by-step execution.",
            image: <img src={Card3} alt="" />,
            // Product GIF of a fake production app and then seeing the feedback in the trace UI
          },
          {
            title: "Integrated Chat App",
            body: "Deploy new app versions to the Review App's chat UI. Domain experts can interact, give instant feedback, and help rapidly assess quality and pinpoint issues.",
            image: <img src={Card4} alt="" />,
            // Product GIF of the review app chat mode and then seeing the feedback in the trace UI
          },
        ]}
      />

      <BelowTheFold contentType="genai" />
    </Layout>
  );
}
