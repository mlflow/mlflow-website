import ReviewApp from "@site/static/img/review-app.jpg";
import TraceTab from "@site/static/img/trace-tab.jpg";
import EvalsTab from "@site/static/img/evals-tab.jpg";
import AssessmentsTab from "@site/static/img/assessments.jpg";

import {
  Layout,
  CopyCommand,
  VerticalTabs,
  VerticalTabsList,
  VerticalTabsTrigger,
  VerticalTabsContent,
  SectionLabel,
  LogosCarousel,
  Grid,
  GridItem,
  GetStartedWithMLflow,
  LatestNews,
  SocialWidget,
  Button,
} from "../../components";

export default function GenAi(): JSX.Element {
  return (
    <Layout variant="red">
      <div
        className="flex flex-col bg-[linear-gradient(to_bottom,rgba(12,20,20,0),rgba(14,20,20,100)),url('/img/background-image-2.png')]
 bg-top bg-no-repeat bg-cover w-full pt-42 pb-20 py-20"
      >
        <div className="flex flex-col gap-16 w-full px-6 md:px-20 max-w-container">
          <div className="flex flex-col justify-center items-center gap-6 w-full">
            <h1 className="text-center text-wrap">
              Ship high-quality AI, fast
            </h1>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white">
              Traditional software and ML tests aren't built for GenAI's
              free-form language, making it difficult for teams to measure and
              improve quality.
            </p>
            <p className="text-center text-wrap text-lg max-w-3xl w-full mx-auto text-white">
              MLflow combines metrics that reliably measure GenAI quality with
              trace observability so you can measure, improve, and monitor
              quality, cost, and latency.
            </p>
          </div>
          <div className="flex flex-col md:flex-row gap-10">
            <CopyCommand code="pip install mlflow" />
          </div>
        </div>
      </div>
      <div className="flex flex-col gap-40 w-full px-6 md:px-20 max-w-container">
        <div className="flex flex-col w-full items-center justify-center gap-16">
          <div className="flex flex-col w-full items-center justify-center gap-6">
            <SectionLabel color="red" label="CORE FEATURES" />
            <h1>Build confidently, deploy seamlessly</h1>
            <p className="text-white">
              Tackle the challenges of building GenAI head on
            </p>
          </div>
          <Grid columns={2}>
            <GridItem width="wide">
              <div className="flex flex-col gap-4">
                <h2 className="text-white">Tracing</h2>
                <h3 className="text-white">
                  Capture and debug application logs with end-to-end
                  observability
                </h3>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Trace your app with OpenTelemetry-compatible SDKs that capture
                  every invocation's inputs, outputs, and step-by-step execution
                  - prompts, retrievals, tool calls, and more - alongside cost,
                  latency, and errors. g{" "}
                </p>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Use these traces to quickly debug errors, improve business
                  logic, and optimize cost and latency.
                </p>
                <a href="/genai/observability">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </a>
              </div>
              <div>
                <img
                  className="rounded-xl"
                  src={TraceTab}
                  alt="MLflow tracing"
                />
              </div>
            </GridItem>
            <GridItem width="wide" direction="reverse">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  Adapt to evolving user behavior with production log evaluation
                </h3>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Use production logs to understand user behavior, turning
                  low-quality responses into evaluation datasets and
                  high-quality responses into regression tests.
                </p>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Replay these datasets to evaluate new prompts or app variants
                  in development so you can ship the best variants to ensure
                  your application continues to deliver high-quality responses
                  as user behavior evolves.
                </p>
                <a href="/genai/evaluations">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </a>
              </div>
              <div>
                <img
                  className="rounded-xl"
                  src={EvalsTab}
                  alt="MLflow evaluations"
                />
              </div>
            </GridItem>
            <GridItem direction="reverse">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  Measure and improve quality with human-aligned, automated
                  metrics
                </h3>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Capture and convert expert feedback into metrics (LLM judges)
                  that understand your business requirements and can measure the
                  nuances of plain-language GenAI outputs.
                </p>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Use these metrics to evaluate, monitor, and improve quality in
                  development and production at scale, without waiting for human
                  review.
                </p>
                <a href="/genai/quality-metrics">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </a>
              </div>
              <div>
                <img
                  className="rounded-xl mt-10 mb-4"
                  src={AssessmentsTab}
                  alt="MLflow LLM judges"
                />
              </div>
            </GridItem>
            <GridItem direction="reverse">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  Incorporate human insight with an intuitive labeling and
                  review experience
                </h3>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Capture domain expert feedback via web-based UIs and end-user
                  ratings from your app via APIs.
                </p>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Use this feedback to enrich your understanding of how the app
                  should behave and improve your custom LLM-judge metrics.
                </p>
                <a href="/genai/human-feedback">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </a>
              </div>
              <div>
                <img
                  className="rounded-xl mt-10 mb-4"
                  src={AssessmentsTab}
                  alt="MLflow LLM judges"
                />
              </div>
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  Continuously Monitor and Alert on Production AI Quality & Performance
                </h3>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  GenAI application behavior can drift in production due to changing data patterns or user interactions, potentially degrading quality or increasing costs. MLflow enables you to continuously monitor key quality, cost, and latency metrics for your deployed GenAI applications.
                </p>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Set up automated alerts to be notified of issues in real-time, allowing you to proactively address problems, maintain high performance, and ensure a consistent user experience.
                </p>
                <a href="/genai/monitoring"> {/* Placeholder link */}
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </a>
              </div>
              <div>
                <img
                  className="rounded-xl mt-10 mb-4"
                  src={AssessmentsTab} /* Placeholder image */
                  alt="MLflow Monitoring"
                />
              </div>
            </GridItem>
            <GridItem direction="reverse">
              <div className="flex flex-col gap-4">
                <h3 className="text-white">
                  Secure and Streamline LLM Access with AI Gateway
                </h3>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Organizations often struggle to securely manage and standardize access to various Large Language Models (LLMs) at scale. MLflow's AI Gateway provides a centralized and secure interface for managing LLM API keys and endpoints.
                </p>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Control who has access to specific LLMs, enforce rate limits, and leverage AI guardrails to protect your applications from exposing sensitive data or generating inappropriate responses, ensuring both security and compliance. (Managed MLflow feature)
                </p>
                <a href="/genai/ai-gateway"> {/* Placeholder link */}
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </a>
              </div>
              <div>
                <img
                  className="rounded-xl mt-10 mb-4"
                  src={AssessmentsTab} /* Placeholder image */
                  alt="MLflow AI Gateway"
                />
              </div>
            </GridItem>
            <GridItem className="hidden md:block"></GridItem>
            <GridItem>
              <div className="flex flex-col gap-4 justify-center">
                <h3 className="text-white">
                  Comprehensive GenAI Lifecycle Management with Unity Catalog and Git Integration
                </h3>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Tracking GenAI assets like prompts, their versions, and associated quality evaluations can be challenging with traditional tools. MLflow, integrated with Unity Catalog, provides a robust Prompt Registry to manage the lifecycle of your prompts.
                </p>
                <p style={{ marginBottom: 0 }} className="text-white/60">
                  Furthermore, MLflow complements Git by linking evaluation data, traces, and specific prompt versions from Unity Catalog directly to your Git revisions. This offers end-to-end versioning and lineage for your entire GenAI application, ensuring reproducibility and streamlined collaboration. It also tracks models, datasets, and metrics, applying access controls for enterprise governance.
                </p>
                <a href="/genai/governance">
                  <Button variant="outline" size="small">
                    Learn more &gt;
                  </Button>
                </a>
              </div>
            </GridItem>
          </Grid>
        </div>
        <LogosCarousel />
        <div className="flex flex-col items-center justify-center gap-16">
          <div className="flex flex-col gap-6">
            <SectionLabel color="red" label="WHY US?" />
            <h1>Why MLflow is unique</h1>
          </div>
          <Grid columns={3}>
            <GridItem>
              <div className="flex flex-col gap-4 h-full">
                <h2 className="text-white">Unified, End-to-End MLOps and AI Observability</h2>
                <p className="text-white/60">
                  Streamline your AI and ML workflows with a cohesive platform that standardizes the entire development and deployment lifecycle for your cutting-edge generative AI applications and machine learning models. MLflow provides robust support for generative AI tracing, prompt management, rich LLM-as-a-judge evaluations, and continuous AI quality monitoring for production deployments. For classical ML, MLflow offers comprehensive experiment tracking, model packaging, collaborative model management, and deployment solutions for batch and real-time inference. This unified platform delivers a consistent, simplified experience, accelerating collaboration and innovation by removing the friction of integrating multiple tools.
                </p>
              </div>
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4 h-full">
                <h2 className="text-white">Enterprise-Grade Security &amp; Governance on a Unified Data &amp; AI Platform</h2>
                <p className="text-white/60">
                  Managed MLflow is powered by Databricks, built on an enterprise data platform (the Databricks Lakehouse), so you can trust it with your most sensitive data without the burden of managing a self-hosted deployment. As a core component of the Databricks Data Intelligence Platform, it offers deep integrations with the broader Mosaic AI product suite, enabling high-quality training datasets, accelerated model development, powerful RAG applications, scalable Model Serving, and secure AI Gateway deployments. All of this is underpinned by Databricks Unity Catalog, giving you centralized, enterprise-grade governance over every AI asset—from data and models to agents and applications—ensuring compliance and control. Furthermore, leverage your GenAI and ML data for downstream business processes by building rich performance dashboards, reports, and queries with Databricks AI/BI and Databricks SQL.
                </p>
              </div>
            </GridItem>
            <GridItem>
              <div className="flex flex-col gap-4 h-full">
                <h2 className="text-white">Open, Flexible, and Extensible</h2>
                <p className="text-white/60">
                  MLflow is open source, allowing you to maintain agility and avoid vendor lock-in. Its extensible architecture integrates with the whole GenAI and ML ecosystem, enabling you to use your preferred frameworks, libraries, and languages. Data is stored and transmitted according to open protocols, giving you ownership of your data. Managed MLflow seamlessly integrates with a rich ecosystem of data sources, ML/AI tools, and deployment solutions, ensuring it can adapt to your existing stack and evolve with your future needs.
                </p>
              </div>
            </GridItem>
          </Grid>
        </div>
        <GetStartedWithMLflow />
        <LatestNews variant="red" />
        <SocialWidget variant="red" />
      </div>
    </Layout>
  );
}
