import { useState } from "react";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Header } from "../components/Header/Header";
import { SocialLinksFooter } from "../components/SocialLinksFooter/SocialLinksFooter";
import {
  MLFLOW_DOCS_URL,
  MLFLOW_GENAI_DOCS_URL,
  MLFLOW_DBX_TRIAL_URL,
} from "@site/src/constants";

const SEO_TITLE = "Frequently Asked Questions | MLflow AI Platform";
const SEO_DESCRIPTION =
  "Get answers to common questions about MLflow, the open source AI platform for LLMs, agents, and ML models. Learn about features, pricing, integrations, and how MLflow compares to other tools.";

type FAQCategory = {
  title: string;
  id: string;
  faqs: {
    question: string;
    answer: React.ReactNode;
    answerText?: string;
  }[];
};

const faqCategories: FAQCategory[] = [
  {
    title: "About MLflow",
    id: "about",
    faqs: [
      {
        question: "What is MLflow?",
        answer: (
          <>
            MLflow is the largest{" "}
            <strong>open source AI engineering platform</strong>. MLflow enables
            teams of all sizes to debug, evaluate, monitor, and optimize
            production-quality{" "}
            <Link href="/genai">AI agents, LLM applications</Link>, and{" "}
            <Link href="/classical-ml">ML models</Link> while controlling costs
            and managing access to models and data. With over 30 million monthly
            downloads, thousands of organizations rely on MLflow each day to
            ship AI to production with confidence.
            <br />
            <br />
            MLflow's comprehensive feature set for agents and LLM applications
            includes production-grade{" "}
            <Link href="/ai-observability">observability</Link>,{" "}
            <Link href="/llm-evaluation">evaluation</Link>,{" "}
            <Link href="/genai/prompt-registry">prompt management</Link>, an{" "}
            <Link href="/ai-gateway">AI Gateway</Link> for managing costs and
            model access, and more. Learn more at{" "}
            <Link href="/genai">MLflow for LLMs and Agents</Link>.
            <br />
            <br />
            For machine learning (ML) model development, MLflow provides{" "}
            <Link href="/classical-ml/experiment-tracking">
              experiment tracking
            </Link>
            ,{" "}
            <Link href="/classical-ml/model-evaluation">model evaluation</Link>{" "}
            capabilities, a production{" "}
            <Link href="/classical-ml/model-registry">model registry</Link>, and{" "}
            <Link href="/classical-ml/serving">model deployment</Link> tools.
          </>
        ),
        answerText:
          "MLflow is the largest open source AI engineering platform. MLflow enables teams of all sizes to debug, evaluate, monitor, and optimize production-quality AI agents, LLM applications, and ML models while controlling costs and managing access to models and data. With over 30 million monthly downloads, thousands of organizations rely on MLflow each day to ship AI to production with confidence. MLflow's comprehensive feature set for agents and LLM applications includes production-grade observability, evaluation, prompt management, an AI Gateway for managing costs and model access, and more. For machine learning (ML) model development, MLflow provides experiment tracking, model evaluation capabilities, a production model registry, and model deployment tools.",
      },
      {
        question: "Why do I need an AI engineering platform like MLflow?",
        answer: (
          <>
            Getting an AI agent to work in a demo is easy. Getting it to work
            reliably in production is a different problem entirely. Agents can
            take incorrect or destructive actions, leak sensitive data, generate
            harmful responses, burn through API budgets with unnecessary model
            calls, or unexpectedly degrade in quality over time. Overcoming
            these challenges requires full visibility into what your agents are
            doing, control over what they can access, and a systematic way to
            measure and improve their quality. MLflow provides the{" "}
            <Link href="/ai-observability">observability</Link>,{" "}
            <Link href="/llm-evaluation">evaluation</Link>, and{" "}
            <Link href="/ai-gateway">governance</Link> capabilities you need to
            take your agents from prototype to production with confidence.
          </>
        ),
        answerText:
          "Getting an AI agent to work in a demo is easy. Getting it to work reliably in production is a different problem entirely. Agents can take incorrect or destructive actions, leak sensitive data, generate harmful responses, burn through API budgets with unnecessary model calls, or unexpectedly degrade in quality over time. Overcoming these challenges requires full visibility into what your agents are doing, control over what they can access, and a systematic way to measure and improve their quality. MLflow provides the observability, evaluation, and governance capabilities you need to take your agents from prototype to production with confidence.",
      },
      {
        question: "Is MLflow free?",
        answer: (
          <>
            Yes! MLflow is 100% open source under the Apache 2.0 license. You
            can use it for any purpose, including commercial applications,
            without any licensing fees. There are no per-seat fees, no usage
            limits, and no vendor lock-in. The project is hosted on{" "}
            <Link href="https://github.com/mlflow/mlflow">GitHub</Link> and
            backed by the{" "}
            <Link href="https://lfaidata.foundation/">Linux Foundation</Link>,
            ensuring it remains open and community-driven.
          </>
        ),
        answerText:
          "Yes! MLflow is 100% open source under the Apache 2.0 license. You can use it for any purpose, including commercial applications, without any licensing fees. There are no per-seat fees, no usage limits, and no vendor lock-in. The project is hosted on GitHub and backed by the Linux Foundation, ensuring it remains open and community-driven.",
      },
      {
        question: "How do I get started with MLflow?",
        answer: (
          <>
            The fastest way to get started depends on your use case. For{" "}
            <strong>LLMs and AI agents</strong>, install MLflow with{" "}
            <code>pip install mlflow</code> and enable automatic tracing with a
            single line of code (e.g., <code>mlflow.openai.autolog()</code>).
            See the{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "getting-started/"}>
              GenAI getting started guide
            </Link>
            . For <strong>machine learning</strong>, MLflow's autologging
            captures experiments automatically for popular frameworks like
            scikit-learn, PyTorch, and TensorFlow. See the{" "}
            <Link href={MLFLOW_DOCS_URL + "ml/getting-started/"}>
              ML getting started guide
            </Link>
            .
          </>
        ),
        answerText:
          "The fastest way to get started depends on your use case. For LLMs and AI agents, install MLflow with pip install mlflow and enable automatic tracing with a single line of code (e.g., mlflow.openai.autolog()). For machine learning, MLflow's autologging captures experiments automatically for popular frameworks like scikit-learn, PyTorch, and TensorFlow.",
      },
      {
        question:
          "What frameworks and programming languages does MLflow support?",
        answer: (
          <>
            MLflow supports{" "}
            <strong>
              all major AI/ML frameworks, LLMs, models, and languages
            </strong>
            . It provides native SDKs for Python, TypeScript/JavaScript, Java,
            and R, and its{" "}
            <Link href={MLFLOW_DOCS_URL + "api_reference/rest-api.html"}>
              REST API
            </Link>{" "}
            and OpenTelemetry integrations work with any programming language.
            For AI and ML frameworks, MLflow integrates with all{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL + "tracing/integrations/#model-providers"
              }
            >
              LLM providers
            </Link>{" "}
            (OpenAI, Anthropic, Gemini, Bedrock, DeepSeek, and more),{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/integrations/"}>
              agent frameworks
            </Link>{" "}
            (LangChain, LangGraph, ADK, DSPy, CrewAI, and more), and ML
            libraries (scikit-learn, PyTorch, TensorFlow, XGBoost, Spark ML,
            HuggingFace Transformers, and more). If it's a language, AI
            framework, or LLM provider, MLflow certainly supports it.
          </>
        ),
        answerText:
          "MLflow supports all major AI/ML frameworks and languages. It provides native SDKs for Python, TypeScript/JavaScript, Java, and R, and its REST API and OpenTelemetry integrations work with any programming language. For AI and ML frameworks, MLflow integrates with all major LLM providers, agent frameworks, and ML libraries. If it's a language, framework, or provider used in AI/ML, MLflow almost certainly supports it.",
      },
      {
        question: "Can I use MLflow with my existing AI infrastructure?",
        answer: (
          <>
            Absolutely. MLflow works on any major cloud provider (AWS, Azure,
            GCP, Databricks) or on-premises infrastructure. You can{" "}
            <Link href={MLFLOW_DOCS_URL + "self-hosting/"}>
              self-host MLflow
            </Link>{" "}
            or use managed services. Regardless of which platform and framework
            you use, MLflow can be used to track, evaluate, and deploy your AI
            projects.
          </>
        ),
        answerText:
          "Absolutely. MLflow works on any major cloud provider (AWS, Azure, GCP, Databricks) or on-premises infrastructure. You can self-host MLflow or use managed services. Regardless of which platform and framework you use, MLflow can be used to track, evaluate, and deploy your AI projects.",
      },
      {
        question: "Can I use MLflow in my enterprise organization?",
        answer: (
          <>
            Yes, MLflow is trusted by many enterprise organizations who operate
            at scale. If you don't want to manage MLflow infrastructure
            yourself, try{" "}
            <Link href={MLFLOW_DBX_TRIAL_URL}>
              managed MLflow on Databricks
            </Link>
            ,{" "}
            <Link href="https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html">
              AWS SageMaker
            </Link>
            ,{" "}
            <Link href="https://nebius.com/services/managed-mlflow">
              Nebius
            </Link>
            , or others.
          </>
        ),
        answerText:
          "Yes, MLflow is trusted by many enterprise organizations who operate at scale. If you don't want to manage MLflow infrastructure yourself, try managed MLflow on Databricks, AWS SageMaker, Nebius, or others.",
      },
      {
        question: "Does MLflow support OpenTelemetry?",
        answer: (
          <>
            Yes.{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
              MLflow's tracing
            </Link>{" "}
            is fully compatible with OpenTelemetry, so you can export traces to
            any OpenTelemetry-compatible backend. This gives you total ownership
            and portability of your trace data without vendor lock-in.
          </>
        ),
        answerText:
          "Yes. MLflow's tracing is fully compatible with OpenTelemetry, so you can export traces to any OpenTelemetry-compatible backend. This gives you total ownership and portability of your trace data without vendor lock-in.",
      },
    ],
  },
  {
    title: "LLMs & AI Agents",
    id: "llms-agents",
    faqs: [
      {
        question:
          "What LLM providers and agent frameworks does MLflow support?",
        answer: (
          <>
            MLflow supports <strong>all major LLM providers</strong> including
            OpenAI, Anthropic, Google Gemini, AWS Bedrock, Azure OpenAI,
            Mistral, Cohere, Groq, DeepSeek, and many more. For agent
            frameworks, MLflow integrates with LangChain, LangGraph, OpenAI
            Agents SDK, Google ADK, DSPy, CrewAI, AutoGen, Pydantic AI, and
            others. It also supports coding assistants like Claude Code, Codex,
            and Cursor. See the{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/integrations/"}>
              full list of integrations
            </Link>
            .
          </>
        ),
        answerText:
          "MLflow supports all major LLM providers including OpenAI, Anthropic, Google Gemini, AWS Bedrock, Azure OpenAI, Mistral, Cohere, Groq, DeepSeek, and many more. For agent frameworks, MLflow integrates with LangChain, LangGraph, OpenAI Agents SDK, Google ADK, DSPy, CrewAI, AutoGen, Pydantic AI, and others. It also supports coding assistants like Claude Code, Codex, and Cursor.",
      },
      {
        question: "What can MLflow do for my LLM or agent application?",
        answer: (
          <>
            MLflow provides a comprehensive platform for building
            production-quality LLM and agent applications. Key capabilities
            include: <Link href="/genai/observability">observability</Link>{" "}
            (tracing every LLM call and agent step),{" "}
            <Link href="/genai/evaluations">evaluation</Link> (automated quality
            assessment with 70+ built-in LLM judges),{" "}
            <Link href="/genai/prompt-registry">prompt management</Link>{" "}
            (versioning and optimization),{" "}
            <Link href="/genai/ai-gateway">AI Gateway</Link> (managing costs and
            model access across providers), and{" "}
            <Link href="/genai/human-feedback">human feedback</Link> collection.
          </>
        ),
        answerText:
          "MLflow provides a comprehensive platform for building production-quality LLM and agent applications. Key capabilities include: observability (tracing every LLM call and agent step), evaluation (automated quality assessment with 70+ built-in LLM judges), prompt management (versioning and optimization), AI Gateway (managing costs and model access across providers), and human feedback collection.",
      },
      {
        question: "What is LLM tracing and why do I need it?",
        answer: (
          <>
            <Link href="/llm-tracing">LLM tracing</Link> is the practice of
            recording every step of your AI application's execution: prompts,
            completions, tool calls, retrieval results, token counts, latency,
            and costs. Unlike traditional logging, tracing captures the full
            execution context so you can debug hallucinations, optimize costs,
            and monitor quality in production. MLflow enables automatic tracing
            with a single line of code for 50+{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/integrations/"}>
              LLM providers and agent frameworks
            </Link>
            . See the{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/quickstart"}>
              tracing quickstart
            </Link>
            .
          </>
        ),
        answerText:
          "LLM tracing is the practice of recording every step of your AI application's execution: prompts, completions, tool calls, retrieval results, token counts, latency, and costs. Unlike traditional logging, tracing captures the full execution context so you can debug hallucinations, optimize costs, and monitor quality in production. MLflow enables automatic tracing with a single line of code for 50+ LLM providers and agent frameworks.",
      },
      {
        question:
          "How do I evaluate the quality of my LLM or agent application?",
        answer: (
          <>
            MLflow provides{" "}
            <Link href="/llm-evaluation">automated evaluation</Link> using
            LLM-as-a-judge scorers. You can use 70+ built-in judges that assess
            correctness, relevance, safety, groundedness (for RAG), and more.
            You can also create custom scorers for domain-specific quality
            criteria. MLflow supports evaluating single-turn responses,
            multi-turn conversations, and autonomous agent workflows. See the{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/quickstart"}>
              evaluation quickstart
            </Link>
            .
          </>
        ),
        answerText:
          "MLflow provides automated evaluation using LLM-as-a-judge scorers. You can use 70+ built-in judges that assess correctness, relevance, safety, groundedness (for RAG), and more. You can also create custom scorers for domain-specific quality criteria. MLflow supports evaluating single-turn responses, multi-turn conversations, and autonomous agent workflows.",
      },
      {
        question: "What is an AI Gateway and do I need one?",
        answer: (
          <>
            An <Link href="/ai-gateway">AI Gateway</Link> is a centralized proxy
            that sits between your application and LLM providers. It manages
            credentials, tracks usage and costs, enforces rate limits, provides
            fallback routing, and enables governance policies, all without
            changing your application code. If you use multiple LLM providers,
            want to control costs, or need to enforce compliance policies, an AI
            Gateway is essential. MLflow AI Gateway supports all major providers
            and integrates directly with MLflow's{" "}
            <Link href="/llm-tracing">tracing</Link> and{" "}
            <Link href="/llm-evaluation">evaluation</Link>.
          </>
        ),
        answerText:
          "An AI Gateway is a centralized proxy that sits between your application and LLM providers. It manages credentials, tracks usage and costs, enforces rate limits, provides fallback routing, and enables governance policies, all without changing your application code. If you use multiple LLM providers, want to control costs, or need to enforce compliance policies, an AI Gateway is essential.",
      },
      {
        question: "What is MLflow's Prompt Registry?",
        answer: (
          <>
            The{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "prompt-registry/"}>
              MLflow Prompt Registry
            </Link>{" "}
            lets you version, compare, and iterate on prompt templates directly
            through the MLflow UI or SDK. You can manage prompts separately from
            application code, reuse them across multiple versions of your agent,
            and view rich lineage showing which deployments use each prompt
            version. This makes it easy to optimize prompts without redeploying
            your application. You can also use{" "}
            <Link href="/prompt-optimization">
              automated prompt optimization
            </Link>{" "}
            to improve prompts algorithmically. See the{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "prompt-registry/create-and-edit-prompts"
              }
            >
              prompt management quickstart
            </Link>
            .
          </>
        ),
        answerText:
          "The MLflow Prompt Registry lets you version, compare, and iterate on prompt templates directly through the MLflow UI or SDK. You can manage prompts separately from application code, reuse them across multiple versions of your agent, and view rich lineage showing which deployments use each prompt version. This makes it easy to optimize prompts without redeploying your application.",
      },
      {
        question:
          "How does MLflow support human feedback for LLM applications?",
        answer: (
          <>
            MLflow's{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "assessments/feedback"}>
              human feedback
            </Link>{" "}
            capabilities let you collect annotations and quality assessments
            from domain experts directly within the MLflow UI. Reviewers can
            label traces, flag issues, and provide structured feedback that
            feeds into evaluation datasets and monitoring dashboards. This
            closes the loop between production observations and systematic
            quality improvement.
          </>
        ),
        answerText:
          "MLflow's human feedback capabilities let you collect annotations and quality assessments from domain experts directly within the MLflow UI. Reviewers can label traces, flag issues, and provide structured feedback that feeds into evaluation datasets and monitoring dashboards. This closes the loop between production observations and systematic quality improvement.",
      },
      {
        question: "What is the difference between LLMOps and MLOps?",
        answer: (
          <>
            <Link href="/llmops">LLMOps</Link> deals with challenges unique to
            LLMs: prompt management, non-deterministic outputs, token cost
            optimization, multi-step agent orchestration, retrieval-augmented
            generation, and evaluation with LLM judges.{" "}
            <Link href="/classical-ml">MLOps</Link> focuses on training,
            versioning, and deploying traditional machine learning models with
            static metrics. MLflow is the only open source platform that
            provides both full-stack LLMOps and MLOps capabilities in a single
            platform.
          </>
        ),
        answerText:
          "LLMOps deals with challenges unique to LLMs: prompt management, non-deterministic outputs, token cost optimization, multi-step agent orchestration, retrieval-augmented generation, and evaluation with LLM judges. MLOps focuses on training, versioning, and deploying traditional machine learning models with static metrics. MLflow is the only open source platform that provides both full-stack LLMOps and MLOps capabilities in a single platform.",
      },
    ],
  },
  {
    title: "Machine Learning",
    id: "machine-learning",
    faqs: [
      {
        question: "What can MLflow do for my ML projects?",
        answer: (
          <>
            MLflow provides the full machine learning lifecycle toolkit:{" "}
            <Link href="/classical-ml/experiment-tracking">
              experiment tracking
            </Link>{" "}
            (log parameters, metrics, and artifacts),{" "}
            <Link href="/classical-ml/hyperparam-tuning">
              hyperparameter tuning
            </Link>
            ,{" "}
            <Link href="/classical-ml/model-evaluation">model evaluation</Link>,
            a <Link href="/classical-ml/model-registry">model registry</Link>{" "}
            for version control and deployment management,{" "}
            <Link href="/classical-ml/models">unified model packaging</Link>{" "}
            across frameworks, and flexible{" "}
            <Link href="/classical-ml/serving">model serving</Link> for
            real-time and batch inference.
          </>
        ),
        answerText:
          "MLflow provides the full machine learning lifecycle toolkit: experiment tracking (log parameters, metrics, and artifacts), hyperparameter tuning, model evaluation, a model registry for version control and deployment management, unified model packaging across frameworks, and flexible model serving for real-time and batch inference.",
      },
      {
        question: "What ML frameworks does MLflow integrate with?",
        answer: (
          <>
            MLflow integrates with all major ML and deep learning frameworks
            including scikit-learn, PyTorch, TensorFlow, Keras, XGBoost, Spark
            ML, HuggingFace Transformers, and many more. MLflow's{" "}
            <Link href={MLFLOW_DOCS_URL + "ml/tracking/autolog/"}>
              autologging
            </Link>{" "}
            automatically captures experiments for these frameworks with no code
            changes required.
          </>
        ),
        answerText:
          "MLflow integrates with all major ML and deep learning frameworks including scikit-learn, PyTorch, TensorFlow, Keras, XGBoost, Spark ML, HuggingFace Transformers, and many more. MLflow's autologging automatically captures experiments for these frameworks with no code changes required.",
      },
      {
        question: "How does MLflow experiment tracking work?",
        answer: (
          <>
            MLflow{" "}
            <Link href="/classical-ml/experiment-tracking">
              experiment tracking
            </Link>{" "}
            lets you log parameters, metrics, artifacts, and code versions for
            every training run. You can compare runs side-by-side in the MLflow
            UI, search and filter by any metric or parameter, and reproduce past
            experiments. With{" "}
            <Link href={MLFLOW_DOCS_URL + "ml/tracking/autolog/"}>
              autologging
            </Link>
            , MLflow captures all of this automatically for popular frameworks
            with no code changes.
          </>
        ),
        answerText:
          "MLflow experiment tracking lets you log parameters, metrics, artifacts, and code versions for every training run. You can compare runs side-by-side in the MLflow UI, search and filter by any metric or parameter, and reproduce past experiments. With autologging, MLflow captures all of this automatically for popular frameworks with no code changes.",
      },
      {
        question: "How do I deploy my ML model with MLflow?",
        answer: (
          <>
            MLflow offers flexible{" "}
            <Link href="/classical-ml/serving">deployment options</Link>{" "}
            including local REST API serving, Docker containers, Kubernetes
            clusters, and cloud platforms like Databricks, AWS SageMaker, and
            Azure ML. MLflow's unified{" "}
            <Link href="/classical-ml/models">model packaging format</Link>{" "}
            ensures your model behaves consistently across any deployment
            target. For batch inference, you can deploy models directly on
            Apache Spark to process billions of predictions.
          </>
        ),
        answerText:
          "MLflow offers flexible deployment options including local REST API serving, Docker containers, Kubernetes clusters, and cloud platforms like Databricks, AWS SageMaker, and Azure ML. MLflow's unified model packaging format ensures your model behaves consistently across any deployment target. For batch inference, you can deploy models directly on Apache Spark to process billions of predictions.",
      },
      {
        question: "What is the MLflow Model Registry?",
        answer: (
          <>
            The{" "}
            <Link href="/classical-ml/model-registry">
              MLflow Model Registry
            </Link>{" "}
            is a centralized hub for managing the full lifecycle of your ML
            models. It provides version control, stage transitions (e.g.,
            Staging to Production), approval workflows, and deployment
            management. Teams use it to track which model versions are in
            production, who approved them, and when they were deployed.
          </>
        ),
        answerText:
          "The MLflow Model Registry is a centralized hub for managing the full lifecycle of your ML models. It provides version control, stage transitions (e.g., Staging to Production), approval workflows, and deployment management. Teams use it to track which model versions are in production, who approved them, and when they were deployed.",
      },
    ],
  },
  {
    title: "How MLflow Compares",
    id: "comparisons",
    faqs: [
      {
        question:
          "How does MLflow compare to other LLMOps and observability tools?",
        answer: (
          <>
            Many LLMOps tools are either proprietary or only cover part of the
            lifecycle. MLflow gives you full-stack LLMOps capabilities in a
            single open source platform:{" "}
            <Link href="/genai/observability">tracing</Link>,{" "}
            <Link href="/genai/evaluations">evaluation</Link>,{" "}
            <Link href="/genai/prompt-registry">prompt management</Link>,{" "}
            <Link href="/genai/ai-gateway">AI Gateway</Link>,{" "}
            <Link href="/genai/human-feedback">human feedback</Link>, and{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "serving/"}>agent serving</Link>
            . Unlike proprietary alternatives, MLflow supports any LLM provider
            and{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/integrations/"}>
              agent framework
            </Link>
            , is fully OpenTelemetry-compatible, and is backed by the Linux
            Foundation under the Apache 2.0 license, ensuring your AI
            infrastructure remains open, vendor-neutral, and fully in your
            control.
          </>
        ),
        answerText:
          "Many LLMOps tools are either proprietary or only cover part of the lifecycle. MLflow gives you full-stack LLMOps capabilities in a single open source platform: tracing, evaluation, prompt management, AI Gateway, human feedback, and agent serving. Unlike proprietary alternatives, MLflow supports any LLM provider and agent framework, is fully OpenTelemetry-compatible, and is backed by the Linux Foundation under the Apache 2.0 license.",
      },
      {
        question: "How does MLflow compare to other MLOps tools?",
        answer: (
          <>
            Most MLOps tools focus on a single part of the ML lifecycle. MLflow
            is unique in providing the complete toolkit:{" "}
            <Link href="/classical-ml/experiment-tracking">
              experiment tracking
            </Link>
            ,{" "}
            <Link href="/classical-ml/hyperparam-tuning">
              hyperparameter tuning
            </Link>
            ,{" "}
            <Link href="/classical-ml/model-evaluation">model evaluation</Link>,
            a <Link href="/classical-ml/model-registry">model registry</Link>,{" "}
            <Link href="/classical-ml/models">unified model packaging</Link>,
            and flexible <Link href="/classical-ml/serving">model serving</Link>
            , all in a single open source platform. It works with any ML
            framework, any cloud provider, and any deployment target. With over
            30 million monthly downloads, MLflow is by far the most widely
            adopted MLOps platform.
          </>
        ),
        answerText:
          "Most MLOps tools focus on a single part of the ML lifecycle. MLflow is unique in providing the complete toolkit: experiment tracking, hyperparameter tuning, model evaluation, a model registry, unified model packaging, and flexible model serving, all in a single open source platform. It works with any ML framework, any cloud provider, and any deployment target. With over 30 million monthly downloads, MLflow is by far the most widely adopted MLOps platform.",
      },
      {
        question:
          "How does MLflow compare to proprietary observability tools like Braintrust and LangSmith?",
        answer: (
          <>
            Unlike tools that charge per seat or per trace volume, MLflow is
            100% free and open source with no usage fees. You maintain complete
            control over your data and infrastructure. Getting started takes
            minutes. You can run MLflow on your laptop, deploy on any cloud, or
            use <Link href={MLFLOW_DBX_TRIAL_URL}>managed versions</Link> on
            Databricks, AWS, and others. MLflow supports any LLM provider and
            agent framework through{" "}
            <Link href="/ai-observability">
              OpenTelemetry-compatible tracing
            </Link>
            , so you're never locked into a single vendor's ecosystem. With over
            30 million monthly downloads and thousands of organizations in
            production, MLflow is the most widely adopted platform in the space.
          </>
        ),
        answerText:
          "Unlike tools that charge per seat or per trace volume, MLflow is 100% free and open source with no usage fees. You maintain complete control over your data and infrastructure. Getting started takes minutes. You can run MLflow on your laptop, deploy on any cloud, or use managed versions on Databricks, AWS, and others. MLflow supports any LLM provider and agent framework through OpenTelemetry-compatible tracing, so you're never locked into a single vendor's ecosystem. With over 30 million monthly downloads and thousands of organizations in production, MLflow is the most widely adopted platform in the space.",
      },
      {
        question:
          "Why choose MLflow over standalone tools like Langfuse or LiteLLM?",
        answer: (
          <>
            Tools like Langfuse (observability) and LiteLLM (AI Gateway) each
            solve one piece of the puzzle, but building production AI
            applications requires all the pieces to work together. With
            standalone tools, you end up stitching together separate systems for{" "}
            <Link href="/llm-tracing">tracing</Link>,{" "}
            <Link href="/llm-evaluation">evaluation</Link>,{" "}
            <Link href="/ai-gateway">gateway routing</Link>, and{" "}
            <Link href="/genai/prompt-registry">prompt management</Link>,
            creating data silos, duplicated configuration, and fragile
            integrations. MLflow is an{" "}
            <strong>integrated end-to-end platform</strong> where every
            capability works together out of the box: gateway requests
            automatically become traces, traces feed directly into evaluation,
            and evaluation results drive production monitoring. No glue code, no
            sync issues, no extra infrastructure. You get a unified UI, a
            unified data model, and a unified open source project backed by the{" "}
            <Link href="https://lfaidata.foundation/">Linux Foundation</Link>.
          </>
        ),
        answerText:
          "Tools like Langfuse (observability) and LiteLLM (AI Gateway) each solve one piece of the puzzle, but building production AI applications requires all the pieces to work together. With standalone tools, you end up stitching together separate systems for tracing, evaluation, gateway routing, and prompt management, creating data silos, duplicated configuration, and fragile integrations. MLflow is an integrated end-to-end platform where every capability works together out of the box: gateway requests automatically become traces, traces feed directly into evaluation, and evaluation results drive production monitoring. No glue code, no sync issues, no extra infrastructure. You get a unified UI, a unified data model, and a unified open source project backed by the Linux Foundation.",
      },
      {
        question:
          "Why should I use MLflow instead of a managed/proprietary platform?",
        answer: (
          <>
            <strong>No vendor lock-in:</strong> MLflow is{" "}
            <Link href="https://github.com/mlflow/mlflow">open source</Link>{" "}
            under Apache 2.0, backed by the{" "}
            <Link href="https://lfaidata.foundation/">Linux Foundation</Link>{" "}
            and not controlled by a single vendor.
            <br />
            <br />
            <strong>No per-seat or usage fees:</strong> Use all features
            (tracing, evaluation, prompt management, AI Gateway, model registry)
            for free, including in commercial applications.
            <br />
            <br />
            <strong>Data sovereignty:</strong> Your telemetry data, models, and
            prompts stay under your control. Deploy on-premises or on any cloud.
            <br />
            <br />
            <strong>Universal compatibility:</strong> Works with any{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/integrations/"}>
              LLM provider, agent framework
            </Link>
            , ML library, and programming language.
            <br />
            <br />
            <strong>Production proven:</strong> Over 30 million monthly
            downloads. Trusted by thousands of organizations from startups to
            Fortune 500 companies.
          </>
        ),
        answerText:
          "No vendor lock-in: MLflow is open source under Apache 2.0, backed by the Linux Foundation and not controlled by a single vendor. No per-seat or usage fees: Use all features for free, including in commercial applications. Data sovereignty: Your telemetry data, models, and prompts stay under your control. Universal compatibility: Works with any LLM provider, agent framework, ML library, and programming language. Production proven: Over 30 million monthly downloads, trusted by thousands of organizations.",
      },
      {
        question: "What is the best AI observability tool?",
        answer: (
          <>
            The best <Link href="/ai-observability">AI observability tool</Link>{" "}
            depends on your needs. MLflow is the leading open source option,
            with over 30 million monthly downloads. Thousands of organizations,
            developers, and research teams use MLflow each day to build and
            deploy production-grade agents and LLM applications. It offers
            complete <Link href="/llm-tracing">tracing</Link>,{" "}
            <Link href="/llm-evaluation">evaluation</Link>, and monitoring
            without vendor lock-in. Unlike proprietary tools that lock you into
            a vendor's ecosystem, MLflow is fully open source,
            OpenTelemetry-compatible, and supports any{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/integrations/"}>
              LLM provider or agent framework
            </Link>
            .
          </>
        ),
        answerText:
          "The best AI observability tool depends on your needs. MLflow is the leading open source option, with over 30 million monthly downloads. Thousands of organizations, developers, and research teams use MLflow each day to build and deploy production-grade agents and LLM applications. It offers complete tracing, evaluation, and monitoring without vendor lock-in. Unlike proprietary tools that lock you into a vendor's ecosystem, MLflow is fully open source, OpenTelemetry-compatible, and supports any LLM provider or agent framework.",
      },
      {
        question: "What is the best LLM evaluation tool?",
        answer: (
          <>
            The best <Link href="/llm-evaluation">LLM evaluation tool</Link>{" "}
            depends on your requirements. MLflow is the leading open source
            option, offering 70+{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "eval-monitor/scorers/llm-judge/predefined"
              }
            >
              built-in LLM judges
            </Link>
            , custom scorer APIs, evaluation datasets, version comparison, and
            production monitoring. Unlike proprietary tools that charge per
            evaluation or lock you into their ecosystem, MLflow is 100% free and
            open source. It supports any{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/integrations/"}>
              agent framework
            </Link>{" "}
            (LangGraph, CrewAI, ADK, Pydantic AI, etc.) and any LLM provider
            (OpenAI, Anthropic, Bedrock, etc.).
          </>
        ),
        answerText:
          "The best LLM evaluation tool depends on your requirements. MLflow is the leading open source option, offering 70+ built-in LLM judges, custom scorer APIs, evaluation datasets, version comparison, and production monitoring. Unlike proprietary tools that charge per evaluation or lock you into their ecosystem, MLflow is 100% free and open source. It supports any agent framework (LangGraph, CrewAI, ADK, Pydantic AI, etc.) and any LLM provider (OpenAI, Anthropic, Bedrock, etc.).",
      },
    ],
  },
];

const allFaqs = faqCategories.flatMap((cat) => cat.faqs);

const faqJsonLd = {
  "@context": "https://schema.org",
  "@type": "FAQPage",
  mainEntity: allFaqs.map((faq) => ({
    "@type": "Question",
    name: faq.question,
    acceptedAnswer: {
      "@type": "Answer",
      text: faq.answerText || faq.answer,
    },
  })),
};

const softwareJsonLd = {
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
  description:
    "Open source AI platform for LLMs, agents, and ML models. Debug, evaluate, monitor, and optimize production-quality AI applications.",
  url: "https://mlflow.org",
  license: "https://www.apache.org/licenses/LICENSE-2.0",
};

export default function FAQ() {
  const [openFaqIndex, setOpenFaqIndex] = useState<string | null>("about-0");

  const toggleFaq = (key: string) => {
    setOpenFaqIndex(openFaqIndex === key ? null : key);
  };

  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://mlflow.org/faq" />
        <link rel="canonical" href="https://mlflow.org/faq" />
        <script type="application/ld+json">{JSON.stringify(faqJsonLd)}</script>
        <script type="application/ld+json">
          {JSON.stringify(softwareJsonLd)}
        </script>
        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300;1,400;1,500&family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&display=swap');

          header, nav {
            background: #000000 !important;
          }

          body {
            background: #ffffff;
            margin: 0;
            padding: 0;
            font-family: 'DM Sans', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          }
          .faq-page {
            background: #ffffff;
            min-height: 100vh;
          }
          .faq-container {
            max-width: 900px;
            margin: 0 auto;
            padding: 60px 24px 100px;
          }
          .faq-container h1 {
            font-family: 'DM Sans', sans-serif;
            font-size: 3rem !important;
            font-weight: 700 !important;
            color: #1a1a1a !important;
            margin: 48px 0 16px 0 !important;
            line-height: 1.0 !important;
            letter-spacing: -0.03em !important;
          }
          .faq-container .subtitle {
            font-family: 'DM Sans', sans-serif;
            font-size: 18px;
            color: #6b7280;
            line-height: 1.6;
            margin: 0 0 20px 0;
          }
          .faq-layout {
            position: relative;
          }
          .faq-sidebar {
            position: fixed;
            top: 100px;
            left: calc(50% + 900px / 2 + 48px);
            width: 280px;
            max-height: calc(100vh - 120px);
            overflow-y: auto;
          }
          .faq-sidebar .toc-title {
            font-family: 'DM Sans', sans-serif;
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9ca3af;
            margin: 0 0 12px 0;
          }
          .faq-sidebar ul {
            margin: 0;
            padding: 0;
            list-style: none;
            border-left: 1px solid #e5e7eb;
          }
          .faq-sidebar li {
            margin: 0;
            padding: 0;
          }
          .faq-sidebar a {
            font-family: 'DM Sans', sans-serif;
            display: block;
            padding: 8px 0 8px 16px;
            font-size: 16px;
            color: #6b7280 !important;
            text-decoration: none !important;
            transition: all 0.15s ease;
            line-height: 1.4;
          }
          .faq-sidebar a:hover {
            color: #1a1a1a !important;
          }
          .faq-sidebar .toc-divider {
            border: none;
            border-top: 1px solid #e5e7eb;
            margin: 12px 0 12px 0;
          }
          @media (max-width: 1400px) {
            .faq-sidebar {
              display: none;
            }
          }
          .faq-container h2 {
            font-family: 'DM Sans', sans-serif;
            font-size: 1.75rem;
            font-weight: 700;
            color: #1a1a1a;
            margin: 48px 0 24px 0;
            line-height: 1.2;
            letter-spacing: -0.01em;
          }
          .faq-container a {
            color: #0194e2 !important;
            text-decoration: none;
            transition: all 0.2s ease;
          }
          .faq-container a:hover {
            color: #0072b0 !important;
            text-decoration: underline;
          }
          .faq-list {
            margin: 0 0 16px 0;
          }
          .faq-item {
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            margin-bottom: 12px;
            background: #ffffff;
            transition: all 0.2s ease;
          }
          .faq-item:hover {
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
          }
          .faq-question {
            font-family: 'DM Sans', sans-serif;
            padding: 20px 24px;
            font-size: 17px;
            font-weight: 500;
            color: #1a1a1a;
            cursor: pointer;
            background: transparent;
            border: none;
            width: 100%;
            text-align: left;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.2s ease;
          }
          .faq-question:hover {
            background: #f9fafb;
          }
          .faq-answer {
            font-family: 'DM Sans', sans-serif;
            padding: 0 24px 20px;
            font-size: 16px;
            color: #3d3d3d;
            line-height: 1.7;
          }
          .faq-answer code {
            font-family: 'DM Mono', monospace;
            background: #f3f4f6;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 14px;
            color: #1a1a1a;
          }
          .faq-chevron {
            transition: transform 0.2s ease;
            flex-shrink: 0;
            margin-left: 16px;
            color: #6b7280;
            font-size: 12px;
          }
          .faq-chevron.open {
            transform: rotate(180deg);
          }
          .learn-more-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin: 0 0 48px 0;
          }
          .learn-more-card {
            display: block;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            padding: 24px;
            text-decoration: none !important;
            transition: all 0.2s ease;
            background: #ffffff;
          }
          .learn-more-card:hover {
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            border-color: #0194e2;
          }
          .learn-more-card h3 {
            font-family: 'DM Sans', sans-serif;
            font-size: 17px;
            font-weight: 600;
            color: #1a1a1a !important;
            margin: 0 0 20px 0 !important;
          }
          .learn-more-card p {
            font-family: 'DM Sans', sans-serif;
            font-size: 14px;
            color: #6b7280 !important;
            line-height: 1.5;
            margin: 0;
          }
          @media (max-width: 768px) {
            .learn-more-grid {
              grid-template-columns: 1fr;
            }
            .faq-container {
              padding: 40px 20px 80px;
            }
            .faq-container h1 {
              font-size: 2.25rem !important;
              margin-bottom: 16px !important;
            }
            .faq-container h2 {
              font-size: 1.5rem;
              margin: 40px 0 20px 0;
            }
            .faq-question {
              padding: 18px 20px;
              font-size: 16px;
            }
            .faq-answer {
              padding: 0 20px 20px;
              font-size: 15px;
            }
          }
        `}</style>
      </Head>

      <div className="faq-page">
        <Header />

        <div className="faq-container">
          <h1>MLflow - Frequently Asked Questions</h1>
          <p className="subtitle">
            Answers to common questions about MLflow and what it can do for your
            AI and ML projects.
          </p>

          <div
            style={{
              margin: "0 0 32px 0",
              borderRadius: "8px",
              overflow: "hidden",
              border: "1px solid #e5e7eb",
            }}
          >
            <video width="100%" controls autoPlay loop muted playsInline>
              <source
                src={
                  require("@site/static/img/releases/3.10.0/demo-experiment.mp4")
                    .default
                }
                type="video/mp4"
              />
              Your browser does not support the video tag.
            </video>
          </div>

          <div className="faq-layout">
            {faqCategories.map((category) => (
              <div key={category.id}>
                <h2 id={category.id}>{category.title}</h2>
                <div className="faq-list">
                  {category.faqs.map((faq, index) => {
                    const key = `${category.id}-${index}`;
                    return (
                      <div key={key} className="faq-item">
                        <button
                          className="faq-question"
                          onClick={() => toggleFaq(key)}
                        >
                          <span>{faq.question}</span>
                          <span
                            className={`faq-chevron ${openFaqIndex === key ? "open" : ""}`}
                          >
                            ▼
                          </span>
                        </button>
                        {openFaqIndex === key && (
                          <div className="faq-answer">{faq.answer}</div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}

            <aside className="faq-sidebar">
              <p className="toc-title">On this page</p>
              <ul>
                {faqCategories.map((cat) => (
                  <li key={cat.id}>
                    <a href={`#${cat.id}`}>{cat.title}</a>
                  </li>
                ))}
                <li>
                  <a href="#learn-more">Learn More</a>
                </li>
              </ul>
              <hr className="toc-divider" />
              <p className="toc-title">Resources</p>
              <ul>
                <li>
                  <a href={MLFLOW_GENAI_DOCS_URL}>Documentation</a>
                </li>
                <li>
                  <a href="https://go.mlflow.org/slack">Slack</a>
                </li>
                <li>
                  <a href="https://github.com/mlflow/mlflow">GitHub</a>
                </li>
              </ul>
            </aside>
          </div>

          <h2 id="learn-more">Learn More</h2>
          <div className="learn-more-grid">
            <Link href="/ai-observability" className="learn-more-card">
              <h3>AI Observability</h3>
              <p>
                Monitor and debug AI agents and LLM applications with
                OpenTelemetry-compatible tracing.
              </p>
            </Link>
            <Link href="/llm-tracing" className="learn-more-card">
              <h3>LLM Tracing</h3>
              <p>
                Capture every step of your LLM and agent workflows with
                detailed, production-grade traces.
              </p>
            </Link>
            <Link href="/llm-evaluation" className="learn-more-card">
              <h3>LLM Evaluation</h3>
              <p>
                Systematically measure and improve AI quality with 70+ built-in
                LLM judges and scorers, as well as custom evaluators.
              </p>
            </Link>
            <Link href="/ai-gateway" className="learn-more-card">
              <h3>AI Gateway</h3>
              <p>
                Manage costs, enforce access controls, and route across LLM
                providers through a unified proxy.
              </p>
            </Link>
            <Link href="/llmops" className="learn-more-card">
              <h3>LLMOps</h3>
              <p>
                Operationalize LLM applications with end-to-end lifecycle
                management from development to production.
              </p>
            </Link>
            <Link href="/prompt-optimization" className="learn-more-card">
              <h3>Prompt Optimization</h3>
              <p>
                Automate prompt engineering with algorithms that systematically
                improve prompts using training data and LLM-driven analysis.
              </p>
            </Link>
            <Link href="/ai-monitoring" className="learn-more-card">
              <h3>AI Monitoring</h3>
              <p>
                Continuously evaluate quality, detect drift, and track costs for
                AI agents and LLM applications in production.
              </p>
            </Link>
            <Link href="/llm-optimization" className="learn-more-card">
              <h3>LLM Optimization</h3>
              <p>
                Reduce costs, improve quality, and lower latency for LLM
                applications with tracing, evaluation, and prompt optimization.
              </p>
            </Link>
            <Link href="/genai" className="learn-more-card">
              <h3>MLflow for LLMs & Agents</h3>
              <p>
                The complete platform for building production-quality AI agents
                and LLM applications.
              </p>
            </Link>
          </div>
        </div>

        <SocialLinksFooter />
      </div>
    </>
  );
}
