import { useState, type ReactNode } from "react";
import { motion, AnimatePresence } from "motion/react";
import Link from "@docusaurus/Link";
import { Section } from "../Section/Section";
import { ChevronDown } from "lucide-react";

const faqs: { question: string; answer: ReactNode }[] = [
  {
    question: "What is MLflow?",
    answer: (
      <>
        MLflow is the largest{" "}
        <strong>open source AI engineering platform</strong> for{" "}
        <Link
          to="/genai"
          className="!text-white/70 !underline decoration-white/50 underline-offset-2 hover:decoration-white transition-all"
        >
          AI agents and LLM applications
        </Link>
        , and{" "}
        <Link
          to="/classical-ml"
          className="!text-white/70 !underline decoration-white/50 underline-offset-2 hover:decoration-white transition-all"
        >
          ML models
        </Link>
        . MLflow enables teams of all sizes to debug, evaluate, monitor, and
        optimize their AI applications while controlling costs and managing
        access to models and data. With
        over 30 million monthly downloads, thousands of organizations rely on
        MLflow each day to ship AI to production with confidence.
        <br />
        <br />
        MLflow's comprehensive feature set for agents and LLM applications
        includes production-grade{" "}
        <Link
          to="/genai/observability"
          className="!text-white/70 !underline decoration-white/50 underline-offset-2 hover:decoration-white transition-all"
        >
          observability
        </Link>
        ,{" "}
        <Link
          to="/genai/evaluations"
          className="!text-white/70 !underline decoration-white/50 underline-offset-2 hover:decoration-white transition-all"
        >
          evaluation
        </Link>
        ,{" "}
        <Link
          to="/prompt-registry"
          className="!text-white/70 !underline decoration-white/50 underline-offset-2 hover:decoration-white transition-all"
        >
          prompt management
        </Link>
        ,{" "}
        <Link
          to="/prompt-optimization"
          className="!text-white/70 !underline decoration-white/50 underline-offset-2 hover:decoration-white transition-all"
        >
          prompt optimization
        </Link>
        , an{" "}
        <Link
          to="/genai/ai-gateway"
          className="!text-white/70 !underline decoration-white/50 underline-offset-2 hover:decoration-white transition-all"
        >
          AI Gateway
        </Link>{" "}
        for managing costs and model access, and more. Learn more at{" "}
        <Link
          to="/genai"
          className="!text-white/70 !underline decoration-white/50 underline-offset-2 hover:decoration-white transition-all"
        >
          MLflow for LLMs and Agents
        </Link>
        .
        <br />
        <br />
        For machine learning (ML) model development, MLflow provides{" "}
        <Link
          to="/classical-ml/experiment-tracking"
          className="!text-white/70 !underline decoration-white/50 underline-offset-2 hover:decoration-white transition-all"
        >
          experiment tracking
        </Link>
        ,{" "}
        <Link
          to="/classical-ml/model-evaluation"
          className="!text-white/70 !underline decoration-white/50 underline-offset-2 hover:decoration-white transition-all"
        >
          model evaluation
        </Link>{" "}
        capabilities, a production{" "}
        <Link
          to="/classical-ml/model-registry"
          className="!text-white/70 !underline decoration-white/50 underline-offset-2 hover:decoration-white transition-all"
        >
          model registry
        </Link>
        , and{" "}
        <Link
          to="/classical-ml/serving"
          className="!text-white/70 !underline decoration-white/50 underline-offset-2 hover:decoration-white transition-all"
        >
          model deployment
        </Link>{" "}
        tools.
      </>
    ),
  },
  {
    question: "Why do I need an AI engineering platform like MLflow?",
    answer: (
      <>
        Getting an AI agent or LLM application to work in a demo is easy.
        Getting it to work reliably in production is a different problem
        entirely. Agents can take incorrect or destructive actions, leak
        sensitive data, generate harmful responses, burn through API budgets
        with unnecessary model calls, or unexpectedly degrade in quality over
        time. Overcoming these challenges requires full visibility into what
        your agents are doing, control over what they can access, and a
        systematic way to measure and improve their quality. MLflow provides the{" "}
        <Link
          to="/ai-observability"
          className="!text-white/70 !underline decoration-white/50 underline-offset-2 hover:decoration-white transition-all"
        >
          observability
        </Link>
        ,{" "}
        <Link
          to="/llm-evaluation"
          className="!text-white/70 !underline decoration-white/50 underline-offset-2 hover:decoration-white transition-all"
        >
          evaluation
        </Link>
        , and{" "}
        <Link
          to="/ai-gateway"
          className="!text-white/70 !underline decoration-white/50 underline-offset-2 hover:decoration-white transition-all"
        >
          governance
        </Link>{" "}
        capabilities you need to take your agents from prototype to production
        with confidence.
      </>
    ),
  },
  {
    question: "Is MLflow free?",
    answer:
      "Yes! MLflow is 100% open source under the Apache 2.0 license. You can use it for any purpose, including commercial applications, without any licensing fees. The project is backed by the Linux Foundation, ensuring it remains open and community-driven.",
  },
  {
    question: "How does MLflow compare to other LLMOps/MLOps tools?",
    answer:
      "Many LLMOps tools are either proprietary or only cover part of the lifecycle. MLflow gives you full stack MLOps and LLMOps capabilities in single open-source platform. Backed by the Linux Foundation, MLflow ensures your AI infrastructure remains open, vendor-neutral, and fully in your control.",
  },
  {
    question: "Can I use MLflow with my existing AI infrastructure?",
    answer:
      "Absolutely. MLflow works on any major cloud provider (AWS, Azure, GCP, Databricks) or on-premises infrastructure. Regardless of which platform and framework you use, MLflow can be used to track, evaluate, and deploy your AI projects.",
  },
  {
    question: "Do I need to use Python to use MLflow?",
    answer:
      "No, MLflow provides native SDKs for other languages such as TypeScript/JavaScript, Java, and R. If you use other languages, you can use MLflow's REST API and OpenTelemetry integrations to connect your projects to MLflow.",
  },
  {
    question: "Can I use MLflow in my enterprise organization?",
    answer:
      "Yes, MLflow is trusted by many enterprise organizations who operate at scale. If you don't want to manage MLflow infrastructure yourself, try managed MLflow services provided by Databricks, AWS SageMaker, Nebius or others.",
  },
];

function FAQItem({
  question,
  answer,
  isOpen,
  onClick,
  index,
}: {
  question: string;
  answer: ReactNode;
  isOpen: boolean;
  onClick: () => void;
  index: number;
}) {
  return (
    <motion.div
      className="border border-white/10 rounded-xl overflow-hidden"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4, delay: index * 0.1 }}
    >
      <button
        onClick={onClick}
        className="w-full flex items-center justify-between p-5 text-left bg-white/5 hover:bg-white/10 transition-colors"
      >
        <span className="text-white font-medium pr-4">{question}</span>
        <motion.span
          animate={{ rotate: isOpen ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="flex-shrink-0 text-white/60"
        >
          <ChevronDown className="w-5 h-5" />
        </motion.span>
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
          >
            <div className="p-5 pt-0 bg-white/5">
              <p className="text-white/70 text-sm leading-relaxed">{answer}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

export function FAQSection() {
  const [openIndex, setOpenIndex] = useState<number | null>(0);

  const handleClick = (index: number) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <Section
      title="Frequently Asked Questions"
      body={
        <>
          Visit our{" "}
          <Link
            to="/faq"
            className="!text-white/70 !underline decoration-white/50 underline-offset-2 hover:decoration-white transition-all"
          >
            FAQ page
          </Link>{" "}
          for everything you need to know about MLflow.
        </>
      }
      align="center"
      ambient
    >
      <div className="w-full max-w-3xl mx-auto space-y-4">
        {faqs.map((faq, index) => (
          <FAQItem
            key={index}
            question={faq.question}
            answer={faq.answer}
            isOpen={openIndex === index}
            onClick={() => handleClick(index)}
            index={index}
          />
        ))}
      </div>
    </Section>
  );
}
