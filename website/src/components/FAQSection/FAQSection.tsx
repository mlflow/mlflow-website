import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Section } from "../Section/Section";

const faqs = [
  {
    question: "Is MLflow really free?",
    answer:
      "Yes! MLflow is 100% open source under the Apache 2.0 license. You can use it for any purpose, including commercial applications, without any licensing fees. The project is backed by the Linux Foundation, ensuring it remains open and community-driven.",
  },
  {
    question: "How does MLflow compare to other MLOps tools?",
    answer:
      "MLflow stands out with its vendor-neutral approach, comprehensive feature set, and massive community support. Unlike proprietary alternatives, MLflow doesn't lock you into any specific cloud or vendor. With 50M+ monthly downloads and 20K+ GitHub stars, it's the most widely adopted open-source MLOps platform.",
  },
  {
    question: "Can I use MLflow with my existing ML infrastructure?",
    answer:
      "Absolutely. MLflow integrates seamlessly with 30+ frameworks and tools including PyTorch, TensorFlow, scikit-learn, LangChain, OpenAI, and more. It works with any cloud provider (AWS, Azure, GCP, Databricks) or on-premises infrastructure.",
  },
  {
    question: "Is MLflow compatible with OpenTelemetry?",
    answer:
      "Yes, MLflow Tracing is fully compatible with OpenTelemetry, the industry standard for observability. This means you can easily integrate MLflow traces with your existing observability stack and avoid vendor lock-in.",
  },
  {
    question: "What LLM providers does MLflow support?",
    answer:
      "MLflow supports all major LLM providers including OpenAI, Anthropic, Google (Gemini), AWS Bedrock, Azure OpenAI, Mistral, Groq, and many more. You can also use any OpenAI-compatible API or self-hosted models through Ollama.",
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
  answer: string;
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
          <svg
            className="w-5 h-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
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
      body="Everything you need to know about MLflow. Can't find what you're looking for? Join our community."
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
