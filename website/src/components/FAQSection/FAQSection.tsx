import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Section } from "../Section/Section";

const faqs = [
  {
    question: "Is MLflow free?",
    answer:
      "Yes! MLflow is 100% open source under the Apache 2.0 license. You can use it for any purpose, including commercial applications, without any licensing fees. The project is backed by the Linux Foundation, ensuring it remains open and community-driven.",
  },
  {
    question: "How does MLflow compare to other MLOps/LLMOps tools?",
    answer:
      "MLflow stands out with its vendor-neutral approach, comprehensive feature set, and massive community support. Unlike proprietary alternatives, MLflow doesn't lock you into any specific cloud or vendor. With 50M+ monthly downloads and 20K+ GitHub stars, it's the most widely adopted open-source MLOps platform.",
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
  }
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
