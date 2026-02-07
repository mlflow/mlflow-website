import { motion } from "motion/react";
import { Section } from "../Section/Section";

const benefits = [
  {
    icon: (
      <svg
        className="w-6 h-6"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M8 11V7a4 4 0 118 0m-4 8v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2z"
        />
      </svg>
    ),
    title: "Open Source",
    description:
      "100% open source under Apache 2.0 license. Forever free, no strings attached.",
    color: "from-blue-500/20 to-blue-600/20",
    iconBg: "bg-blue-500/20",
    iconColor: "text-blue-400",
  },
  {
    icon: (
      <svg
        className="w-6 h-6"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"
        />
      </svg>
    ),
    title: "No Vendor Lock-in",
    description:
      "Works with any cloud, framework, or tool you use. Switch vendors anytime.",
    color: "from-purple-500/20 to-purple-600/20",
    iconBg: "bg-purple-500/20",
    iconColor: "text-purple-400",
  },
  {
    icon: (
      <svg
        className="w-6 h-6"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M13 10V3L4 14h7v7l9-11h-7z"
        />
      </svg>
    ),
    title: "Production Ready",
    description:
      "Battle-tested at scale by Fortune 500 companies and thousands of teams.",
    color: "from-amber-500/20 to-amber-600/20",
    iconBg: "bg-amber-500/20",
    iconColor: "text-amber-400",
  },
  {
    icon: (
      <svg
        className="w-6 h-6"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
        />
      </svg>
    ),
    title: "Full Visibility",
    description:
      "Complete tracking and observability for all your AI applications and agents.",
    color: "from-cyan-500/20 to-cyan-600/20",
    iconBg: "bg-cyan-500/20",
    iconColor: "text-cyan-400",
  },
  {
    icon: (
      <svg
        className="w-6 h-6"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
        />
      </svg>
    ),
    title: "Community",
    description:
      "20K+ GitHub stars, 900+ contributors. Join the fastest-growing MLOps community.",
    color: "from-green-500/20 to-green-600/20",
    iconBg: "bg-green-500/20",
    iconColor: "text-green-400",
  },
  {
    icon: (
      <svg
        className="w-6 h-6"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M11 4a2 2 0 114 0v1a1 1 0 001 1h3a1 1 0 011 1v3a1 1 0 01-1 1h-1a2 2 0 100 4h1a1 1 0 011 1v3a1 1 0 01-1 1h-3a1 1 0 01-1-1v-1a2 2 0 10-4 0v1a1 1 0 01-1 1H7a1 1 0 01-1-1v-3a1 1 0 00-1-1H4a2 2 0 110-4h1a1 1 0 001-1V7a1 1 0 011-1h3a1 1 0 001-1V4z"
        />
      </svg>
    ),
    title: "Integrations",
    description:
      "Works out of the box with LangChain, OpenAI, PyTorch, and 100+ AI frameworks.",
    color: "from-rose-500/20 to-rose-600/20",
    iconBg: "bg-rose-500/20",
    iconColor: "text-rose-400",
  },
];

export function BenefitsSection() {
  return (
    <Section
      title="Why Teams Choose MLflow"
      body="Focus on building great AI, not managing infrastructure. MLflow handles the complexity so you can ship faster."
      align="center"
    >
      <motion.div
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-px max-w-6xl mx-auto bg-white/10"
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
      >
        {benefits.map((benefit, index) => (
          <motion.div
            key={benefit.title}
            className="relative p-6 bg-[#0E1416]"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
          >
            {/* Icon */}
            <div
              className={`w-12 h-12 ${benefit.iconBg} ${benefit.iconColor} flex items-center justify-center mb-4`}
            >
              {benefit.icon}
            </div>

            {/* Title */}
            <h3 className="text-lg font-semibold text-white mb-2">
              {benefit.title}
            </h3>

            {/* Description */}
            <p className="text-sm text-white/60 leading-relaxed">
              {benefit.description}
            </p>
          </motion.div>
        ))}
      </motion.div>
    </Section>
  );
}
