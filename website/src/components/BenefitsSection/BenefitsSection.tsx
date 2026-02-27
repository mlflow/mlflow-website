import { type ReactNode } from "react";
import { motion } from "motion/react";
import { Section } from "../Section/Section";
import { LockOpen, Link, Zap, BarChart3, Users, Puzzle } from "lucide-react";

const benefits: {
  icon: ReactNode;
  title: string;
  description: string;
  color: string;
  iconBg: string;
  iconColor: string;
}[] = [
  {
    icon: <LockOpen className="w-6 h-6" />,
    title: "Open Source",
    description:
      "100% open source under Apache 2.0 license. Forever free, no strings attached.",
    color: "from-blue-500/20 to-blue-600/20",
    iconBg: "bg-blue-500/20",
    iconColor: "text-blue-400",
  },
  {
    icon: <Link className="w-6 h-6" />,
    title: "No Vendor Lock-in",
    description:
      "Works with any cloud, framework, or tool you use. Switch vendors anytime.",
    color: "from-purple-500/20 to-purple-600/20",
    iconBg: "bg-purple-500/20",
    iconColor: "text-purple-400",
  },
  {
    icon: <Zap className="w-6 h-6" />,
    title: "Production Ready",
    description:
      "Battle-tested at scale by Fortune 500 companies and thousands of teams.",
    color: "from-amber-500/20 to-amber-600/20",
    iconBg: "bg-amber-500/20",
    iconColor: "text-amber-400",
  },
  {
    icon: <BarChart3 className="w-6 h-6" />,
    title: "Full Visibility",
    description:
      "Complete tracking and observability for all your AI applications and agents.",
    color: "from-cyan-500/20 to-cyan-600/20",
    iconBg: "bg-cyan-500/20",
    iconColor: "text-cyan-400",
  },
  {
    icon: <Users className="w-6 h-6" />,
    title: "Community",
    description:
      "20K+ GitHub stars, 900+ contributors. Join the fastest-growing MLOps community.",
    color: "from-green-500/20 to-green-600/20",
    iconBg: "bg-green-500/20",
    iconColor: "text-green-400",
  },
  {
    icon: <Puzzle className="w-6 h-6" />,
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
