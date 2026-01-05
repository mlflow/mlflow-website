import React from "react";
import { motion } from "motion/react";
import { Section } from "../Section/Section";

export const Challenges = () => {
  return (
    <div className="w-full py-24 relative overflow-hidden">
      {/* Ambient Background Effects - Subtle */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/4 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/3 rounded-full blur-3xl" />
      </div>

      <Section
        title="The Challenges"
        body="Building reliable AI applications comes with a unique set of hurdles."
      >
        <div className="flex flex-col gap-32 w-full max-w-6xl mx-auto relative z-10 mt-16">
          {/* Problem 1: Observability */}
          <ProblemSection
            title="Observability"
            description="See what LLMs and agents are doing and record them is important but hard. Gain deep insights into every step of your AI application's execution."
            align="left"
            color="blue"
            visual={<ObservabilityVisual />}
          />

          {/* Problem 3: Measuring */}
          <ProblemSection
            title="Measuring & Testing"
            description="Systematically measure the performance of the agents is critical for hillclimbing but not easy. Track metrics over time to ensure your models are improving."
            align="right"
            color="emerald"
            visual={<MeasuringVisual />}
          />

          {/* Problem 2: Reproducibility */}
          <ProblemSection
            title="Reproducibility"
            description="Reproducing AI application behavior across different environments and runs is challenging. Track all components, configurations, and dependencies to ensure consistent results."
            align="left"
            color="cyan"
            visual={<ReproducibilityVisual />}
          />
        </div>
      </Section>
    </div>
  );
};

const ProblemSection = ({
  title,
  description,
  align,
  color,
  visual,
}: {
  title: string;
  description: string;
  align: "left" | "right";
  color: "blue" | "cyan" | "emerald";
  visual: React.ReactNode;
}) => {
  const isLeft = align === "left";

  const colorConfig = {
    blue: {
      text: "text-blue-300/80",
      gradient: "from-blue-500/8 via-blue-400/4 to-transparent",
      glow: "shadow-blue-500/10",
      border: "border-white/10",
      accent: "bg-blue-400/60",
    },
    cyan: {
      text: "text-cyan-300/80",
      gradient: "from-cyan-500/8 via-cyan-400/4 to-transparent",
      glow: "shadow-cyan-500/10",
      border: "border-white/10",
      accent: "bg-cyan-400/60",
    },
    emerald: {
      text: "text-emerald-300/80",
      gradient: "from-emerald-500/8 via-emerald-400/4 to-transparent",
      glow: "shadow-emerald-500/10",
      border: "border-white/10",
      accent: "bg-emerald-400/60",
    },
  }[color];

  return (
    <div
      className={`flex flex-col md:flex-row items-center gap-12 md:gap-24 ${isLeft ? "" : "md:flex-row-reverse"}`}
    >
      {/* Text Content */}
      <motion.div
        className="flex-1 flex flex-col gap-6"
        initial={{ opacity: 0, x: isLeft ? -50 : 50 }}
        whileInView={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
        viewport={{ once: true, margin: "-100px" }}
      >
        {/* Title */}
        <h3 className="text-4xl md:text-6xl font-bold text-white">{title}</h3>

        {/* Description with better typography */}
        <p className="text-xl text-gray-300/90 leading-relaxed max-w-xl font-light">
          {description}
        </p>
      </motion.div>

      {/* Visual Content with Enhanced Effects */}
      <motion.div
        className="flex-1 w-full"
        initial={{ opacity: 0, scale: 0.9, x: isLeft ? 50 : -50 }}
        whileInView={{ opacity: 1, scale: 1, x: 0 }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}
        viewport={{ once: true, margin: "-100px" }}
      >
        <div className="relative group">
          {/* Main Card */}
          <div
            className={`relative aspect-video rounded-2xl border ${colorConfig.border} bg-gradient-to-br from-white/10 via-white/5 to-transparent backdrop-blur-xl overflow-hidden shadow-2xl transition-all duration-500`}
          >
            {/* Grid Pattern */}
            <div
              className="absolute inset-0 opacity-[0.02]"
              style={{
                backgroundImage: `linear-gradient(0deg, transparent 24%, rgba(255, 255, 255, .05) 25%, rgba(255, 255, 255, .05) 26%, transparent 27%, transparent 74%, rgba(255, 255, 255, .05) 75%, rgba(255, 255, 255, .05) 76%, transparent 77%, transparent), linear-gradient(90deg, transparent 24%, rgba(255, 255, 255, .05) 25%, rgba(255, 255, 255, .05) 26%, transparent 27%, transparent 74%, rgba(255, 255, 255, .05) 75%, rgba(255, 255, 255, .05) 76%, transparent 77%, transparent)`,
                backgroundSize: "50px 50px",
              }}
            />

            {/* Content */}
            <div className="absolute inset-0 flex items-center justify-center">
              {visual}
            </div>

            {/* Glass Overlay */}
            <div className="absolute inset-0 bg-gradient-to-tr from-white/5 via-transparent to-white/3 pointer-events-none" />
          </div>
        </div>
      </motion.div>
    </div>
  );
};

const ObservabilityVisual = () => {
  const traces = [
    { width: 85, label: "LLM Request", time: "12ms", status: "success" },
    { width: 65, label: "Vector Search", time: "8ms", status: "success" },
    { width: 92, label: "Prompt Template", time: "2ms", status: "success" },
    { width: 78, label: "Response Gen", time: "145ms", status: "success" },
  ];

  return (
    <div className="w-full h-full p-8 flex flex-col items-center justify-center bg-gradient-to-br from-black/60 via-black/40 to-transparent">
      {/* Header */}
      <motion.div
        className="w-full mb-6 flex items-center justify-between px-2"
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
          <span className="text-xs text-blue-300 font-semibold">
            Live Trace
          </span>
        </div>
        <span className="text-xs text-gray-400 font-mono">347ms total</span>
      </motion.div>

      {/* Trace Timeline */}
      <div className="flex flex-col gap-3 w-full">
        {traces.map((trace, i) => (
          <motion.div
            key={i}
            className="group"
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: i * 0.15 }}
          >
            <div className="flex items-center gap-3 mb-1">
              <span className="text-xs text-gray-400 w-28 truncate">
                {trace.label}
              </span>
              <span className="text-xs text-gray-500 font-mono ml-auto">
                {trace.time}
              </span>
            </div>
            <div className="relative h-2 bg-blue-950/30 rounded-full overflow-hidden">
              <motion.div
                className="absolute inset-y-0 left-0 bg-gradient-to-r from-blue-500 via-blue-400 to-blue-500 rounded-full"
                initial={{ width: "0%" }}
                whileInView={{ width: `${trace.width}%` }}
                transition={{
                  duration: 1.2,
                  delay: i * 0.15,
                  ease: [0.16, 1, 0.3, 1],
                }}
              >
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer" />
              </motion.div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Trace ID Card */}
      <motion.div
        className="mt-6 w-full p-4 rounded-xl border border-blue-500/30 bg-gradient-to-br from-blue-500/10 to-blue-500/5 backdrop-blur-sm group hover:border-blue-500/50 transition-colors"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.8 }}
      >
        <div className="flex items-center justify-between">
          <div className="flex flex-col gap-1">
            <span className="text-[10px] text-gray-400 uppercase tracking-wider">
              Trace ID
            </span>
            <span className="text-sm text-blue-300 font-mono font-semibold">
              8f9a2b4c...
            </span>
          </div>
          <div className="flex gap-1">
            {[...Array(3)].map((_, i) => (
              <div
                key={i}
                className="w-1 h-1 rounded-full bg-blue-400/50 group-hover:bg-blue-400 transition-colors"
              />
            ))}
          </div>
        </div>
      </motion.div>
    </div>
  );
};

const ReproducibilityVisual = () => {
  const nodes = [
    { x: 80, y: 112, size: 24, label: "Code", icon: "📦" },
    { x: 200, y: 80, size: 28, label: "Environment", icon: "⚙️" },
    { x: 200, y: 145, size: 22, label: "Data", icon: "📊" },
    { x: 320, y: 112, size: 24, label: "Results", icon: "✓" },
  ];

  const connections = [
    { from: { x: 104, y: 112 }, to: { x: 172, y: 90 } },
    { from: { x: 104, y: 112 }, to: { x: 178, y: 135 } },
    { from: { x: 228, y: 85 }, to: { x: 296, y: 108 } },
    { from: { x: 222, y: 147 }, to: { x: 296, y: 116 } },
  ];

  return (
    <div className="w-full h-full relative flex items-center justify-center p-8">
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 400 225">
        <defs>
          <marker
            id="arrow-cyan"
            markerWidth="10"
            markerHeight="10"
            refX="9"
            refY="3"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <path d="M0,0 L0,6 L9,3 z" fill="#22d3ee" />
          </marker>
          <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.2" />
            <stop offset="50%" stopColor="#22d3ee" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.2" />
          </linearGradient>
        </defs>

        {/* Connections with Animation */}
        {connections.map((conn, i) => (
          <motion.g key={i}>
            <motion.line
              x1={conn.from.x}
              y1={conn.from.y}
              x2={conn.to.x}
              y2={conn.to.y}
              stroke="url(#lineGradient)"
              strokeWidth="2"
              markerEnd="url(#arrow-cyan)"
              initial={{ pathLength: 0, opacity: 0 }}
              whileInView={{ pathLength: 1, opacity: 1 }}
              transition={{ duration: 1, delay: i * 0.2, ease: "easeInOut" }}
            />
            {/* Animated particle */}
            <motion.circle
              r="3"
              fill="#22d3ee"
              initial={{ x: conn.from.x, y: conn.from.y, opacity: 0 }}
              animate={{
                x: [conn.from.x, conn.to.x],
                y: [conn.from.y, conn.to.y],
                opacity: [0, 1, 1, 0],
              }}
              transition={{
                duration: 2,
                delay: i * 0.3,
                repeat: Infinity,
                repeatDelay: 1,
                ease: "easeInOut",
              }}
            />
          </motion.g>
        ))}

        {/* Nodes */}
        {nodes.map((node, i) => (
          <motion.g
            key={i}
            initial={{ opacity: 0, scale: 0 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{
              duration: 0.5,
              delay: i * 0.15,
              type: "spring",
              stiffness: 200,
            }}
          >
            <circle
              cx={node.x}
              cy={node.y}
              r={node.size}
              fill="#0e7490"
              stroke="#22d3ee"
              strokeWidth="2"
              opacity="0.9"
            />
            <circle
              cx={node.x}
              cy={node.y}
              r={node.size + 8}
              fill="none"
              stroke="#22d3ee"
              strokeWidth="1"
              opacity="0.3"
            />
          </motion.g>
        ))}
      </svg>

      {/* Labels */}
      <div className="absolute inset-0 pointer-events-none">
        {nodes.map((node, i) => (
          <motion.div
            key={i}
            className="absolute"
            style={{
              left: `${(node.x / 400) * 100}%`,
              top: `${(node.y / 225) * 100}%`,
              transform: "translate(-50%, -50%)",
            }}
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: i * 0.15 + 0.3 }}
          >
            <div className="flex flex-col items-center gap-1">
              <span className="text-2xl">{node.icon}</span>
              <span className="text-[10px] text-cyan-300 font-semibold whitespace-nowrap bg-black/50 px-2 py-0.5 rounded-full">
                {node.label}
              </span>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

const MeasuringVisual = () => {
  const metrics = [
    { value: 68, label: "Mon", trend: "+5%" },
    { value: 82, label: "Tue", trend: "+14%" },
    { value: 75, label: "Wed", trend: "-7%" },
    { value: 92, label: "Thu", trend: "+17%" },
    { value: 88, label: "Fri", trend: "-4%" },
    { value: 96, label: "Sat", trend: "+8%" },
  ];

  return (
    <div className="w-full h-full flex flex-col justify-between p-8 bg-gradient-to-br from-black/60 via-black/40 to-transparent">
      {/* Header */}
      <motion.div
        className="flex items-center justify-between mb-4"
        initial={{ opacity: 0, y: -10 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <div>
          <div className="text-xs text-gray-400 mb-1">Performance Score</div>
          <div className="text-2xl font-bold text-emerald-400">94.2%</div>
        </div>
        <div className="flex items-center gap-1 px-2 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30">
          <span className="text-emerald-400 text-xs">↗</span>
          <span className="text-emerald-400 text-xs font-semibold">+12%</span>
        </div>
      </motion.div>

      {/* Chart */}
      <div className="flex-1 flex items-end justify-between gap-3 relative">
        {/* Grid lines */}
        <div className="absolute inset-0 flex flex-col justify-between pointer-events-none">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="w-full h-px bg-white/5" />
          ))}
        </div>

        {/* Bars */}
        {metrics.map((metric, i) => (
          <motion.div
            key={i}
            className="flex-1 flex flex-col items-center gap-2 relative group cursor-pointer"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: i * 0.1 }}
          >
            {/* Trend indicator */}
            <motion.div
              className="absolute -top-6 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity"
              initial={{ y: 10 }}
              whileHover={{ y: 0 }}
            >
              <div className="px-2 py-1 rounded-md bg-emerald-500/20 border border-emerald-500/30 backdrop-blur-sm">
                <span className="text-[10px] text-emerald-300 font-semibold whitespace-nowrap">
                  {metric.trend}
                </span>
              </div>
            </motion.div>

            {/* Bar */}
            <div className="w-full relative" style={{ height: "140px" }}>
              <motion.div
                className="absolute bottom-0 w-full bg-gradient-to-t from-emerald-500 via-emerald-400 to-emerald-300 rounded-t-lg shadow-lg shadow-emerald-500/50 group-hover:shadow-emerald-500/70 transition-shadow overflow-hidden"
                initial={{ height: "0%" }}
                whileInView={{ height: `${metric.value}%` }}
                transition={{
                  duration: 1,
                  delay: i * 0.1,
                  ease: [0.16, 1, 0.3, 1],
                }}
              >
                {/* Shimmer effect */}
                <div className="absolute inset-0 bg-gradient-to-t from-transparent via-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

                {/* Value on hover */}
                <div className="absolute top-2 inset-x-0 flex justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                  <span className="text-xs font-bold text-white drop-shadow-lg">
                    {metric.value}%
                  </span>
                </div>
              </motion.div>
            </div>

            {/* Label */}
            <span className="text-xs text-gray-400 font-medium">
              {metric.label}
            </span>
          </motion.div>
        ))}
      </div>

      {/* Footer */}
      <motion.div
        className="flex items-center justify-center gap-2 mt-4 pt-4 border-t border-white/5"
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.8 }}
      >
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-emerald-400" />
          <span className="text-[10px] text-gray-400">Accuracy Score</span>
        </div>
        <span className="text-gray-600">•</span>
        <span className="text-[10px] text-gray-500">Last 7 days</span>
      </motion.div>
    </div>
  );
};
