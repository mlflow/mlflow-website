import { ReactNode } from "react";
import { motion } from "motion/react";
import Link from "@docusaurus/Link";
import { Button } from "../Button/Button";

type Props = {
  title: ReactNode;
  subtitle: ReactNode;
  primaryCTA: {
    label: string;
    href: string;
  };
  secondaryCTA?: {
    label: string;
    href: string;
    icon?: ReactNode;
  };
};

export function HeroSection({
  title,
  subtitle,
  primaryCTA,
  secondaryCTA,
}: Props) {
  return (
    <div className="w-full px-4 md:px-8 lg:px-16 pt-16 md:pt-24">
      <div className="max-w-5xl mx-auto text-center flex flex-col items-center gap-6">
        {/* Title */}
        <motion.h1
          className="text-balance font-light text-[40px] xxs:text-[52px] xs:text-[72px] leading-[100%] tracking-[-3%] text-white text-center !border-none !m-0"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          {title}
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          className="text-lg md:text-xl text-white/70 max-w-2xl mx-auto"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          {subtitle}
        </motion.p>

        {/* CTA Buttons */}
        <motion.div
          className="flex flex-wrap justify-center gap-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <Link to={primaryCTA.href}>
            <Button variant="primary" size="medium">
              {primaryCTA.label}
            </Button>
          </Link>
          {secondaryCTA && (
            <Link to={secondaryCTA.href}>
              <Button variant="outline" size="medium">
                {secondaryCTA.icon && (
                  <span className="inline-flex">{secondaryCTA.icon}</span>
                )}
                {secondaryCTA.label}
              </Button>
            </Link>
          )}
        </motion.div>
      </div>
    </div>
  );
}

// Helper component for highlighted/underlined keywords - inherits font size from parent
export function HighlightedKeyword({
  children,
  href,
}: {
  children: ReactNode;
  href?: string;
}) {
  if (href) {
    return (
      <Link to={href}>
        <span className="text-white underline decoration-white/50 underline-offset-4 hover:decoration-white transition-all cursor-pointer">
          {children}
        </span>
      </Link>
    );
  }

  return (
    <span className="text-white underline decoration-white/50 underline-offset-4">
      {children}
    </span>
  );
}
