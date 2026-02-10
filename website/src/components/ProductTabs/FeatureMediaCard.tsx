import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Highlight } from "prism-react-renderer";
import { CopyButton } from "../CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../CodeSnippet/codeTheme";
import { Code, Image } from "lucide-react";
import type { Feature } from "./features";

const CodeBlock = ({
  code,
  language = "python",
}: {
  code: string;
  language?: "python" | "typescript";
}) => {
  const prismLanguage = language === "typescript" ? "tsx" : "python";

  return (
    <Highlight
      theme={customNightOwl}
      code={code.trim()}
      language={prismLanguage}
    >
      {({ style, tokens, getLineProps, getTokenProps }) => (
        <div className="relative h-full" style={{ backgroundColor: CODE_BG }}>
          <CopyButton
            code={code.trim()}
            className="absolute top-3 right-3 p-2 rounded-md z-10"
          />
          <pre
            className="h-full overflow-auto leading-snug font-mono p-4 m-0 dark-scrollbar"
            style={{
              ...style,
              backgroundColor: CODE_BG,
              fontSize: "13px",
              lineHeight: "1.5",
            }}
          >
            {tokens.map((line, i) => (
              <div key={i} {...getLineProps({ line })}>
                {line.map((token, key) => (
                  <span key={key} {...getTokenProps({ token })} />
                ))}
              </div>
            ))}
          </pre>
        </div>
      )}
    </Highlight>
  );
};

export const FeatureMediaCard = ({ feature }: { feature: Feature }) => {
  const [showCode, setShowCode] = useState(false);

  return (
    <div className="relative h-[350px] rounded-xl overflow-hidden border border-white/10 group">
      {/* Content area */}
      <AnimatePresence mode="wait">
        {!showCode ? (
          <motion.div
            key="screenshot"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="absolute inset-0"
          >
            {/* Dark gradient background with vivid red-to-blue theme */}
            <div
              className="absolute inset-0"
              style={{
                background:
                  "linear-gradient(135deg, #2a1020 0%, #251535 25%, #152040 50%, #102545 100%)",
              }}
            />

            {/* Subtle grid pattern */}
            <div
              className="absolute inset-0 opacity-40"
              style={{
                backgroundImage: `
                  linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
                  linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px)
                `,
                backgroundSize: "50px 50px",
              }}
            />

            {/* Blurred orbs for visual interest */}
            <div
              className="absolute -top-20 -left-20 w-72 h-72 rounded-full opacity-50"
              style={{
                background:
                  "radial-gradient(circle, rgba(224,85,133,0.5) 0%, transparent 70%)",
              }}
            />
            <div
              className="absolute -bottom-20 -right-20 w-64 h-64 rounded-full opacity-45"
              style={{
                background:
                  "radial-gradient(circle, rgba(79,172,254,0.5) 0%, transparent 70%)",
              }}
            />
            <div
              className="absolute top-1/3 -right-10 w-56 h-56 rounded-full opacity-40"
              style={{
                background:
                  "radial-gradient(circle, rgba(168,85,247,0.4) 0%, transparent 70%)",
              }}
            />

            {/* Screenshot image */}
            <div
              className="absolute bottom-0 right-0 w-[93%] h-[93%] z-10 pt-[1px] rounded-tl-lg pl-[1px]"
              style={{
                background:
                  "linear-gradient(135deg, rgba(255,255,255,0.25) 0%, rgba(255,255,255,0.05) 50%, transparent 100%)",
              }}
            >
              <div
                className="w-full h-full pt-[4px] overflow-hidden rounded-tl-lg pl-[4px]"
                style={{ backgroundColor: "#11171d" }}
              >
                <img
                  src={feature.imageSrc}
                  alt={`${feature.title} screenshot`}
                  className="object-cover rounded-tl"
                  style={{
                    width: `${feature.imageZoom ?? 115}%`,
                    height: `${feature.imageZoom ?? 115}%`,
                    objectPosition: feature.imagePosition ?? "left top",
                  }}
                  loading="lazy"
                />
              </div>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="code"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="absolute inset-0"
          >
            <CodeBlock
              code={feature.codeSnippet}
              language={feature.codeLanguage}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Toggle button - bottom right with glow effect */}
      <motion.button
        onClick={() => setShowCode(!showCode)}
        className="absolute bottom-3 right-3 z-20 px-3 py-1.5 rounded-lg bg-black/80 hover:bg-black/90 text-white/90 hover:text-white transition-all backdrop-blur-sm border border-white/20 flex items-center gap-1.5 text-xs font-medium"
        style={{
          boxShadow: "0 0 8px rgba(99, 102, 241, 0.2)",
        }}
        whileHover={{
          boxShadow: "0 0 12px rgba(99, 102, 241, 0.35)",
        }}
        aria-label={showCode ? "Show screenshot" : "Show code"}
      >
        {showCode ? (
          <>
            <Image className="w-4 h-4" />
            <span>Screenshot</span>
          </>
        ) : (
          <>
            <Code className="w-4 h-4" />
            <span>Code</span>
          </>
        )}
      </motion.button>
    </div>
  );
};
