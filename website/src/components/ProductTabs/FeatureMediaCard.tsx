import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Highlight } from "prism-react-renderer";
import { CopyButton } from "../CodeSnippet/CopyButton";
import {
  customNightOwl,
  customNightOwlRed,
  CODE_BG,
} from "../CodeSnippet/codeTheme";
import { Code, Image } from "lucide-react";
import type { Feature, FeatureImage } from "./features";

const CardBackground = ({
  colorTheme = "default",
}: {
  colorTheme?: "default" | "red";
}) => (
  <>
    <div
      className="absolute inset-0"
      style={{
        background:
          colorTheme === "red"
            ? "linear-gradient(135deg, #5a1518 0%, #6a1a1e 25%, #4a1015 50%, #350a10 100%)"
            : "linear-gradient(135deg, #2a1020 0%, #251535 25%, #152040 50%, #102545 100%)",
      }}
    />
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
    <div
      className="absolute -top-20 -left-20 w-72 h-72 rounded-full opacity-50"
      style={{
        background:
          colorTheme === "red"
            ? "radial-gradient(circle, rgba(230,50,50,0.7) 0%, transparent 70%)"
            : "radial-gradient(circle, rgba(224,85,133,0.5) 0%, transparent 70%)",
      }}
    />
    <div
      className="absolute -bottom-20 -right-20 w-64 h-64 rounded-full opacity-45"
      style={{
        background:
          colorTheme === "red"
            ? "radial-gradient(circle, rgba(200,40,60,0.6) 0%, transparent 70%)"
            : "radial-gradient(circle, rgba(79,172,254,0.5) 0%, transparent 70%)",
      }}
    />
  </>
);

const SideBySideImages = ({
  images,
  title,
  colorTheme = "default",
}: {
  images: FeatureImage[];
  title: string;
  colorTheme?: "default" | "red";
}) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  return (
    <div className="absolute inset-0 flex">
      <CardBackground colorTheme={colorTheme} />
      <div className="absolute inset-4 flex gap-2 z-10">
        {images.map((image, index) => (
          <motion.div
            key={index}
            className="relative overflow-hidden rounded-lg border border-white/10 cursor-pointer"
            style={{ backgroundColor: "#11171d" }}
            animate={{
              flex:
                hoveredIndex === index ? 5 : hoveredIndex === null ? 1 : 0.3,
            }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            onMouseEnter={() => setHoveredIndex(index)}
            onMouseLeave={() => setHoveredIndex(null)}
          >
            <img
              src={image.src}
              alt={`${title} screenshot ${index + 1}`}
              className={`${image.fit === "contain" ? "object-contain" : "object-cover"} transition-transform duration-300 ease-out`}
              style={{
                width: `${image.zoom ?? 100}%`,
                height: `${image.zoom ?? 100}%`,
                objectPosition: image.position ?? "left top",
                transform: hoveredIndex === index ? "scale(1.2)" : "scale(1)",
                transformOrigin: "left top",
                // index === 0
                //   ? "right top"
                //   : index === images.length - 1
                //     ? "left top"
                //     : "center top",
              }}
              loading="lazy"
            />
          </motion.div>
        ))}
      </div>
    </div>
  );
};

const CodeBlock = ({
  code,
  language = "python",
  colorTheme = "default",
}: {
  code: string;
  language?: "python" | "typescript";
  colorTheme?: "default" | "red";
}) => {
  const prismLanguage = language === "typescript" ? "tsx" : "python";
  const theme = colorTheme === "red" ? customNightOwlRed : customNightOwl;

  return (
    <Highlight theme={theme} code={code.trim()} language={prismLanguage}>
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

export const FeatureMediaCard = ({
  feature,
  colorTheme = "default",
}: {
  feature: Feature;
  colorTheme?: "default" | "red";
}) => {
  const hasImages = feature.images && feature.images.length > 0;
  const [showCode, setShowCode] = useState(!feature.imageSrc && !hasImages);

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
            {hasImages ? (
              <SideBySideImages
                images={feature.images!}
                title={feature.title}
                colorTheme={colorTheme}
              />
            ) : feature.fullBleedImage ? (
              /* Full-bleed: image fills entire card, no gradient */
              <div
                className="absolute inset-0 overflow-hidden"
                style={{ backgroundColor: "#11171d" }}
              >
                <img
                  src={feature.imageSrc}
                  alt={`${feature.title} screenshot`}
                  className="object-cover w-full h-full"
                  style={{
                    objectPosition: feature.imagePosition ?? "left top",
                  }}
                  loading="lazy"
                />
              </div>
            ) : (
              <>
                <CardBackground colorTheme={colorTheme} />
                <div
                  className="absolute top-1/3 -right-10 w-56 h-56 rounded-full opacity-40"
                  style={{
                    background:
                      colorTheme === "red"
                        ? "radial-gradient(circle, rgba(220,45,45,0.6) 0%, transparent 70%)"
                        : "radial-gradient(circle, rgba(168,85,247,0.4) 0%, transparent 70%)",
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
                    className="w-full h-full overflow-hidden rounded-tl-lg"
                    style={{ backgroundColor: "#ffffff" }}
                  >
                    <img
                      src={feature.imageSrc}
                      alt={`${feature.title} screenshot`}
                      className={`${feature.imageFit === "contain" ? "object-contain" : "object-cover"} rounded-tl`}
                      style={{
                        width: `${feature.imageZoom ?? 115}%`,
                        height: `${feature.imageZoom ?? 115}%`,
                        objectPosition: feature.imagePosition ?? "left top",
                      }}
                      loading="lazy"
                    />
                  </div>
                </div>
              </>
            )}
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
              colorTheme={colorTheme}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Toggle button - bottom right with glow effect */}
      {(feature.imageSrc || hasImages) && (
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
      )}
    </div>
  );
};
