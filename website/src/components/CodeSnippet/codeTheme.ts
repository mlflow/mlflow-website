import { themes } from "prism-react-renderer";

export const customNightOwl = {
  ...themes.nightOwl,
  styles: themes.nightOwl.styles.map((style) => {
    if (style.types.includes("string")) {
      return { ...style, style: { ...style.style, color: "#58a6ff" } };
    }
    return style;
  }),
};

export const customNightOwlRed = {
  ...themes.nightOwl,
  styles: themes.nightOwl.styles.map((style) => {
    if (style.types.includes("string")) {
      return { ...style, style: { ...style.style, color: "#ff7b72" } };
    }
    if (style.types.includes("keyword")) {
      return { ...style, style: { ...style.style, color: "#ff6b8a" } };
    }
    if (style.types.includes("function")) {
      return { ...style, style: { ...style.style, color: "#ffa0a0" } };
    }
    if (
      style.types.includes("operator") ||
      style.types.includes("punctuation")
    ) {
      return { ...style, style: { ...style.style, color: "#f97583" } };
    }
    return style;
  }),
};

export const CODE_BG = "#0d1117";
