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

export const CODE_BG = "#0d1117";
