const defaultTheme = require("tailwindcss/defaultTheme");

/** @type {import('tailwindcss').Config} */
module.exports = {
  theme: {
    ...defaultTheme,
    colors: {
      ...defaultTheme.colors,
      gray: {
        1000: "rgb(255 255 255 / 100%)",
        900: "rgb(255 255 255 / 90%)",
        800: "rgb(255 255 255 / 80%)",
        700: "rgb(255 255 255 / 70%)",
        600: "rgb(255 255 255 / 60%)",
        500: "rgb(255 255 255 / 50%)",
        400: "rgb(255 255 255 / 40%)",
        300: "rgb(255 255 255 / 30%)",
        200: "rgb(255 255 255 / 20%)",
        100: "rgb(255 255 255 / 10%)",
        50: "rgb(255 255 255 / 5%)",
      },
    },
    extend: {
      fontFamily: {
        sans: ["DM Sans", "sans-serif"],
      },
      colors: {
        primary: {
          50: "#f0f9ff",
          100: "#e0f2fe",
          200: "#bae6fd",
          300: "#7dd3fc",
          400: "#38bdf8",
          500: "#0ea5e9",
          600: "#0284c7",
          700: "#0369a1",
          800: "#075985",
          900: "#0c4a6e",
        },
        brand: {
          black: "#0E1416",
          red: "#EB1700",
          teal: "#44EDBC",
        },
      },
    },
  },
  plugins: [],
  corePlugins: {
    preflight: false, // Disable Tailwind's reset to avoid conflicts with Docusaurus
  },
};
