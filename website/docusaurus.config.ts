import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";

import tailwindPlugin from "./plugins/tailwind-config.cjs";
import extendedBlogPlugin from "./plugins/extended-blog-plugin.cjs";

// ensure baseUrl always ends in `/`
const baseUrl = (process.env.BASE_URL ?? "/").replace(/\/?$/, "/");

const config: Config = {
  title: "MLflow",
  // tagline: 'Dinosaurs are cool',
  favicon: "img/mlflow-favicon.ico",

  // Set the production url of your site here
  url: "http://mlflow.org",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl,

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "mlflow", // Usually your GitHub org/user name.
  projectName: "mlflow-website", // Usually your repo name.

  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "throw",
  onBrokenAnchors: "throw",

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  headTags: [
    {
      tagName: "link",
      attributes: {
        rel: "preconnect",
        href: "https://fonts.googleapis.com",
      },
    },
    {
      tagName: "link",
      attributes: {
        rel: "preconnect",
        href: "https://fonts.gstatic.com",
        crossorigin: "anonymous",
      },
    },
    {
      tagName: "link",
      attributes: {
        rel: "stylesheet",
        href: "https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&display=swap",
        display: "swap",
      },
    },
  ],

  presets: [
    [
      "classic",
      {
        docs: false,
        blog: false,
        theme: {
          customCss: "./src/css/custom.css",
        },
        googleTagManager: {
          containerId: process.env.GTM_ID || "GTM-TEST",
        },
        gtag: {
          trackingID: "AW-16857946923",
          anonymizeIP: true,
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    tailwindPlugin,
    [
      extendedBlogPlugin,
      {
        showReadingTime: true,
        blogSidebarTitle: "All posts",
        blogSidebarCount: "ALL",
        // Please change this to your repo.
        // Remove this to remove the "edit this page" links.
        // editUrl:
        //   "https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/",
        onUntruncatedBlogPosts: "ignore",
      },
    ],
    [
      "@docusaurus/plugin-content-blog",
      {
        /**
         * Required for any multi-instance plugin
         */
        id: "releases",
        /**
         * URL route for the blog section of your site.
         * *DO NOT* include a trailing slash.
         */
        routeBasePath: "releases",
        /**
         * Path to data on filesystem relative to site dir.
         */
        path: "./releases",
        authorsMapPath: "../blog/authors.yml",
        blogListComponent: "@site/src/theme/ReleaseItemsList",
        blogPostComponent: "@site/src/theme/ReleasePost",
        blogSidebarTitle: "All posts",
        blogSidebarCount: "ALL",
        blogTitle: "Releases",
        onUntruncatedBlogPosts: "ignore",
      },
    ],
    [
      // This plugin is always inactive in development and only active in production because it works on the build output.
      // Run `yarn build` and then `yarn serve` for testing.
      "@docusaurus/plugin-client-redirects",
      {
        redirects: [
          {
            // See https://slack.com/help/articles/201330256-Invite-new-members-to-your-workspace for how to create a new invite link
            from: "/slack",
            to: "https://join.slack.com/t/mlflow-users/shared_invite/zt-3585cbav7-2pDXIcSPyycbVd7s5E1E9w",
          },
          {
            from: "/blog/2024/01/25/databricks-ce",
            to: "/blog/databricks-ce",
          },
          {
            from: "/blog/2024/01/26/mlflow-year-in-review",
            to: "/blog/mlflow-year-in-review",
          },
          {
            from: "/blog/2023/11/30/mlflow-autolog",
            to: "/blog/mlflow-autolog",
          },
          {
            from: "/blog/2023/10/31/mlflow-docs-overhaul",
            to: "/blog/mlflow-docs-overhaul",
          },
          {
            from: "/blog/Deep Learning",
            to: "/blog/deep-learning-part-1",
          },
          {
            from: "/blog/mlflow",
            to: "/blog/langgraph-model-from-code",
          },
        ],
      },
    ],
  ],

  themeConfig: {
    prism: {
      theme: prismThemes.vsDark,
      darkTheme: prismThemes.vsDark,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
