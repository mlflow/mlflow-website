import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";

const config: Config = {
  title: "MLflow",
  // tagline: 'Dinosaurs are cool',
  favicon: "img/mlflow-favicon.ico",

  // Set the production url of your site here
  url: "https://your-docusaurus-site.example.com",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: process.env.GITHUB_ACTIONS ? "/mlflow-site-refresh/" : "/",

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "facebook", // Usually your GitHub org/user name.
  projectName: "docusaurus", // Usually your repo name.

  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          // editUrl:
          //   "https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/",
        },
        blog: {
          showReadingTime: true,
          blogSidebarTitle: "All posts",
          blogSidebarCount: "ALL",
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          // editUrl:
          //   "https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/",
        },
        theme: {
          customCss: "./src/css/custom.css",
        },
        gtag: {
          // containerId: process.env.GTM_ID || "GTM-TEST",
          trackingID: "G-E8ZPTRLNBL",
          anonymizeIP: true,
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
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
        blogSidebarTitle: "All posts",
        blogSidebarCount: "ALL",
        blogTitle: "Releases",
      },
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: "img/docusaurus-social-card.jpg",
    colorMode: {
      defaultMode: "dark",
      disableSwitch: true,
      respectPrefersColorScheme: false,
    },
    // announcementBar: {
    //   id: "support_us",
    //   content: "WIP",
    //   isCloseable: true,
    // },
    navbar: {
      // title: 'My Site',
      logo: {
        alt: "MLflow",
        src: "img/mlflow-white.svg",
        srcDark: "img/mlflow-black.svg",
      },
      items: [
        // {
        //   type: 'docSidebar',
        //   sidebarId: 'tutorialSidebar',
        //   position: 'left',
        //   label: 'Tutorial',
        // },
        // { to: "/features", label: "Features", position: "right" },
        { to: "/blog", label: "Blog", position: "right" },
        { to: "/releases", label: "Releases", position: "right" },
        {
          label: "Docs",
          to: "pathname:///docs/latest/index.html",
          position: "right",
          target: "_self",
        },
        {
          href: "https://github.com/mlflow/mlflow",
          label: "Contribute",
          position: "right",
        },
        { to: "/ambassador", label: "Ambassador Program", position: "right" },
        {
          to: "pathname:///docs/latest/getting-started/index.html",
          label: "Get Started",
          position: "right",
          target: "_self",
          className: "navbar__item__get-started",
        },
      ],
    },
    footer: {
      style: "dark",
      logo: {
        alt: "MLflow",
        src: "img/mlflow-white.svg",
        srcDark: "img/mlflow-black.svg",
        href: "/",
        width: "200px",
      },
      links: [
        {
          title: "Community",
          items: [
            {
              label: "Stack Overflow",
              href: "https://stackoverflow.com/questions/tagged/mlflow",
            },
            {
              label: "LinkedIn",
              href: "https://www.linkedin.com/company/mlflow-org",
            },
            {
              label: "X",
              href: "https://x.com/mlflow",
            },
          ],
        },
        {
          title: "Resources",
          items: [
            {
              label: "Docs",
              to: "pathname:///docs/latest/index.html",
            },
            {
              label: "Releases",
              to: "/releases",
            },
            {
              label: "Blog",
              to: "/blog",
            },
          ],
        },
      ],
      copyright: `© ${new Date().getFullYear()} MLflow Project, a Series of LF Projects, LLC.`,
    },
    prism: {
      theme: prismThemes.vsDark,
      darkTheme: prismThemes.vsDark,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
