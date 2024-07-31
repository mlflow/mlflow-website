import type { PlaywrightTestConfig } from "@playwright/test";
import { devices } from "@playwright/test";

const config: PlaywrightTestConfig = {
  testDir: "./tests",
  use: {
    baseURL: "http://localhost:3000",
  },
  projects: [
    {
      name: "chromium",
      use: devices["Desktop Chrome"],
    },
  ],
  webServer: {
    // Several docusaurus plugins are disabled in development.
    // Use production build to enable them.
    command: "yarn build && yarn serve",
    port: 3000,
    timeout: 30 * 1000, // 30 seconds
  },
};

export default config;
