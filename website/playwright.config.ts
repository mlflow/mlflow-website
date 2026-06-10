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
    command: "npm run build && npm run serve",
    port: 3000,
    timeout: 60 * 1000, // 1 minute
  },
};

export default config;
