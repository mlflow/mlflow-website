"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_1 = require("@playwright/test");
var config = {
  testDir: "./tests",
  use: {
    baseURL: "http://localhost:3000",
  },
  projects: [
    {
      name: "chromium",
      use: test_1.devices["Desktop Chrome"],
    },
  ],
  webServer: {
    // Several docusaurus plugins are disabled in development.
    // Use production build to enable them.
    command: "npm run build && npm run serve",
    port: 3000,
    timeout: 30 * 1000, // 30 seconds
  },
};
exports.default = config;
