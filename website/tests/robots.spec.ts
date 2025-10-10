import { test, expect } from "@playwright/test";

test.describe("Robots.txt", () => {
  test("robots.txt is accessible", async ({ page }) => {
    const response = await page.goto("/robots.txt");
    expect(response?.status()).toBe(200);
  });

  test("robots.txt allows latest docs", async ({ page }) => {
    const response = await page.goto("/robots.txt");
    const content = await response?.text();
    expect(content).toContain("Allow: /docs/latest/");
  });

  test("robots.txt disallows legacy versions", async ({ page }) => {
    const response = await page.goto("/robots.txt");
    const content = await response?.text();
    expect(content).toContain("Disallow: /docs/1.*/");
    expect(content).toContain("Disallow: /docs/2.*/");
    expect(content).toContain("Disallow: /docs/0.*/");
  });

  test("robots.txt includes AI crawler configurations", async ({ page }) => {
    const response = await page.goto("/robots.txt");
    const content = await response?.text();
    // Check for various AI crawlers
    expect(content).toContain("GPTBot");
    expect(content).toContain("ClaudeBot");
    expect(content).toContain("Google-Extended");
    expect(content).toContain("CCBot");
  });

  test("robots.txt includes sitemap", async ({ page }) => {
    const response = await page.goto("/robots.txt");
    const content = await response?.text();
    expect(content).toContain("Sitemap:");
  });
});
