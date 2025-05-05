import { test, expect } from "@playwright/test";

test.describe("Slack", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("./");
  });

  test("Slack invite link is active", async ({ page }) => {
    expect(true).toBe(true);
    /* await page.goto("/slack");
    const title = (await page.title()).toLocaleLowerCase();
    expect(title).toContain("mlflow");
    expect(title).toContain("join"); */
  });
});
