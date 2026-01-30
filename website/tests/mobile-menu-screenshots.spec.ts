import { test } from "@playwright/test";

test("capture mobile menu screenshots", async ({ page }) => {
  // Set mobile viewport (iPhone 12 size)
  await page.setViewportSize({ width: 390, height: 844 });

  await page.goto("/");

  // Wait for page to load
  await page.waitForLoadState("networkidle");

  // Screenshot 1: Initial state (menu closed)
  await page.screenshot({
    path: "../mobile-menu-closed.png",
    fullPage: false,
  });

  // Open mobile menu (click hamburger button)
  const menuButton = page.locator(
    'button[data-collapse-toggle="navbar-sticky"]',
  );
  await menuButton.click();
  await page.waitForTimeout(500);

  // Screenshot 2: Menu open - submenus collapsed by default
  await page.screenshot({
    path: "../mobile-menu-open.png",
    fullPage: false,
  });

  // Click on "Components" to expand it
  const componentsButton = page.locator("button", { hasText: "Components" });
  await componentsButton.click();
  await page.waitForTimeout(400);

  // Screenshot 3: Components expanded
  await page.screenshot({
    path: "../mobile-menu-components-expanded.png",
    fullPage: false,
  });

  // Click on "Docs" to expand it (should collapse Components due to accordion)
  const docsButton = page.locator("button", { hasText: "Docs" });
  await docsButton.click();
  await page.waitForTimeout(400);

  // Screenshot 4: Docs expanded (Components should be collapsed)
  await page.screenshot({
    path: "../mobile-menu-docs-expanded.png",
    fullPage: false,
  });
});
