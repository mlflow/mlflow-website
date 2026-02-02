import { test, expect } from "@playwright/test";

test.describe("Mobile menu accordion behavior", () => {
  test.beforeEach(async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/");
    await page.waitForLoadState("networkidle");
  });

  test("submenus are collapsed by default", async ({ page }) => {
    // Open mobile menu
    const menuButton = page.locator(
      'button[data-collapse-toggle="navbar-sticky"]',
    );
    await menuButton.click();

    // Both submenus should be collapsed (max-h-0)
    const componentsSubmenu = page.locator("#mobile-components-submenu");
    const docsSubmenu = page.locator("#mobile-docs-submenu");

    await expect(componentsSubmenu).toHaveClass(/max-h-0/);
    await expect(docsSubmenu).toHaveClass(/max-h-0/);
  });

  test("accordion expands one section at a time", async ({ page }) => {
    // Open mobile menu
    const menuButton = page.locator(
      'button[data-collapse-toggle="navbar-sticky"]',
    );
    await menuButton.click();

    const componentsButton = page.locator(
      'button[aria-controls="mobile-components-submenu"]',
    );
    const docsButton = page.locator(
      'button[aria-controls="mobile-docs-submenu"]',
    );
    const componentsSubmenu = page.locator("#mobile-components-submenu");
    const docsSubmenu = page.locator("#mobile-docs-submenu");

    // Click Components - should expand
    await componentsButton.click();
    await expect(componentsButton).toHaveAttribute("aria-expanded", "true");
    await expect(componentsSubmenu).toHaveClass(/max-h-\[600px\]/);
    await expect(docsSubmenu).toHaveClass(/max-h-0/);

    // Click Docs - Components should collapse, Docs should expand
    await docsButton.click();
    await expect(docsButton).toHaveAttribute("aria-expanded", "true");
    await expect(componentsButton).toHaveAttribute("aria-expanded", "false");
    await expect(docsSubmenu).toHaveClass(/max-h-\[300px\]/);
    await expect(componentsSubmenu).toHaveClass(/max-h-0/);
  });

  test("submenus reset when menu closes", async ({ page }) => {
    const menuButton = page.locator(
      'button[data-collapse-toggle="navbar-sticky"]',
    );

    // Open menu and expand a submenu
    await menuButton.click();
    const componentsButton = page.locator(
      'button[aria-controls="mobile-components-submenu"]',
    );
    await componentsButton.click();
    await expect(componentsButton).toHaveAttribute("aria-expanded", "true");

    // Close menu
    await menuButton.click();

    // Reopen menu - submenu should be collapsed
    await menuButton.click();
    await expect(componentsButton).toHaveAttribute("aria-expanded", "false");
    await expect(page.locator("#mobile-components-submenu")).toHaveClass(
      /max-h-0/,
    );
  });
});
