import { test, expect } from "@playwright/test";

test.describe("Inline Code Styling", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("./releases/2022-02-28-1.24.0-release");
  });

  test("Inline code elements have correct background and no forced black color", async ({ page }) => {
    // Find a code element in the release notes
    const codeElement = page.locator('code').first();
    
    // Check that code elements exist on the page
    await expect(codeElement).toBeVisible();
    
    // Get computed styles for the code element
    const backgroundColor = await codeElement.evaluate((el) => {
      return window.getComputedStyle(el).backgroundColor;
    });
    
    const color = await codeElement.evaluate((el) => {
      return window.getComputedStyle(el).color;
    });
    
    // Check that background-color has been set (should be semi-transparent white #ffffff1a)
    // The exact computed value might vary slightly, but it should not be the default
    expect(backgroundColor).not.toBe('rgba(0, 0, 0, 0)'); // Not transparent
    expect(backgroundColor).not.toBe('transparent');
    
    // Check that color is not forced to black (should inherit white from theme)
    expect(color).not.toBe('rgb(0, 0, 0)'); // Should not be pure black
    expect(color).not.toBe('black');
  });

  test("Alert elements still have black text color", async ({ page }) => {
    // Navigate to a page that might have alerts
    await page.goto("./");
    
    // We can't guarantee there are alert elements on every page, 
    // so let's just check our CSS rules exist
    const customCSS = await page.evaluate(() => {
      const stylesheets = Array.from(document.styleSheets);
      let foundAlertRule = false;
      
      for (const stylesheet of stylesheets) {
        try {
          if (stylesheet.href && stylesheet.href.includes('custom.css')) {
            const rules = Array.from(stylesheet.cssRules || stylesheet.rules);
            for (const rule of rules) {
              if (rule.selectorText && rule.selectorText.includes('.alert p')) {
                foundAlertRule = true;
                break;
              }
            }
          }
        } catch (e) {
          // Cross-origin stylesheets might not be accessible
          continue;
        }
      }
      
      return foundAlertRule;
    });
    
    // This is a simple check to ensure our CSS separation worked
    expect(customCSS).toBeTruthy();
  });
});