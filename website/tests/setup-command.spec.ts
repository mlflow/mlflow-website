import { test, expect } from "@playwright/test";

const SETUP_COMMAND = "curl -LsSf https://mlflow.org/wizard/setup.sh | sh";

for (const width of [320, 390, 1440]) {
  test(`setup command is visible and copyable at ${width}px`, async ({
    page,
    context,
  }) => {
    await page.setViewportSize({ width, height: 900 });
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);
    await page.goto("/");

    const command = page.getByText(SETUP_COMMAND, { exact: true });
    await expect(command).toBeVisible();
    const snippet = command.locator("..");
    const box = await snippet.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.x).toBeGreaterThanOrEqual(0);
    expect(box!.x + box!.width).toBeLessThanOrEqual(width);

    await snippet.getByRole("button", { name: "Copy code" }).click();
    await expect(
      snippet.getByRole("button", { name: "Copied!" }),
    ).toBeVisible();
    expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(
      SETUP_COMMAND,
    );
  });
}
