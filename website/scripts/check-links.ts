import https from "https";
import http from "http";
import fs from "fs";
import path from "path";

// Match mlflow.org URLs, stopping at characters that are markdown/JSX
// delimiters rather than part of the URL. GitHub links are excluded because
// their rate limiting (429) makes CI flaky.
const TARGET_PATTERN = /https?:\/\/(www\.)?mlflow\.org[^\s)"'>\]`]*/g;

function getFiles(
  dir: string,
  extensions: string[],
  fileList: string[] = [],
): string[] {
  for (const file of fs.readdirSync(dir)) {
    const filePath = path.join(dir, file);
    if (fs.statSync(filePath).isDirectory()) {
      getFiles(filePath, extensions, fileList);
    } else if (extensions.some((ext) => filePath.endsWith(ext))) {
      fileList.push(filePath);
    }
  }
  return fileList;
}

function extractUrls(content: string): string[] {
  const urls = new Set<string>();
  for (const match of content.matchAll(TARGET_PATTERN)) {
    // Strip trailing sentence punctuation that isn't part of the URL
    const url = match[0].replace(/[.,;:!]+$/, "");
    urls.add(url);
  }
  return [...urls];
}

const REQUEST_HEADERS = {
  "User-Agent":
    "Mozilla/5.0 (compatible; MLflowLinkChecker/1.0; +https://mlflow.org)",
};

function checkUrl(url: string): Promise<{ url: string; status: number }> {
  return new Promise((resolve) => {
    const client = url.startsWith("https") ? https : http;
    const req = client.get(
      url,
      { timeout: 10000, headers: REQUEST_HEADERS },
      (res) => {
        res.resume();
        resolve({ url, status: res.statusCode ?? 0 });
      },
    );
    req.on("error", () => resolve({ url, status: 0 }));
    req.on("timeout", () => {
      req.destroy();
      resolve({ url, status: 0 });
    });
  });
}

function isOldRelease(filePath: string): boolean {
  const match = path.basename(filePath).match(/^(\d{4})-/);
  return match !== null && parseInt(match[1], 10) < 2025;
}

async function main() {
  const contentDirs = ["./blog", "./releases", "./cookbook"];
  const files: string[] = [];
  for (const dir of contentDirs) {
    if (fs.existsSync(dir)) {
      getFiles(dir, [".md", ".mdx"], files);
    }
  }
  if (fs.existsSync("./src/pages")) {
    getFiles("./src/pages", [".tsx", ".ts"], files);
  }

  // Skip release notes before 2025 -- too old to be worth checking
  const filtered = files.filter(
    (f) => !f.startsWith("./releases/") || !isOldRelease(f),
  );

  let hasFailures = false;

  for (const file of filtered) {
    const content = fs.readFileSync(file, "utf8");
    const urls = extractUrls(content);
    if (urls.length === 0) continue;

    console.log(`[CHECKING] ${file}`);
    const results = await Promise.all(urls.map(checkUrl));

    for (const { url, status } of results) {
      if (status >= 400 || status === 0) {
        console.log(`  BROKEN ${url} (${status})`);
        hasFailures = true;
      }
    }
  }

  if (hasFailures) {
    console.error("\nFound broken links!");
    process.exit(1);
  } else {
    console.log("\nAll links are valid.");
  }
}

main();
