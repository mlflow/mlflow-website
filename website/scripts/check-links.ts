import markdownLinkCheck from "markdown-link-check";
import fs from "fs";
import path from "path";

function getFiles(
  dir: string,
  extensions: string[],
  fileList: string[] = [],
): string[] {
  const files = fs.readdirSync(dir);

  files.forEach((file) => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      getFiles(filePath, extensions, fileList);
    } else if (extensions.some((ext) => filePath.endsWith(ext))) {
      fileList.push(filePath);
    }
  });

  return fileList;
}

/**
 * Extract URLs from TSX/TS files and convert them into markdown links
 * so that markdown-link-check can validate them.
 */
function extractLinksAsMarkdown(tsxContent: string): string {
  const urlPattern = /href=["'](https?:\/\/[^"']+)["']/g;
  const links: string[] = [];
  let match;
  while ((match = urlPattern.exec(tsxContent)) !== null) {
    links.push(`[link](${match[1]})`);
  }
  return links.join("\n");
}

type Result = {
  link: string;
  status: "alive" | "dead" | "ignored";
  statusCode: number;
  err: Error | null;
};

async function check(content: string): Promise<Result[]> {
  return new Promise((resolve, reject) => {
    const config = {
      httpHeaders: [
        {
          urls: ["https://openai.com", "https://platform.openai.com"],
          headers: {
            "User-Agent":
              "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15",
          },
        },
      ],
      ignorePatterns: [
        {
          pattern:
            "^(?!https?:\\/\\/(www\\.)?mlflow\\.org|https:\\/\\/(www\\.)?github\\.com\\/mlflow\\/mlflow)",
        },
      ],
    };

    markdownLinkCheck(
      content,
      config,
      function (err: Error | null, results: Result[]) {
        if (err) {
          reject(err);
          return;
        }

        const brokenLinks: Result[] = [];
        results.forEach(function (result: Result) {
          console.info(
            `  ${result.status} ${result.link} (${result.statusCode})`,
          );
          if (result.status === "dead") {
            brokenLinks.push(result);
          }
        });

        resolve(brokenLinks);
      },
    );
  });
}

async function main() {
  let encounteredBrokenLinks = false;

  // Markdown/MDX content directories
  const contentDirs = ["./blog", "./releases", "./cookbook"];
  const markdownFiles: string[] = [];
  for (const dir of contentDirs) {
    if (fs.existsSync(dir)) {
      getFiles(dir, [".md", ".mdx"], markdownFiles);
    }
  }

  // TSX/TS pages
  const pagesDir = "./src/pages";
  const pageFiles: string[] = [];
  if (fs.existsSync(pagesDir)) {
    getFiles(pagesDir, [".tsx", ".ts"], pageFiles);
  }

  const allFiles = [
    ...markdownFiles.map((f) => ({ path: f, type: "markdown" as const })),
    ...pageFiles.map((f) => ({ path: f, type: "tsx" as const })),
  ];

  for (const file of allFiles) {
    console.log(`[CHECKING] ${file.path}`);

    const rawContent = fs.readFileSync(file.path, "utf8");
    const content =
      file.type === "tsx" ? extractLinksAsMarkdown(rawContent) : rawContent;

    if (file.type === "tsx" && content.length === 0) {
      continue; // no external links to check
    }

    const brokenLinks = await check(content);

    if (brokenLinks.length > 0) {
      console.log("[BROKEN LINKS]");
      brokenLinks.forEach((result) =>
        console.log(`  ${result.link} (${result.statusCode})`),
      );
      encounteredBrokenLinks = true;
    }
  }

  if (encounteredBrokenLinks) {
    console.error("\nFound broken links!");
    process.exit(1);
  } else {
    console.log("\nAll links are valid.");
  }
}

main();
