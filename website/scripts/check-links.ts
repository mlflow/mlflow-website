import markdownLinkCheck from "markdown-link-check";
import fs from "fs";
import path from "path";

function getMarkdownFiles(dir: string, fileList: string[] = []): string[] {
  const files = fs.readdirSync(dir);

  files.forEach((file) => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      getMarkdownFiles(filePath, fileList);
    } else if (filePath.endsWith(".md") || filePath.endsWith(".mdx")) {
      fileList.push(filePath);
    }
  });

  return fileList;
}

type Result = {
  link: string;
  status: "alive" | "dead" | "ignored";
  statusCode: number;
  err: Error | null;
};

async function check(
  content: string,
  checkExternalLinks: boolean,
): Promise<Result[]> {
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
      ignorePatterns: checkExternalLinks
        ? [
            { pattern: "^(?!http)" }, // relative links
            { pattern: "^http:\\/\\/127\\.0\\.0\\.1" },
            { pattern: "^http:\\/\\/localhost" },
          ]
        : [
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

  const checkExternalLinks = process.env.CHECK_EXTERNAL_LINKS === "true";
  const contentDirs = ["./blog", "./releases", "./cookbook"];
  const markdownFiles: string[] = [];

  for (const dir of contentDirs) {
    if (fs.existsSync(dir)) {
      getMarkdownFiles(dir, markdownFiles);
    }
  }

  for (const filename of markdownFiles) {
    console.log(`[CHECKING] ${filename}`);

    const content = fs.readFileSync(filename, "utf8");
    const brokenLinks = await check(content, checkExternalLinks);

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
