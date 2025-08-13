import fs from "fs";
import path from "path";

function getBlogPostFiles(dir: string): string[] {
  const files = fs.readdirSync(dir);
  const blogPostFiles: string[] = [];

  files.forEach((file) => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      // Check for index.md or index.mdx in subdirectories
      const indexMd = path.join(filePath, "index.md");
      const indexMdx = path.join(filePath, "index.mdx");

      if (fs.existsSync(indexMd)) {
        blogPostFiles.push(indexMd);
      } else if (fs.existsSync(indexMdx)) {
        blogPostFiles.push(indexMdx);
      }
    } else if (file.endsWith(".md") || file.endsWith(".mdx")) {
      // Direct markdown files in blog directory
      blogPostFiles.push(filePath);
    }
  });

  return blogPostFiles;
}

function extractFrontmatter(content: string): Record<string, any> {
  const frontmatterRegex = /^---\s*\n([\s\S]*?)\n---/;
  const match = content.match(frontmatterRegex);

  if (!match) {
    return {};
  }

  const frontmatterContent = match[1];
  const frontmatter: Record<string, any> = {};

  // Simple YAML parsing for key: value pairs
  frontmatterContent.split("\n").forEach((line) => {
    const trimmedLine = line.trim();
    if (trimmedLine && !trimmedLine.startsWith("#")) {
      const colonIndex = trimmedLine.indexOf(":");
      if (colonIndex > 0) {
        const key = trimmedLine.substring(0, colonIndex).trim();
        const value = trimmedLine.substring(colonIndex + 1).trim();
        frontmatter[key] = value;
      }
    }
  });

  return frontmatter;
}

function checkForImageFrontmatter(filePaths: string[]): string[] {
  const filesWithoutImage: string[] = [];

  filePaths.forEach((filePath) => {
    const fileContent = fs.readFileSync(filePath, "utf-8");
    const frontmatter = extractFrontmatter(fileContent);

    // Check if the image field is missing or empty
    if (!frontmatter.image || frontmatter.image.trim() === "") {
      filesWithoutImage.push(filePath);
    }
  });

  return filesWithoutImage;
}

const blogPostFiles = getBlogPostFiles("./blog");
const filesWithoutImage = checkForImageFrontmatter(blogPostFiles);

if (filesWithoutImage.length > 0) {
  console.log("Found blog posts missing 'image' frontmatter field:");
  filesWithoutImage.forEach((file) => console.log(`  - ${file}`));
  console.log(
    "\nAll blog posts must include an 'image' field in their YAML frontmatter. This sets the og:image metadata tag, which is used for preview images in various social media platforms.",
  );
  console.log("Example:");
  console.log("---");
  console.log("title: Your Blog Post Title");
  console.log("image: /img/blog/your-post-image.png");
  console.log("---");
  process.exit(1);
}
