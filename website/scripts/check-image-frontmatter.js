"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var fs_1 = require("fs");
var path_1 = require("path");
function getBlogPostFiles(dir) {
  var files = fs_1.default.readdirSync(dir);
  var blogPostFiles = [];
  files.forEach(function (file) {
    var filePath = path_1.default.join(dir, file);
    var stat = fs_1.default.statSync(filePath);
    if (stat.isDirectory()) {
      // Check for index.md or index.mdx in subdirectories
      var indexMd = path_1.default.join(filePath, "index.md");
      var indexMdx = path_1.default.join(filePath, "index.mdx");
      if (fs_1.default.existsSync(indexMd)) {
        blogPostFiles.push(indexMd);
      } else if (fs_1.default.existsSync(indexMdx)) {
        blogPostFiles.push(indexMdx);
      }
    } else if (file.endsWith(".md") || file.endsWith(".mdx")) {
      // Direct markdown files in blog directory
      blogPostFiles.push(filePath);
    }
  });
  return blogPostFiles;
}
function extractFrontmatter(content) {
  var frontmatterRegex = /^---\s*\n([\s\S]*?)\n---/;
  var match = content.match(frontmatterRegex);
  if (!match) {
    return {};
  }
  var frontmatterContent = match[1];
  var frontmatter = {};
  // Simple YAML parsing for key: value pairs
  frontmatterContent.split("\n").forEach(function (line) {
    var trimmedLine = line.trim();
    if (trimmedLine && !trimmedLine.startsWith("#")) {
      var colonIndex = trimmedLine.indexOf(":");
      if (colonIndex > 0) {
        var key = trimmedLine.substring(0, colonIndex).trim();
        var value = trimmedLine.substring(colonIndex + 1).trim();
        frontmatter[key] = value;
      }
    }
  });
  return frontmatter;
}
function checkForImageFrontmatter(filePaths) {
  var filesWithoutImage = [];
  filePaths.forEach(function (filePath) {
    var fileContent = fs_1.default.readFileSync(filePath, "utf-8");
    var frontmatter = extractFrontmatter(fileContent);
    // Check if the image field is missing or empty
    if (!frontmatter.image || frontmatter.image.trim() === "") {
      filesWithoutImage.push(filePath);
    }
  });
  return filesWithoutImage;
}
var blogPostFiles = getBlogPostFiles("./blog");
var filesWithoutImage = checkForImageFrontmatter(blogPostFiles);
if (filesWithoutImage.length > 0) {
  console.log("Found blog posts missing 'image' frontmatter field:");
  filesWithoutImage.forEach(function (file) {
    return console.log("  - ".concat(file));
  });
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
