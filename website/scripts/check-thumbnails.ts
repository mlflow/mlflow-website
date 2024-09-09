import fs from "fs";
import path from "path";

function getMarkdownFiles(dir: string, fileList: string[] = []): string[] {
  const files = fs.readdirSync(dir);

  files.forEach((file) => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      getMarkdownFiles(filePath, fileList);
    } else if (filePath.endsWith(".md")) {
      fileList.push(filePath);
    }
  });

  return fileList;
}

function checkForThumbnailString(filePaths: string[]): string[] {
  const matchingFiles: string[] = [];

  filePaths.forEach((filePath) => {
    const fileContent = fs.readFileSync(filePath, "utf-8");

    // check for the absence of a leading slash in the thumbnail path
    // this will cause the image to be broken on /blog/tags/... pages
    if (fileContent.includes("thumbnail: img")) {
      matchingFiles.push(filePath);
    }
  });

  return matchingFiles;
}

const markdownFiles = getMarkdownFiles("./blog");
const filesWithoutTrailingSlash = checkForThumbnailString(markdownFiles);

if (filesWithoutTrailingSlash.length > 0) {
  console.log("Found files with broken thumbnail paths:");
  console.log(filesWithoutTrailingSlash);
  console.log(
    "Please add a leading slash to the thumbnail path, otherwise the image will be broken on /blog/tags/... pages",
  );
  process.exit(1);
}
