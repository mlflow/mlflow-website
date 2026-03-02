"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var fs_1 = require("fs");
var path_1 = require("path");
function getMarkdownFiles(dir, fileList) {
  if (fileList === void 0) {
    fileList = [];
  }
  var files = fs_1.default.readdirSync(dir);
  files.forEach(function (file) {
    var filePath = path_1.default.join(dir, file);
    var stat = fs_1.default.statSync(filePath);
    if (stat.isDirectory()) {
      getMarkdownFiles(filePath, fileList);
    } else if (filePath.endsWith(".md")) {
      fileList.push(filePath);
    }
  });
  return fileList;
}
function checkForThumbnailString(filePaths) {
  var matchingFiles = [];
  filePaths.forEach(function (filePath) {
    var fileContent = fs_1.default.readFileSync(filePath, "utf-8");
    // check for the absence of a leading slash in the thumbnail path
    // this will cause the image to be broken on /blog/tags/... pages
    if (fileContent.includes("thumbnail: img")) {
      matchingFiles.push(filePath);
    }
  });
  return matchingFiles;
}
var markdownFiles = getMarkdownFiles("./blog");
var filesWithoutTrailingSlash = checkForThumbnailString(markdownFiles);
if (filesWithoutTrailingSlash.length > 0) {
  console.log("Found files with broken thumbnail paths:");
  console.log(filesWithoutTrailingSlash);
  console.log(
    "Please add a leading slash to the thumbnail path, otherwise the image will be broken on /blog/tags/... pages",
  );
  process.exit(1);
}
