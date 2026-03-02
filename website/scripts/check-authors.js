"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var fs_1 = require("fs");
var path_1 = require("path");
var js_yaml_1 = require("js-yaml");
function fileExistsWithCaseSync(filepath) {
  var dir = path_1.default.dirname(filepath);
  if (dir === "/" || dir === ".") {
    return true;
  }
  var filenames = fs_1.default.readdirSync(dir);
  if (filenames.indexOf(path_1.default.basename(filepath)) === -1) {
    return false;
  }
  return fileExistsWithCaseSync(dir);
}
function readAuthors() {
  var authorsFile = fs_1.default.readFileSync(
    path_1.default.join(process.cwd(), "blog", "authors.yml"),
    "utf-8",
  );
  return js_yaml_1.default.load(authorsFile);
}
function main() {
  var authors = readAuthors();
  var authorsWithInvalidImageUrl = [];
  Object.entries(authors).forEach(function (_a) {
    var author = _a[0],
      authorData = _a[1];
    if (
      authorData.image_url &&
      !/^https?:\/\//.test(authorData.image_url) &&
      !fileExistsWithCaseSync(
        path_1.default.join(process.cwd(), "static", authorData.image_url),
      )
    ) {
      authorsWithInvalidImageUrl.push(author);
    }
  });
  if (authorsWithInvalidImageUrl.length > 0) {
    console.log("Found authors with invalid image URLs:");
    console.log(authorsWithInvalidImageUrl);
    console.log("Please make sure the image exists in the static folder.");
    process.exit(1);
  }
}
main();
