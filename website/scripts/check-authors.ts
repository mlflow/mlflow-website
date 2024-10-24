import fs from "fs";
import path from "path";
import yaml from "js-yaml";

function fileExistsWithCaseSync(filepath: string): boolean {
  const dir = path.dirname(filepath);
  if (dir === "/" || dir === ".") {
    return true;
  }
  const filenames = fs.readdirSync(dir);
  if (filenames.indexOf(path.basename(filepath)) === -1) {
    return false;
  }
  return fileExistsWithCaseSync(dir);
}

type Author = {
  name: string;
  title: string;
  url: string;
  image_url?: string;
};

function readAuthors(): Record<string, Author> {
  const authorsFile = fs.readFileSync(
    path.join(process.cwd(), "blog", "authors.yml"),
    "utf-8",
  );
  return yaml.load(authorsFile) as Record<string, Author>;
}

function main(): void {
  const authors = readAuthors();
  const authorsWithInvalidImageUrl: string[] = [];
  Object.entries(authors).forEach(([author, authorData]) => {
    if (
      authorData.image_url &&
      !/^https?:\/\//.test(authorData.image_url) &&
      !fileExistsWithCaseSync(
        path.join(process.cwd(), "static", authorData.image_url),
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
