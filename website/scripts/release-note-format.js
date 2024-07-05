const fs = require('fs');
const path = require('path');

const { generateLinks } = require('../../.github/workflows/format-release-note');

const getMarkdownFiles = (dir) => {
  let results = [];
  const list = fs.readdirSync(dir, { withFileTypes: true});
  list.forEach(file => {
    const filePath = path.resolve(dir, file.name);
    if (file.isDirectory()) {
      results = results.concat(getMarkdownFiles(filePath));
    } else {
      if (file.name.endsWith("release.md") && !isPreRelease(file.name)) {
        results.push(filePath);
      }
    }
  });
  return results;
}

// Don't update the old releases prior to 1.x
const isPreRelease = (filename) => {
  const versionMatch = filename.match(/(\d+)\.(\d+)\.(\d+)-release\.md$/);
  if (versionMatch) {
    const major = parseInt(versionMatch[1], 10);
    return major < 1;
  }
  return false;
}

const formatAllMarkdownFiles = () => {
  const releaseDir = path.resolve(__dirname, '../releases');
  const markdownFiles = getMarkdownFiles(releaseDir);

  const updatedFiles = [];

  markdownFiles.forEach(filename => {
    const content = fs.readFileSync(filename, 'utf8');
    const formatted = generateLinks(content);

    if (content !== formatted) {
      fs.writeFileSync(filename, formatted, 'utf8');
      updatedFiles.push(filename);
    }
  });

  if (updatedFiles.length > 0) {
    console.log('Updated files:');
    updatedFiles.forEach(file => console.log(file));
  } else {
    console.log('No files were updated.');
  }
}

formatAllMarkdownFiles();
