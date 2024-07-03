const fs = require('fs');
const path = require('path');

const { generateLinks } = require('../.github/workflows/format-release-note');

const getMarkdownFiles = (dir) => {
  let results = [];
  const list = fs.readdirSync(dir);
  list.forEach(file => {
    file = path.resolve(dir, file);
    const stat = fs.statSync(file);
    if (stat && stat.isDirectory()) {
      results = results.concat(getMarkdownFiles(file));
    } else {
      if (file.endsWith("release.md") && !isPreRelease(file)) {
        results.push(file);
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
  const releaseDir = path.resolve(__dirname, '../website/releases');
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
