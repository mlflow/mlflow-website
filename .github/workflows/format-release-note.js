const formatMarkdown = ({ filenames }) => {
  const fs = require('fs');

  filenames
    .filter(filename => filename.endsWith("release.md"))
    .forEach(filename => {
      const content = fs.readFileSync(filename, 'utf8');
      const formatted = generateLinks(content);
      fs.writeFileSync(filename, formatted, 'utf8');
    });
}

/**
 * Generate links for PR numbers and GitHub
 * usernames in the release note markdown content.
 */
const generateLinks = (content) => {
  // [^\[] is for the case where the PR 
  // number is already in a markdown link.
  // this prevents errors when running the
  // workflow on every commit.
  const prNumberRegex = /(^|[^\[])#(\d+)/g;
  const usernameRegex = /(^|[^\[])@([\w\d-]+)/g;

  // Replace PR numbers with markdown-styled links
  content = content.replace(
    prNumberRegex,
    "$1[#$2](https://www.github.com/mlflow/mlflow/pull/$2)"
  );

  content = content.replace(
    usernameRegex,
    "$1[@$2](https://www.github.com/$2)"
  );

  return content
}

module.exports = {
  formatMarkdown
}
