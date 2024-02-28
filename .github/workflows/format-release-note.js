const formatMarkdown = ({ core, filenames }) => {
  const fs = require('fs');
  filenames.map(filename => {
    core.info(`file: ${filename}`);
    const content = fs.readFileSync(filename, 'utf8');
    core.info(`file content: ${content}`);
  });
}

module.exports = {
  formatMarkdown
}