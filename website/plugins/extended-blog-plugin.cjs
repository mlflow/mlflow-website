const blogPluginExports = require("@docusaurus/plugin-content-blog");
const fs = require("fs");

const blogPlugin = blogPluginExports.default;

// This approach is inspired by the Docusaurus maintaner's suggestion:
// https://github.com/facebook/docusaurus/discussions/6423#discussioncomment-2008482
async function blogPluginEnhanced(...pluginArgs) {
  const blogPluginInstance = await blogPlugin(...pluginArgs);

  return {
    ...blogPluginInstance,
    contentLoaded: async function (...contentLoadedArgs) {
      const { blogPosts } = contentLoadedArgs[0].content;

      fs.writeFileSync(
        ".docusaurus/blog-posts.json",
        JSON.stringify(blogPosts),
      );

      return blogPluginInstance.contentLoaded(...contentLoadedArgs);
    },
  };
}

module.exports = {
  ...blogPluginExports,
  default: blogPluginEnhanced,
};
