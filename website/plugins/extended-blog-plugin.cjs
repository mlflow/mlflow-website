const blogPluginExports = require("@docusaurus/plugin-content-blog");
const fs = require("fs");

const blogPlugin = blogPluginExports.default;

// This approach is inspired by the Docusaurus maintaner's suggestion:
// https://github.com/facebook/docusaurus/discussions/6423#discussioncomment-2008482
function filterFuturePosts(content) {
  const now = new Date();

  // Determine which posts are published (date <= now)
  const publishedPosts = content.blogPosts.filter(
    (post) => new Date(post.metadata.date) <= now,
  );
  const publishedIds = new Set(publishedPosts.map((post) => post.id));

  content.blogPosts = publishedPosts;

  // Filter paginated listing pages
  content.blogListPaginated = content.blogListPaginated
    .map((page) => ({
      ...page,
      items: page.items.filter((id) => publishedIds.has(id)),
    }))
    .filter((page) => page.items.length > 0);

  // Re-number pagination metadata
  content.blogListPaginated.forEach((page, i) => {
    page.metadata = {
      ...page.metadata,
      page: i + 1,
      totalPages: content.blogListPaginated.length,
      totalCount: publishedPosts.length,
      previousPage:
        i > 0 ? content.blogListPaginated[i - 1].metadata.permalink : undefined,
      nextPage:
        i < content.blogListPaginated.length - 1
          ? content.blogListPaginated[i + 1].metadata.permalink
          : undefined,
    };
  });

  // Filter tags
  for (const [key, tag] of Object.entries(content.blogTags)) {
    tag.items = tag.items.filter((id) => publishedIds.has(id));
    tag.pages = tag.pages
      .map((page) => ({
        ...page,
        items: page.items.filter((id) => publishedIds.has(id)),
      }))
      .filter((page) => page.items.length > 0);

    if (tag.items.length === 0) {
      delete content.blogTags[key];
    }
  }
}

async function blogPluginEnhanced(...pluginArgs) {
  const blogPluginInstance = await blogPlugin(...pluginArgs);

  return {
    ...blogPluginInstance,
    contentLoaded: async function (...contentLoadedArgs) {
      const { content } = contentLoadedArgs[0];

      if (process.env.FILTER_FUTURE_POSTS) {
        filterFuturePosts(content);
      }

      fs.writeFileSync(
        ".docusaurus/blog-posts.json",
        JSON.stringify(content.blogPosts),
      );

      return blogPluginInstance.contentLoaded(...contentLoadedArgs);
    },
  };
}

module.exports = {
  ...blogPluginExports,
  default: blogPluginEnhanced,
};
