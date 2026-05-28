const fs = require("fs");
const path = require("path");

const API_BASE = "https://api.babylovegrowth.ai/api/integrations";
const ARTICLE_DIR = path.join(__dirname, "../../website/article");

function loadApiKey() {
  if (process.env.BABYLOVEGROWTH_API_KEY) {
    return process.env.BABYLOVEGROWTH_API_KEY;
  }
  const envPath = path.join(__dirname, "../../.env");
  if (fs.existsSync(envPath)) {
    const content = fs.readFileSync(envPath, "utf8");
    for (const line of content.split("\n")) {
      const match = line.match(/^\s*BABYLOVEGROWTH_API_KEY\s*=\s*(.+)\s*$/);
      if (match) return match[1].replace(/^["']|["']$/g, "");
    }
  }
  throw new Error(
    "BABYLOVEGROWTH_API_KEY not found in environment or .env file",
  );
}

async function apiFetch(endpoint, apiKey) {
  const maxRetries = 5;
  for (let attempt = 0; ; attempt++) {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      headers: {
        "X-API-Key": apiKey,
        "Content-Type": "application/json",
      },
    });
    if (res.status === 429) {
      if (attempt >= maxRetries) {
        throw new Error(`Rate limited on ${endpoint} after ${maxRetries} retries`);
      }
      console.warn(`Rate limited on ${endpoint}, retrying in 3s (attempt ${attempt + 1}/${maxRetries})...`);
      await new Promise((resolve) => setTimeout(resolve, 3000));
      continue;
    }
    if (!res.ok) {
      throw new Error(`API request failed: ${res.status} ${res.statusText}`);
    }
    return res.json();
  }
}

function getExistingArticleIds() {
  if (!fs.existsSync(ARTICLE_DIR)) return new Set();
  const entries = fs.readdirSync(ARTICLE_DIR, { withFileTypes: true });
  const ids = new Set();
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const match = entry.name.match(/^(\d+)-/);
    if (match) ids.add(Number(match[1]));
  }
  return ids;
}

function buildFrontmatter(article) {
  const lines = ["---"];
  lines.push(`title: ${JSON.stringify(article.title)}`);
  if (article.meta_description) {
    lines.push(`description: ${JSON.stringify(article.meta_description)}`);
  }
  if (article.slug) {
    lines.push(`slug: ${article.slug}`);
  }
  if (article.keywords && article.keywords.length > 0) {
    lines.push(`tags: [${article.keywords.join(", ")}]`);
  }
  if (article.created_at) {
    lines.push(`date: ${article.created_at.split("T")[0]}`);
  }
  if (article.hero_image_url) {
    lines.push(`image: ${article.hero_image_url}`);
  }
  lines.push("---");
  return lines.join("\n");
}

// Replicate github-slugger's algorithm, which Docusaurus uses for heading IDs.
function githubSlug(text) {
  return text
    .toLowerCase()
    .trim()
    .replace(/<[^>]*>/g, "")
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-");
}

function sanitizeMarkdown(md) {
  // Fix internal anchor links to match Docusaurus heading IDs (github-slugger).
  // The API returns URL-encoded anchors (e.g. %3A for colon, %2C for comma)
  // that don't match the slugs Docusaurus generates from headings.
  md = md.replace(/\]\(#([^)]+)\)/g, (match, anchor) => {
    const decoded = decodeURIComponent(anchor);
    return `](#${githubSlug(decoded)})`;
  });

  // Remove the leading H1 that duplicates the frontmatter title
  md = md.replace(/^# .+\n+/, "");

  return md;
}

function datePrefixFromArticle(article) {
  if (article.created_at) {
    return article.created_at.split("T")[0];
  }
  return new Date().toISOString().split("T")[0];
}

async function main() {
  const apiKey = loadApiKey();

  console.log("Fetching article list from BabyLoveGrowth...");
  const articles = await apiFetch("/v1/articles?limit=10", apiKey);
  console.log(`Found ${articles.length} articles in latest batch`);

  const existingIds = getExistingArticleIds();
  const newArticles = articles.filter((a) => !existingIds.has(a.id));

  if (newArticles.length === 0) {
    console.log("No new articles to sync");
    return;
  }

  console.log(`${newArticles.length} new article(s) to fetch`);
  fs.mkdirSync(ARTICLE_DIR, { recursive: true });

  for (const summary of newArticles) {
    console.log(`Fetching full content for: ${summary.title} (id=${summary.id})`);
    const article = await apiFetch(`/v1/articles/${summary.id}`, apiKey);

    const datePrefix = datePrefixFromArticle(article);
    const slug = article.slug || `article-${article.id}`;
    const dirName = `${article.id}-${datePrefix}-${slug}`;
    const dirPath = path.join(ARTICLE_DIR, dirName);

    fs.mkdirSync(dirPath, { recursive: true });

    const frontmatter = buildFrontmatter(article);
    const content = sanitizeMarkdown(article.content_markdown || "");
    const fileContent = `${frontmatter}\n\n${content}\n`;

    fs.writeFileSync(path.join(dirPath, "index.md"), fileContent, "utf8");
    console.log(`Wrote ${dirName}/index.md`);
  }

  console.log("Done syncing articles");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
