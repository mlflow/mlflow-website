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
        throw new Error(
          `Rate limited on ${endpoint} after ${maxRetries} retries`,
        );
      }
      const retryAfter = Math.max(
        0,
        Number(res.headers.get("Retry-After")) || 3,
      );
      console.warn(
        `Rate limited on ${endpoint}, retrying in ${retryAfter}s (attempt ${attempt + 1}/${maxRetries})...`,
      );
      await new Promise((resolve) => setTimeout(resolve, retryAfter * 1000));
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

// Collect slugs already published on disk. The API occasionally emits two
// distinct articles with the same slug, which produces duplicate Docusaurus
// routes and non-deterministic routing (one article silently shadows another).
function getExistingSlugs() {
  if (!fs.existsSync(ARTICLE_DIR)) return new Set();
  const entries = fs.readdirSync(ARTICLE_DIR, { withFileTypes: true });
  const slugs = new Set();
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const indexPath = path.join(ARTICLE_DIR, entry.name, "index.md");
    if (!fs.existsSync(indexPath)) continue;
    const match = fs
      .readFileSync(indexPath, "utf8")
      .match(/^slug:\s*(.+)\s*$/m);
    if (match) slugs.add(match[1].trim());
  }
  return slugs;
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

function getHeadingSlugs(md) {
  const slugs = new Set();
  const counts = new Map();
  let inCodeBlock = false;

  for (const line of md.split("\n")) {
    if (/^\s*```/.test(line)) {
      inCodeBlock = !inCodeBlock;
      continue;
    }
    if (inCodeBlock) continue;

    const match = line.match(/^(#{2,6})\s+(.+?)\s*#*\s*$/);
    if (!match) continue;

    const baseSlug = githubSlug(match[2]);
    const count = counts.get(baseSlug) || 0;
    counts.set(baseSlug, count + 1);
    slugs.add(count === 0 ? baseSlug : `${baseSlug}-${count}`);
  }

  return slugs;
}

function removeBrokenSamePageAnchorLinks(md) {
  const headingSlugs = getHeadingSlugs(md);
  const normalizeAnchor = (anchor) => githubSlug(decodeURIComponent(anchor));

  md = md.replace(
    /^(\s*[-*]\s*)\[([^\]]+)\]\(#([^)]+)\)\s*$/gm,
    (match, bullet, text, anchor) => {
      const slug = normalizeAnchor(anchor);
      return headingSlugs.has(slug) ? `${bullet}[${text}](#${slug})` : "";
    },
  );

  return md.replace(/\[([^\]]+)\]\(#([^)]+)\)/g, (match, text, anchor) => {
    const slug = normalizeAnchor(anchor);
    return headingSlugs.has(slug) ? `[${text}](#${slug})` : text;
  });
}

// MDX treats `<` as the start of a JSX tag, so bare `<` in prose (e.g. "<0.40",
// "< 5ms") breaks the build with "Unexpected character before name". Escape any
// `<` that can't begin a valid tag/comment, leaving fenced code blocks and
// inline code spans untouched (code legitimately contains `<`, e.g. `a < b`).
function escapeStrayAngleBrackets(md) {
  const segments = md.split(/(```[\s\S]*?```|`[^`\n]*`)/g);
  return segments
    .map((seg, i) => {
      // Odd indices are the captured code segments — leave them as-is.
      if (i % 2 === 1) return seg;
      return seg.replace(/<(?![A-Za-z/!$_])/g, "&lt;");
    })
    .join("");
}

function sanitizeMarkdown(md) {
  // Strip <scratchpad> blocks (internal authoring notes from the API).
  // Handles both explicit </scratchpad> closing and unclosed blocks that end
  // at a <markdown section> tag or end of content.
  md = md.replace(
    /<scratchpad>[\s\S]*?(<\/scratchpad>|(?=<markdown[\s>]))/gi,
    "",
  );

  // Remove <markdown section> / </markdown section> wrapper tags.
  md = md.replace(/<\/?markdown[^>]*>/gi, "");

  // Fix internal anchor links to match Docusaurus heading IDs (github-slugger).
  // The API returns URL-encoded anchors (e.g. %3A for colon, %2C for comma)
  // that don't match the slugs Docusaurus generates from headings.
  md = md.replace(/\]\(#([^)]+)\)/g, (match, anchor) => {
    const decoded = decodeURIComponent(anchor);
    return `](#${githubSlug(decoded)})`;
  });

  // Remove the leading H1 that duplicates the frontmatter title
  md = md.replace(/^# .+\n+/, "");

  // Drop same-page links that point to headings that are not present in the
  // generated article. Stale entries in API-provided tables of contents break
  // Docusaurus article and tag pages because excerpts preserve those links.
  md = removeBrokenSamePageAnchorLinks(md);

  // Escape stray `<` last, so earlier tag-based cleanups still see real tags.
  md = escapeStrayAngleBrackets(md);

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
  const usedSlugs = getExistingSlugs();
  const newArticles = articles.filter((a) => !existingIds.has(a.id));

  if (newArticles.length === 0) {
    console.log("No new articles to sync");
    return;
  }

  console.log(`${newArticles.length} new article(s) to fetch`);
  fs.mkdirSync(ARTICLE_DIR, { recursive: true });

  for (const summary of newArticles) {
    console.log(
      `Fetching full content for: ${summary.title} (id=${summary.id})`,
    );
    const article = await apiFetch(`/v1/articles/${summary.id}`, apiKey);

    const datePrefix = datePrefixFromArticle(article);
    let slug = article.slug || `article-${article.id}`;
    // Disambiguate slugs that collide with an already-published article so each
    // article keeps a unique route instead of shadowing an existing one.
    if (usedSlugs.has(slug)) {
      slug = `${slug}-${article.id}`;
      console.warn(`Slug collision for "${article.slug}", using "${slug}"`);
    }
    usedSlugs.add(slug);
    article.slug = slug;
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
