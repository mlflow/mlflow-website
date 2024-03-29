import { load } from "js-yaml";
import { readFileSync, readdirSync, writeFileSync, statSync } from "fs";
import { join } from "path";

type Frontmatter = {
  title: string;
  tags: string[];
  authors: string[];
  thumbnail?: string;
  slug: string;
};

type Author = {
  name: string;
  title: string;
  url: string;
  image_url: string;
};

type AuthorMap = {
  [key: string]: Author;
};

type Blog = {
  title: string;
  tags: string[];
  authors: Author[];
  path: string;
  date: string;
  thumbnail: string;
};

type BlogPath = {
  date: string;
  dir: string;
  path: string;
};

type Release = Blog & { version: string };

function parseFrontmatter(blog: string): Frontmatter {
  return load(/---\n(.*)\n---\n(.*)/s.exec(blog)[1]) as Frontmatter;
}

function walk(dir: string): string[] {
  let files: string[] = [];
  readdirSync(dir).forEach((f: string) => {
    const dirPath = join(dir, f);
    const isDirectory = statSync(dirPath).isDirectory();
    isDirectory ? (files = files.concat(walk(dirPath))) : files.push(dirPath);
  });
  return files;
}

function resolvePath(path: string): BlogPath {
  const regex = new RegExp(/(.+)\/(\d{4}-\d{2}-\d{2})-(.*?)(\/index)?\.mdx?$/);
  const match = regex.exec(path);
  if (match === null) {
    throw new Error(`Invalid path: ${path}`);
  }
  const [, dir, date, base] = match;
  return {
    date,
    dir,
    path: `${dir}/${date.replace(/-/g, "/")}/${base}`,
  };
}

const isBlog = (path: string) => path.endsWith(".md") || path.endsWith(".mdx");

function loadBlogs(dir: string): Blog[] {
  return walk(dir)
    .filter(isBlog)
    .map((blog) => {
      const { date, path } = resolvePath(blog);
      const { title, tags, authors, thumbnail, slug } = parseFrontmatter(
        readFileSync(blog, "utf8"),
      );
      const yaml = load(readFileSync("blog/authors.yml", "utf8")) as AuthorMap;
      return {
        title,
        path: `/${dir}/${slug}`,
        tags,
        authors: authors.map((author) => yaml[author]),
        date,
        thumbnail,
      };
    });
}

const sortByDateDesc = (a: Blog, b: Blog) => {
  return new Date(b.date).getTime() - new Date(a.date).getTime();
};

function extractVersion(s: string): string {
  return /(\d+\.\d+\.\d+)/.exec(s)[1];
}

function loadReleases(dir: string): Release[] {
  return loadBlogs(dir).map((blog) => ({
    ...blog,
    version: extractVersion(blog.title),
  }));
}

const blogs = loadBlogs("blog").sort(sortByDateDesc);
const releases = loadReleases("releases").sort(sortByDateDesc);
const blogsJson = JSON.stringify(blogs, null, 2);
const releasesJson = JSON.stringify(releases, null, 2);
const src = `
// This file is auto-generated by scripts/compile.ts. Do not edit this file directly.

export type Author = {
  name: string;
  title: string;
  url: string;
  image_url: string;
};

export type Blog = {
  title: string;
  tags?: string[];
  authors: Author[];
  path: string;
  date: string;
  thumbnail: string;
};

export type Release = {
  title: string;
  authors: Author[];
  path: string;
  date: string;
  version: string;
};

// Sort by date descending
export const BLOGS: Blog[] = ${blogsJson};

// Sort by date descending
export const RELEASES: Release[] = ${releasesJson};
`;
writeFileSync("src/posts.ts", src);
