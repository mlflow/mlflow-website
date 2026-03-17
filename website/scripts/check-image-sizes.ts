import fs from "fs";
import path from "path";

const MAX_SIZE_BYTES = 1024 * 1024; // 1MB

function getImageFiles(dir: string, files: string[] = []): string[] {
  for (const entry of fs.readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (fs.statSync(full).isDirectory()) {
      getImageFiles(full, files);
    } else if (/\.(png|jpe?g)$/i.test(entry)) {
      // GIFs and videos are excluded — they need separate handling
      files.push(full);
    }
  }
  return files;
}

function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  return `${(bytes / 1024).toFixed(0)}KB`;
}

const dirs = ["static/img", "blog"];
const files = dirs.flatMap((dir) =>
  fs.existsSync(dir) ? getImageFiles(dir) : [],
);

const oversized = files
  .map((f) => ({ path: f, size: fs.statSync(f).size }))
  .filter((f) => f.size > MAX_SIZE_BYTES)
  .sort((a, b) => b.size - a.size);

if (oversized.length > 0) {
  console.error(`Found ${oversized.length} images exceeding 1MB:\n`);
  for (const f of oversized) {
    console.error(`  ${formatSize(f.size).padStart(7)}  ${f.path}`);
  }
  console.error(
    `\nPlease compress these images before committing. You can run:\n` +
      `  npm run compress-images\n`,
  );
  process.exit(1);
}

console.log("All images are under 1MB.");
