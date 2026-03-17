import fs from "fs";
import path from "path";
import sharp from "sharp";

const MAX_WIDTH = 1400;
const PNG_QUALITY = 80;
const JPEG_QUALITY = 80;
const MIN_SIZE_BYTES = 500 * 1024; // only compress files > 500KB

function getImageFiles(dir: string, files: string[] = []): string[] {
  for (const entry of fs.readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (fs.statSync(full).isDirectory()) {
      getImageFiles(full, files);
    } else if (/\.(png|jpe?g)$/i.test(entry)) {
      files.push(full);
    }
  }
  return files;
}

function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  return `${(bytes / 1024).toFixed(0)}KB`;
}

async function compressImage(filePath: string): Promise<void> {
  const stat = fs.statSync(filePath);
  if (stat.size < MIN_SIZE_BYTES) return;

  const ext = path.extname(filePath).toLowerCase();
  const input = fs.readFileSync(filePath);
  let pipeline = sharp(input).resize({
    width: MAX_WIDTH,
    withoutEnlargement: true,
  });

  if (ext === ".png") {
    pipeline = pipeline.png({ quality: PNG_QUALITY });
  } else {
    pipeline = pipeline.jpeg({ quality: JPEG_QUALITY });
  }

  const output = await pipeline.toBuffer();

  // Only write if we actually reduced the size
  if (output.length < stat.size) {
    fs.writeFileSync(filePath, output);
    const pct = ((1 - output.length / stat.size) * 100).toFixed(0);
    console.log(
      `  ${path.relative(".", filePath)}: ${formatSize(stat.size)} → ${formatSize(output.length)} (${pct}% reduction)`,
    );
  } else {
    console.log(`  ${path.relative(".", filePath)}: already optimal, skipped`);
  }
}

async function main() {
  const dirs = ["static/img", "blog"];
  const files = dirs.flatMap((dir) =>
    fs.existsSync(dir) ? getImageFiles(dir) : [],
  );

  console.log(
    `Found ${files.length} image files, compressing those > 500KB...`,
  );

  let compressed = 0;
  for (const file of files) {
    const stat = fs.statSync(file);
    if (stat.size >= MIN_SIZE_BYTES) {
      compressed++;
      await compressImage(file);
    }
  }

  console.log(`\nDone. Processed ${compressed} files.`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
