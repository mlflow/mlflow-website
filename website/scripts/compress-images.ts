import fs from "fs";
import path from "path";
import readline from "readline";
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

async function compressImage(filePath: string): Promise<boolean> {
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
    return true;
  } else {
    console.log(`  ${path.relative(".", filePath)}: already optimal, skipped`);
    return false;
  }
}

async function main() {
  const dirs = ["static/img", "blog"];
  const files = dirs.flatMap((dir) =>
    fs.existsSync(dir) ? getImageFiles(dir) : [],
  );

  const toCompress = files.filter((f) => fs.statSync(f).size >= MIN_SIZE_BYTES);

  console.log(
    `Found ${toCompress.length} image files over 500KB (out of ${files.length} total).`,
  );
  console.log(
    "This script modifies files in-place. Please make a backup if you want to retain the original copies.",
  );

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  const answer = await new Promise<string>((resolve) =>
    rl.question("Proceed? (y/n) ", resolve),
  );
  rl.close();

  if (answer.toLowerCase() !== "y") {
    console.log("Aborted.");
    return;
  }

  let compressed = 0;
  let skipped = 0;
  for (const file of toCompress) {
    const didCompress = await compressImage(file);
    if (didCompress) {
      compressed++;
    } else {
      skipped++;
    }
  }

  console.log(
    `\nDone. Compressed ${compressed} files, skipped ${skipped} files.`,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
