/**
 * SDK build: tsc emits ESM + d.ts preserving the module structure (no
 * bundling — `new Worker(new URL(...))` must survive as-is for consumers'
 * bundlers), the wasm assets are copied alongside, and the worker URL's .ts
 * extension is rewritten to .js. Ends with a gzip size report against the
 * 300 KB budget.
 */
import { execSync } from "node:child_process";
import { cpSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const pkg = dirname(dirname(fileURLToPath(import.meta.url)));

rmSync(join(pkg, "dist"), { recursive: true, force: true });
execSync("npx tsc -p tsconfig.build.json", { cwd: pkg, stdio: "inherit" });
cpSync(join(pkg, "src", "wasm"), join(pkg, "dist", "wasm"), { recursive: true });
rmSync(join(pkg, "dist", "wasm", ".gitignore"), { force: true });

// Source says `new URL("./worker.ts", ...)` (so Vite dev serves TS directly);
// the published artifact must point at the emitted .js.
const indexPath = join(pkg, "dist", "index.js");
writeFileSync(indexPath, readFileSync(indexPath, "utf8").replaceAll("./worker.ts", "./worker.js"));

let total = 0;
const walk = (dir, rel = "") => {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) {
      walk(p, `${rel}${name}/`);
    } else {
      const gz = gzipSync(readFileSync(p)).length;
      total += gz;
      console.log(`  ${(gz / 1024).toFixed(1).padStart(7)} KB gz  ${rel}${name}`);
    }
  }
};
console.log("dist contents (gzipped):");
walk(join(pkg, "dist"));
console.log(`  total: ${(total / 1024).toFixed(1)} KB gzipped (budget: 300 KB)`);
if (total > 300 * 1024) {
  console.error("SIZE BUDGET EXCEEDED");
  process.exit(1);
}
