#!/usr/bin/env node
/**
 * Tracear CLI — compile a marker image into a .tracear target file.
 *   npx tracear compile poster.png [poster.tracear]
 */
import { readFile, writeFile } from "node:fs/promises";

const [cmd, input, output] = process.argv.slice(2);

if (cmd !== "compile" || !input) {
  console.log("Usage: tracear compile <image.png|image.jpg> [output.tracear]");
  process.exit(cmd ? 1 : 0);
}

const buf = await readFile(input);

let width, height, rgba;
if (buf[0] === 0x89 && buf[1] === 0x50) {
  const { PNG } = await import("pngjs");
  const png = PNG.sync.read(buf);
  ({ width, height } = png);
  rgba = png.data;
} else if (buf[0] === 0xff && buf[1] === 0xd8) {
  const { default: jpeg } = await import("jpeg-js");
  const img = jpeg.decode(buf, { useTArray: true, maxMemoryUsageInMB: 1024 });
  ({ width, height } = img);
  rgba = img.data;
} else {
  console.error("Unsupported image format (PNG or JPEG expected).");
  process.exit(1);
}

// Match the browser compiler: cap the long side at 512 px (box downscale).
const maxSide = 512;
if (Math.max(width, height) > maxSide) {
  const scale = maxSide / Math.max(width, height);
  const nw = Math.max(1, Math.round(width * scale));
  const nh = Math.max(1, Math.round(height * scale));
  const out = new Uint8Array(nw * nh * 4);
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      const sx = Math.min(width - 1, Math.floor((x + 0.5) / scale));
      const sy = Math.min(height - 1, Math.floor((y + 0.5) / scale));
      const s = (sy * width + sx) * 4;
      const d = (y * nw + x) * 4;
      out[d] = rgba[s];
      out[d + 1] = rgba[s + 1];
      out[d + 2] = rgba[s + 2];
      out[d + 3] = 255;
    }
  }
  rgba = out;
  width = nw;
  height = nh;
}

const init = (await import("../dist/wasm/tracear_wasm.js")).default;
const { compile_marker_rgba } = await import("../dist/wasm/tracear_wasm.js");
await init({ module_or_path: await readFile(new URL("../dist/wasm/tracear_wasm_bg.wasm", import.meta.url)) });

const data = compile_marker_rgba(new Uint8Array(rgba.buffer, rgba.byteOffset, width * height * 4), width, height);
const featureCount = new DataView(data.buffer, data.byteOffset).getUint32(16, true);
const outPath = output ?? input.replace(/\.[^.]+$/, "") + ".tracear";
await writeFile(outPath, data);
console.log(`${outPath}: ${(data.length / 1024).toFixed(1)} KB, ${featureCount} detection features (${width}x${height})`);
