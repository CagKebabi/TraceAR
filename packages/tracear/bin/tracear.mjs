#!/usr/bin/env node
/**
 * Tracear CLI — compile marker images into .tracear target files.
 *   npx tracear compile poster.png [poster.tracear]
 *   npx tracear compile a.png b.jpg c.png -o album.tracear   (multi-marker pack)
 *   npx tracear pack a.tracear b.tracear -o album.tracear    (bundle, no recompile)
 */
import { readFile, writeFile } from "node:fs/promises";

const argv = process.argv.slice(2);
const cmd = argv.shift();

function usage(code) {
  console.log(
    [
      "Usage:",
      "  tracear compile <image.png|image.jpg> [output.tracear]",
      "  tracear compile <image...> -o <output.tracear>   # multi-marker pack",
      "  tracear pack <file.tracear...> -o <output.tracear>",
    ].join("\n"),
  );
  process.exit(code);
}

// Split "-o out" from positional args.
const inputs = [];
let output = null;
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === "-o" || argv[i] === "--output") {
    output = argv[++i];
  } else {
    inputs.push(argv[i]);
  }
}

if ((cmd !== "compile" && cmd !== "pack") || inputs.length === 0) {
  usage(cmd ? 1 : 0);
}

// Backward compat: `compile input output.tracear` (two positionals).
if (cmd === "compile" && output === null && inputs.length === 2 && inputs[1].endsWith(".tracear")) {
  output = inputs.pop();
}

/** Pack format v1: magic "TRPK" | version u32 | count u32 | lengths | blobs. */
function packBytes(blobs) {
  const total = blobs.reduce((n, b) => n + b.length, 0);
  const out = new Uint8Array(12 + blobs.length * 4 + total);
  const view = new DataView(out.buffer);
  out.set([0x54, 0x52, 0x50, 0x4b], 0); // "TRPK"
  view.setUint32(4, 1, true);
  view.setUint32(8, blobs.length, true);
  let pos = 12;
  for (const b of blobs) {
    view.setUint32(pos, b.length, true);
    pos += 4;
  }
  for (const b of blobs) {
    out.set(b, pos);
    pos += b.length;
  }
  return out;
}

async function compileOne(input, compile_marker_rgba) {
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
    console.error(`${input}: unsupported image format (PNG or JPEG expected).`);
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

  const data = compile_marker_rgba(
    new Uint8Array(rgba.buffer, rgba.byteOffset, width * height * 4),
    width,
    height,
  );
  const featureCount = new DataView(data.buffer, data.byteOffset).getUint32(16, true);
  console.log(
    `  ${input}: ${(data.length / 1024).toFixed(1)} KB, ${featureCount} detection features (${width}x${height})`,
  );
  return data;
}

if (cmd === "pack") {
  if (!output) {
    console.error("pack: -o <output.tracear> is required.");
    process.exit(1);
  }
  const blobs = [];
  for (const input of inputs) {
    const buf = new Uint8Array(await readFile(input));
    const magic = String.fromCharCode(...buf.slice(0, 4));
    if (magic !== "TRCR") {
      console.error(`${input}: not a single-marker .tracear file.`);
      process.exit(1);
    }
    blobs.push(buf);
  }
  const data = packBytes(blobs);
  await writeFile(output, data);
  console.log(`${output}: ${(data.length / 1024).toFixed(1)} KB, ${blobs.length} markers packed`);
} else {
  const init = (await import("../dist/wasm/tracear_wasm.js")).default;
  const { compile_marker_rgba } = await import("../dist/wasm/tracear_wasm.js");
  await init({ module_or_path: await readFile(new URL("../dist/wasm/tracear_wasm_bg.wasm", import.meta.url)) });

  if (inputs.length === 1) {
    const data = await compileOne(inputs[0], compile_marker_rgba);
    const outPath = output ?? inputs[0].replace(/\.[^.]+$/, "") + ".tracear";
    await writeFile(outPath, data);
    console.log(`${outPath} written`);
  } else {
    const outPath = output ?? "targets.tracear";
    const blobs = [];
    for (const input of inputs) {
      blobs.push(await compileOne(input, compile_marker_rgba));
    }
    const data = packBytes(blobs);
    await writeFile(outPath, data);
    console.log(`${outPath}: ${(data.length / 1024).toFixed(1)} KB, ${blobs.length} markers packed`);
  }
}
