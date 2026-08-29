/**
 * Browser-side marker compilation. Rarely-used path, so it initializes its
 * own WASM instance on the calling thread instead of round-tripping through
 * the tracking worker.
 */
import init, { compile_marker_rgba } from "./wasm/tracear_wasm.js";

let wasmReady: Promise<unknown> | null = null;

export interface CompileResult {
  /** `.tracear` bytes — feed to TracearConfig.targets, or save as a file. */
  data: Uint8Array;
  featureCount: number;
  /** Size the image was compiled at (marker level-0 px). */
  width: number;
  height: number;
}

/**
 * Bundle single-marker `.tracear` blobs into one multi-marker pack file.
 * Pure byte concatenation — nothing is recompiled, so adding or removing a
 * target never touches the others. The pack can be passed anywhere a
 * `.tracear` target is accepted; its markers expand in order.
 *
 * Pack format v1 (little-endian): magic "TRPK" | version u32 | count u32 |
 * per-marker byte length u32 | the marker blobs back to back.
 */
export function packMarkers(markers: (Uint8Array | ArrayBuffer)[]): Uint8Array {
  const blobs = markers.map((m) => (m instanceof Uint8Array ? m : new Uint8Array(m)));
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

/**
 * Compile an image into a `.tracear` marker. The image is scaled down so its
 * long side is at most `maxSize` px (512 default — plenty for detection and
 * keeps marker files small).
 */
export async function compileImage(
  source: ImageBitmapSource,
  maxSize = 512,
): Promise<CompileResult> {
  if (!wasmReady) wasmReady = init();
  await wasmReady;
  const bmp = await createImageBitmap(source);
  const scale = Math.min(1, maxSize / Math.max(bmp.width, bmp.height));
  const w = Math.max(1, Math.round(bmp.width * scale));
  const h = Math.max(1, Math.round(bmp.height * scale));
  const canvas: OffscreenCanvas | HTMLCanvasElement =
    typeof OffscreenCanvas !== "undefined"
      ? new OffscreenCanvas(w, h)
      : Object.assign(document.createElement("canvas"), { width: w, height: h });
  const ctx = canvas.getContext("2d", { willReadFrequently: true }) as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D
    | null;
  if (!ctx) throw new Error("tracear: could not create 2d canvas context");
  ctx.drawImage(bmp, 0, 0, w, h);
  bmp.close();
  const img = ctx.getImageData(0, 0, w, h);
  const data = compile_marker_rgba(new Uint8Array(img.data.buffer), w, h);
  const featureCount = new DataView(data.buffer, data.byteOffset).getUint32(16, true);
  return { data, featureCount, width: w, height: h };
}
