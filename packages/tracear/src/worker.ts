/// <reference lib="webworker" />
/**
 * Detection worker: owns the WASM engine so the main thread never blocks on
 * CV work. Frames arrive as transferred RGBA buffers; results go back as a
 * transferred Float64Array (RESULT_STRIDE values per marker, see wasm crate).
 */
import init, { Engine } from "../wasm/tracear_wasm.js";

export interface InitMessage {
  type: "init";
  markers: ArrayBuffer[];
  /** Physical width per marker (meters or any unit; poses use the same unit). */
  widths: number[];
}

export interface FrameMessage {
  type: "frame";
  /** RGBA pixels (canvas fallback path)… */
  buf?: ArrayBuffer;
  /** …or a GPU-side bitmap: the slow pixel readback then happens HERE in the
   * worker instead of blocking the main thread's frame pump. */
  bitmap?: ImageBitmap;
  width: number;
  height: number;
  timestamp: number;
  /** Set for one-shot detectImage() calls; echoed back in the result. */
  requestId?: number;
}

export interface ReadyMessage {
  type: "ready";
  /** [width, height] per added marker. */
  markerSizes: [number, number][];
}

export interface ResultMessage {
  type: "result";
  /** RESULT_STRIDE (12) f64 per marker: [found, h x 9, inliers, matches]. */
  data: Float64Array;
  ms: number;
  timestamp: number;
  width: number;
  height: number;
  requestId?: number;
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

const post = (msg: ReadyMessage | ResultMessage | ErrorMessage, transfer: Transferable[] = []) =>
  (self as unknown as Worker).postMessage(msg, transfer);

let engine: Engine | null = null;
let canvas: OffscreenCanvas | null = null;
let ctx: OffscreenCanvasRenderingContext2D | null = null;

function rgbaFromBitmap(bitmap: ImageBitmap, w: number, h: number): Uint8Array {
  if (!canvas || canvas.width !== w || canvas.height !== h) {
    canvas = new OffscreenCanvas(w, h);
    ctx = canvas.getContext("2d", { willReadFrequently: true });
  }
  if (!ctx) throw new Error("tracear worker: no 2d context");
  ctx.drawImage(bitmap, 0, 0, w, h);
  bitmap.close();
  return new Uint8Array(ctx.getImageData(0, 0, w, h).data.buffer);
}

self.onmessage = async (ev: MessageEvent<InitMessage | FrameMessage>) => {
  const msg = ev.data;
  try {
    if (msg.type === "init") {
      await init();
      engine = new Engine();
      const markerSizes: [number, number][] = [];
      for (let i = 0; i < msg.markers.length; i++) {
        const idx = engine.add_marker(new Uint8Array(msg.markers[i]), msg.widths[i] ?? 1.0);
        const size = engine.marker_size(idx);
        markerSizes.push([size[0], size[1]]);
      }
      post({ type: "ready", markerSizes });
    } else if (msg.type === "frame") {
      if (!engine) return;
      const t0 = performance.now();
      const px = msg.bitmap ? rgbaFromBitmap(msg.bitmap, msg.width, msg.height) : new Uint8Array(msg.buf!);
      // Live frames run the stateful detect<->track pipeline; one-shot
      // requests (detectImage) must not disturb tracking state.
      const data =
        msg.requestId !== undefined
          ? engine.detect_rgba(px, msg.width, msg.height)
          : engine.process_rgba(px, msg.width, msg.height, msg.timestamp);
      const ms = performance.now() - t0;
      post(
        {
          type: "result",
          data,
          ms,
          timestamp: msg.timestamp,
          width: msg.width,
          height: msg.height,
          requestId: msg.requestId,
        },
        [data.buffer],
      );
    }
  } catch (e) {
    post({ type: "error", message: e instanceof Error ? e.message : String(e) });
  }
};
