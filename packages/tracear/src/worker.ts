/// <reference lib="webworker" />
/**
 * Detection worker: owns the WASM engine so the main thread never blocks on
 * CV work. Frames arrive as transferred RGBA buffers; results go back as a
 * transferred Float64Array (RESULT_STRIDE values per marker, see wasm crate).
 */
import init, { Engine } from "./wasm/tracear_wasm.js";

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
  /** …or a GPU-side bitmap (readback happens here in the worker)… */
  bitmap?: ImageBitmap;
  /** …or, fastest, a WebCodecs VideoFrame: for YUV camera formats the luma
   * plane is used directly as grayscale — no canvas, no color conversion. */
  videoFrame?: VideoFrame;
  /** Target (processed) size — homographies come back in this pixel space. */
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

/** The VideoFrame path is unusable here (format/API) — use bitmaps instead. */
export interface FallbackMessage {
  type: "fallback";
}

const post = (
  msg: ReadyMessage | ResultMessage | ErrorMessage | FallbackMessage,
  transfer: Transferable[] = [],
) => (self as unknown as Worker).postMessage(msg, transfer);

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
      if (!engine) {
        msg.videoFrame?.close();
        return;
      }
      const t0 = performance.now();
      let data: Float64Array;
      if (msg.videoFrame) {
        const frame = msg.videoFrame;
        const fmt = frame.format ?? "";
        if (!(fmt.startsWith("I420") || fmt === "NV12")) {
          frame.close();
          post({ type: "fallback" });
          return;
        }
        const rect = frame.visibleRect ?? { x: 0, y: 0, width: frame.codedWidth, height: frame.codedHeight };
        const size = frame.allocationSize({ rect });
        const buf = new Uint8Array(size);
        const layout = await frame.copyTo(buf, { rect });
        frame.close();
        data = engine.process_yplane(
          buf.subarray(layout[0].offset),
          layout[0].stride,
          rect.width,
          rect.height,
          msg.width,
          msg.height,
          msg.timestamp,
        );
      } else {
        const px = msg.bitmap ? rgbaFromBitmap(msg.bitmap, msg.width, msg.height) : new Uint8Array(msg.buf!);
        // Live frames run the stateful detect<->track pipeline; one-shot
        // requests (detectImage) must not disturb tracking state.
        data =
          msg.requestId !== undefined
            ? engine.detect_rgba(px, msg.width, msg.height)
            : engine.process_rgba(px, msg.width, msg.height, msg.timestamp);
      }
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
    const frameMsg = msg as FrameMessage;
    if (frameMsg.videoFrame) {
      // copyTo/allocationSize not usable here — degrade to the bitmap path.
      try {
        frameMsg.videoFrame.close();
      } catch {
        /* already closed */
      }
      post({ type: "fallback" });
    } else {
      post({ type: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }
};
