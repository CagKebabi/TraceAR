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
  buf: ArrayBuffer;
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
      // Live frames run the stateful detect<->track pipeline; one-shot
      // requests (detectImage) must not disturb tracking state.
      const px = new Uint8Array(msg.buf);
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
