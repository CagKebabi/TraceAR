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

// WebGL readback state: texImage2D(bitmap) + readPixels into ONE reused
// buffer. getImageData allocates ~1 MB of garbage per frame (tens of MB/s at
// camera rate), which degrades Safari over long runs; this path is
// steady-state zero-allocation. (readPixels from an FBO-attached texture
// returns rows in uploaded, i.e. top-down, order — verified.)
let gl: WebGLRenderingContext | null | undefined;
let glTex: WebGLTexture | null = null;
let glBuf: Uint8Array | null = null;
let glW = 0;
let glH = 0;

function rgbaFromBitmapGL(bitmap: ImageBitmap, w: number, h: number): Uint8Array | null {
  if (gl === undefined) {
    try {
      gl = new OffscreenCanvas(1, 1).getContext("webgl", {
        antialias: false,
        depth: false,
        stencil: false,
      }) as WebGLRenderingContext | null;
      if (gl) {
        glTex = gl.createTexture();
        gl.bindFramebuffer(gl.FRAMEBUFFER, gl.createFramebuffer());
        gl.bindTexture(gl.TEXTURE_2D, glTex);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      }
    } catch {
      gl = null;
    }
  }
  if (!gl || !glTex) return null;
  try {
    gl.bindTexture(gl.TEXTURE_2D, glTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, glTex, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      gl = null;
      return null;
    }
    if (!glBuf || glW !== w || glH !== h) {
      glBuf = new Uint8Array(w * h * 4);
      glW = w;
      glH = h;
    }
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, glBuf);
    bitmap.close();
    return glBuf;
  } catch {
    gl = null; // fall back to the 2D path (bitmap still open)
    return null;
  }
}

function rgbaFromBitmap(bitmap: ImageBitmap, w: number, h: number): Uint8Array {
  const viaGl = rgbaFromBitmapGL(bitmap, w, h);
  if (viaGl) return viaGl;
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
      // A target file may hold one marker or a multi-marker pack; widths are
      // consumed in expanded-marker order across all targets.
      let widthCursor = 0;
      for (let i = 0; i < msg.markers.length; i++) {
        const widths = new Float64Array(
          msg.widths.slice(widthCursor).map((w) => w ?? 1.0),
        );
        const indices = engine.add_markers(new Uint8Array(msg.markers[i]), widths);
        widthCursor += indices.length;
        for (const idx of indices) {
          const size = engine.marker_size(idx);
          markerSizes.push([size[0], size[1]]);
        }
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
        const rect = frame.visibleRect ?? { x: 0, y: 0, width: frame.codedWidth, height: frame.codedHeight };
        // Camera VideoFrames can arrive in SENSOR orientation (landscape)
        // with rotation metadata the <video> element applies but copyTo does
        // not — feeding that here would squash the image and kill detection.
        // Any aspect mismatch with the target: use the bitmap path instead,
        // which draws the video element (rotation applied).
        const aspectOk =
          rect.width > 0 &&
          rect.height > 0 &&
          Math.abs(rect.width / rect.height - msg.width / msg.height) / (msg.width / msg.height) < 0.01;
        if (!(fmt.startsWith("I420") || fmt === "NV12") || !aspectOk) {
          frame.close();
          post({ type: "fallback" });
          return;
        }
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
