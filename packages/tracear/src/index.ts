/**
 * Tracear SDK — public API.
 *
 * M1 state: continuous *detection* (worker + WASM) with homography output.
 * Sub-pixel tracking (M2) and pose/filtering (M3) slot in behind this same
 * API; `poseAt` stays null until M3.
 */
import { Emitter } from "./events";
import type { ReadyMessage, ResultMessage, ErrorMessage } from "./worker";

export type Homography = Float64Array;

/** Values per marker in a worker result (mirrors the wasm crate). */
const RESULT_STRIDE = 12;

export interface TracearConfig {
  /** Element the managed <video> is appended to; position it yourself (e.g. relative). */
  container: HTMLElement;
  /** Compiled `.tracear` targets: URLs or raw bytes. */
  targets: (string | ArrayBuffer | Uint8Array)[];
  /** Long-side cap for the processed frame, default 640. */
  maxProcessSize?: number;
  /** Consecutive detection misses before `targetLost`, default 8. */
  lostAfterMisses?: number;
  /** Extra getUserMedia video constraints (merged over the defaults). */
  videoConstraints?: MediaTrackConstraints;
}

export interface UpdateEvent {
  index: number;
  /** Maps marker px -> process-frame px (row-major 3x3). */
  homography: Homography;
  markerWidth: number;
  markerHeight: number;
  inliers: number;
  matches: number;
  /** Rough 0..1 confidence from inlier count (placeholder until M2 tracking). */
  quality: number;
  /** Frame capture time (performance.now() domain). */
  timestamp: number;
  processWidth: number;
  processHeight: number;
  /** Detection time inside the worker (ms). */
  workerMs: number;
}

export type TracearEvents = {
  targetFound: { index: number };
  targetLost: { index: number };
  update: UpdateEvent;
  error: { message: string };
};

interface TargetState {
  found: boolean;
  misses: number;
  width: number;
  height: number;
}

export class Tracear {
  /** The managed <video> element (available after create()). */
  readonly video: HTMLVideoElement;

  private worker: Worker;
  private emitter = new Emitter<TracearEvents>();
  private config: Required<Pick<TracearConfig, "maxProcessSize" | "lostAfterMisses">> & TracearConfig;
  private targets: TargetState[] = [];
  private canvas: HTMLCanvasElement | OffscreenCanvas | null = null;
  private ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null = null;
  private stream: MediaStream | null = null;
  private running = false;
  private inflight = false;
  private frameCb: number | null = null;
  private usesRvfc = false;
  private processW = 0;
  private processH = 0;
  private nextRequestId = 1;
  private pending = new Map<number, (r: ResultMessage) => void>();

  private constructor(config: TracearConfig, worker: Worker, markerSizes: [number, number][]) {
    this.config = { maxProcessSize: 640, lostAfterMisses: 8, ...config };
    this.worker = worker;
    this.targets = markerSizes.map(([width, height]) => ({ found: false, misses: 0, width, height }));
    this.video = document.createElement("video");
    this.video.playsInline = true;
    this.video.muted = true;
    this.video.autoplay = true;
    this.video.style.display = "block";
    this.video.style.width = "100%";
    this.worker.onmessage = (ev: MessageEvent<ResultMessage | ErrorMessage>) => {
      if (ev.data.type === "result") {
        const { requestId } = ev.data;
        if (requestId !== undefined) {
          this.pending.get(requestId)?.(ev.data);
          this.pending.delete(requestId);
        } else {
          this.onResult(ev.data);
        }
      } else if (ev.data.type === "error") {
        this.emitter.emit("error", { message: ev.data.message });
      }
    };
  }

  static async create(config: TracearConfig): Promise<Tracear> {
    if (!config.targets.length) throw new Error("tracear: at least one target is required");
    const markers: ArrayBuffer[] = [];
    for (const t of config.targets) {
      if (typeof t === "string") {
        const res = await fetch(t);
        if (!res.ok) throw new Error(`tracear: failed to fetch target ${t} (${res.status})`);
        markers.push(await res.arrayBuffer());
      } else if (t instanceof Uint8Array) {
        // Copy: the buffer is transferred to the worker and must not detach caller data.
        markers.push(t.slice().buffer);
      } else {
        markers.push(t.slice(0));
      }
    }
    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    const ready = new Promise<ReadyMessage>((resolve, reject) => {
      worker.onmessage = (ev: MessageEvent<ReadyMessage | ErrorMessage>) => {
        if (ev.data.type === "ready") resolve(ev.data);
        else if (ev.data.type === "error") reject(new Error(`tracear: worker init failed: ${ev.data.message}`));
      };
      worker.onerror = (e) => reject(new Error(`tracear: worker failed to load: ${e.message}`));
    });
    worker.postMessage({ type: "init", markers }, markers);
    const readyMsg = await ready;
    return new Tracear(config, worker, readyMsg.markerSizes);
  }

  on<K extends keyof TracearEvents>(event: K, cb: (e: TracearEvents[K]) => void): () => void {
    return this.emitter.on(event, cb);
  }

  async start(): Promise<void> {
    if (this.running) return;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        ...this.config.videoConstraints,
      },
    });
    this.video.srcObject = this.stream;
    this.config.container.appendChild(this.video);
    await new Promise<void>((resolve) => {
      if (this.video.readyState >= 2) resolve();
      else this.video.onloadeddata = () => resolve();
    });
    await this.video.play();

    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    const scale = Math.min(1, this.config.maxProcessSize / Math.max(vw, vh));
    this.processW = Math.max(2, Math.round(vw * scale));
    this.processH = Math.max(2, Math.round(vh * scale));
    this.canvas =
      typeof OffscreenCanvas !== "undefined"
        ? new OffscreenCanvas(this.processW, this.processH)
        : Object.assign(document.createElement("canvas"), { width: this.processW, height: this.processH });
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true }) as typeof this.ctx;
    if (!this.ctx) throw new Error("tracear: could not create 2d context");

    this.running = true;
    this.inflight = false;
    this.usesRvfc = "requestVideoFrameCallback" in HTMLVideoElement.prototype;
    this.scheduleFrame();
  }

  stop(): void {
    this.running = false;
    if (this.frameCb !== null) {
      if (this.usesRvfc) this.video.cancelVideoFrameCallback(this.frameCb);
      else cancelAnimationFrame(this.frameCb);
      this.frameCb = null;
    }
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
  }

  dispose(): void {
    this.stop();
    this.worker.terminate();
    this.video.remove();
    this.emitter.clear();
  }

  /** Filtered pose at a render timestamp — lands in M3; null until then. */
  poseAt(_index: number, _timestamp: number): Float32Array | null {
    return null;
  }

  /**
   * One-shot detection on a still image (no camera needed) — useful for
   * validating a marker, thumbnails, or automated tests. Returns one entry
   * per configured target, null where the target wasn't found. Does not
   * emit found/lost/update events.
   */
  async detectImage(source: ImageBitmapSource): Promise<(UpdateEvent | null)[]> {
    const bmp = await createImageBitmap(source);
    const scale = Math.min(1, this.config.maxProcessSize / Math.max(bmp.width, bmp.height));
    const w = Math.max(2, Math.round(bmp.width * scale));
    const h = Math.max(2, Math.round(bmp.height * scale));
    const canvas =
      typeof OffscreenCanvas !== "undefined"
        ? new OffscreenCanvas(w, h)
        : Object.assign(document.createElement("canvas"), { width: w, height: h });
    const ctx = canvas.getContext("2d", { willReadFrequently: true }) as
      | CanvasRenderingContext2D
      | OffscreenCanvasRenderingContext2D
      | null;
    if (!ctx) throw new Error("tracear: could not create 2d context");
    ctx.drawImage(bmp, 0, 0, w, h);
    bmp.close();
    const img = ctx.getImageData(0, 0, w, h);
    const requestId = this.nextRequestId++;
    const result = await new Promise<ResultMessage>((resolve) => {
      this.pending.set(requestId, resolve);
      this.worker.postMessage(
        { type: "frame", buf: img.data.buffer, width: w, height: h, timestamp: performance.now(), requestId },
        [img.data.buffer],
      );
    });
    const out: (UpdateEvent | null)[] = [];
    const d = result.data;
    for (let i = 0; i < this.targets.length; i++) {
      const base = i * RESULT_STRIDE;
      if (d[base] !== 1.0) {
        out.push(null);
        continue;
      }
      const inliers = d[base + 10];
      out.push({
        index: i,
        homography: d.slice(base + 1, base + 10),
        markerWidth: this.targets[i].width,
        markerHeight: this.targets[i].height,
        inliers,
        matches: d[base + 11],
        quality: Math.min(1, inliers / 40),
        timestamp: result.timestamp,
        processWidth: result.width,
        processHeight: result.height,
        workerMs: result.ms,
      });
    }
    return out;
  }

  private scheduleFrame(): void {
    if (!this.running) return;
    if (this.usesRvfc) {
      this.frameCb = this.video.requestVideoFrameCallback(() => this.processFrame());
    } else {
      this.frameCb = requestAnimationFrame(() => this.processFrame());
    }
  }

  private processFrame(): void {
    if (!this.running) return;
    // Drop frames while the worker is busy — never queue.
    if (!this.inflight && this.ctx && this.video.readyState >= 2) {
      this.ctx.drawImage(this.video, 0, 0, this.processW, this.processH);
      const img = this.ctx.getImageData(0, 0, this.processW, this.processH);
      this.inflight = true;
      this.worker.postMessage(
        {
          type: "frame",
          buf: img.data.buffer,
          width: this.processW,
          height: this.processH,
          timestamp: performance.now(),
        },
        [img.data.buffer],
      );
    }
    this.scheduleFrame();
  }

  private onResult(msg: ResultMessage): void {
    this.inflight = false;
    const d = msg.data;
    for (let i = 0; i < this.targets.length; i++) {
      const base = i * RESULT_STRIDE;
      const state = this.targets[i];
      const found = d[base] === 1.0;
      if (found) {
        state.misses = 0;
        if (!state.found) {
          state.found = true;
          this.emitter.emit("targetFound", { index: i });
        }
        const inliers = d[base + 10];
        this.emitter.emit("update", {
          index: i,
          homography: d.slice(base + 1, base + 10),
          markerWidth: state.width,
          markerHeight: state.height,
          inliers,
          matches: d[base + 11],
          quality: Math.min(1, inliers / 40),
          timestamp: msg.timestamp,
          processWidth: msg.width,
          processHeight: msg.height,
          workerMs: msg.ms,
        });
      } else if (state.found) {
        state.misses++;
        if (state.misses >= this.config.lostAfterMisses) {
          state.found = false;
          state.misses = 0;
          this.emitter.emit("targetLost", { index: i });
        }
      }
    }
  }
}
