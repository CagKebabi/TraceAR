/**
 * Tracear SDK — public API.
 *
 * M1 state: continuous *detection* (worker + WASM) with homography output.
 * Sub-pixel tracking (M2) and pose/filtering (M3) slot in behind this same
 * API; `poseAt` stays null until M3.
 */
import { Emitter } from "./events";
import { mat4FromPose, quatFromScaledAxis, quatMultiply, type Quat, type Vec3 } from "./math";
import type { ReadyMessage, ResultMessage, ErrorMessage } from "./worker";

export type Homography = Float64Array;

/** Values per marker in a worker result (mirrors the wasm crate):
 * [status (0/1/2), h00..h22, nGood, nTotal, quality,
 *  poseValid, qx,qy,qz,qw, tx,ty,tz, vx,vy,vz, wx,wy,wz, focalRatio] */
const RESULT_STRIDE = 28;

export interface TracearConfig {
  /** Element the managed <video> is appended to; position it yourself (e.g. relative). */
  container: HTMLElement;
  /** Compiled `.tracear` targets: URLs or raw bytes. */
  targets: (string | ArrayBuffer | Uint8Array)[];
  /** Physical width of each target in meters (or any unit — poses come out
   * in the same unit). Defaults to 1 per target. */
  targetWidthsMeters?: number[];
  /** Long-side cap for the processed frame, default 640. */
  maxProcessSize?: number;
  /** Consecutive detection misses before `targetLost`, default 8. */
  lostAfterMisses?: number;
  /** Extra getUserMedia video constraints (merged over the defaults). */
  videoConstraints?: MediaTrackConstraints;
}

/**
 * Filtered 6DoF pose. Conventions: object frame has its origin at the marker
 * center, X right, Y up, Z out of the marker face; camera frame is OpenCV
 * (X right, Y down, Z forward). Renderer adapters (tracear/three) convert.
 */
export interface PoseData {
  /** Object -> camera translation (physical units). */
  position: Vec3;
  /** Object -> camera rotation quaternion [x, y, z, w]. */
  quaternion: Quat;
  /** Filtered linear velocity, units/s. */
  velocity: Vec3;
  /** Body-frame angular velocity, rad/s: q(t+dt) ~= q * exp(w*dt). */
  angularVelocity: Vec3;
}

export interface CameraIntrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  /** Process-frame size these intrinsics are expressed in. */
  width: number;
  height: number;
}

export interface UpdateEvent {
  index: number;
  /** Maps marker px -> process-frame px (row-major 3x3). */
  homography: Homography;
  markerWidth: number;
  markerHeight: number;
  /** True when this pose came from the sub-pixel tracker; false when it came
   * from full detection (first acquire / re-acquire / detectImage). */
  tracking: boolean;
  /** Detection frames: RANSAC inliers. Tracking frames: surviving patches. */
  inliers: number;
  /** Detection frames: total matches. Tracking frames: attempted patches. */
  matches: number;
  /** 0..1 confidence (patch survival x NCC while tracking). */
  quality: number;
  /** Filtered 6DoF pose (undefined if pose estimation failed this frame). */
  pose?: PoseData;
  /** Frame capture time (performance.now() domain). */
  timestamp: number;
  processWidth: number;
  processHeight: number;
  /** Processing time inside the worker (ms). */
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
  private lastPoses: ({ pose: PoseData; timestamp: number } | null)[] = [];
  private focalRatio = 0;
  private lastProcessW = 0;
  private lastProcessH = 0;

  private constructor(config: TracearConfig, worker: Worker, markerSizes: [number, number][]) {
    this.config = { maxProcessSize: 640, lostAfterMisses: 8, ...config };
    this.worker = worker;
    this.targets = markerSizes.map(([width, height]) => ({ found: false, misses: 0, width, height }));
    this.lastPoses = markerSizes.map(() => null);
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
    const widths = config.targets.map((_, i) => config.targetWidthsMeters?.[i] ?? 1.0);
    worker.postMessage({ type: "init", markers, widths }, markers);
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

  /**
   * Filtered pose extrapolated to a render timestamp (performance.now()
   * domain) — the render-time prediction that cancels filter/pipeline
   * latency. Returns a column-major 4x4 marker-object -> OpenCV-camera
   * matrix, or null while the target is not tracked. `tracear/three`
   * consumes this and handles axis conversion.
   */
  poseAt(index: number, timestamp: number): Float32Array | null {
    const lp = this.lastPoses[index];
    if (!lp) return null;
    const { pose } = lp;
    const dt = Math.min(Math.max((timestamp - lp.timestamp) / 1000, 0), 0.1);
    const p: Vec3 = [
      pose.position[0] + pose.velocity[0] * dt,
      pose.position[1] + pose.velocity[1] * dt,
      pose.position[2] + pose.velocity[2] * dt,
    ];
    const dq = quatFromScaledAxis([
      pose.angularVelocity[0] * dt,
      pose.angularVelocity[1] * dt,
      pose.angularVelocity[2] * dt,
    ]);
    const q = quatMultiply(pose.quaternion, dq);
    return mat4FromPose(q, p);
  }

  /** Latest camera intrinsics estimate (null before the first result). */
  intrinsics(): CameraIntrinsics | null {
    if (!this.focalRatio || !this.lastProcessW) return null;
    const f = this.focalRatio * this.lastProcessW;
    return {
      fx: f,
      fy: f,
      cx: this.lastProcessW / 2,
      cy: this.lastProcessH / 2,
      width: this.lastProcessW,
      height: this.lastProcessH,
    };
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
    for (let i = 0; i < this.targets.length; i++) {
      out.push(this.parseUpdate(result, i));
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

  private parseUpdate(msg: ResultMessage, index: number): UpdateEvent | null {
    const d = msg.data;
    const base = index * RESULT_STRIDE;
    const status = d[base];
    if (status === 0) return null;
    const state = this.targets[index];
    let pose: PoseData | undefined;
    if (d[base + 13] === 1.0) {
      pose = {
        quaternion: [d[base + 14], d[base + 15], d[base + 16], d[base + 17]],
        position: [d[base + 18], d[base + 19], d[base + 20]],
        velocity: [d[base + 21], d[base + 22], d[base + 23]],
        angularVelocity: [d[base + 24], d[base + 25], d[base + 26]],
      };
    }
    return {
      index,
      homography: d.slice(base + 1, base + 10),
      markerWidth: state.width,
      markerHeight: state.height,
      tracking: status === 2,
      inliers: d[base + 10],
      matches: d[base + 11],
      quality: d[base + 12],
      pose,
      timestamp: msg.timestamp,
      processWidth: msg.width,
      processHeight: msg.height,
      workerMs: msg.ms,
    };
  }

  private onResult(msg: ResultMessage): void {
    this.inflight = false;
    this.lastProcessW = msg.width;
    this.lastProcessH = msg.height;
    for (let i = 0; i < this.targets.length; i++) {
      const state = this.targets[i];
      const update = this.parseUpdate(msg, i);
      if (update) {
        this.focalRatio = msg.data[i * RESULT_STRIDE + 27];
        if (update.pose) {
          this.lastPoses[i] = { pose: update.pose, timestamp: msg.timestamp };
        }
        state.misses = 0;
        if (!state.found) {
          state.found = true;
          this.emitter.emit("targetFound", { index: i });
        }
        this.emitter.emit("update", update);
      } else if (state.found) {
        state.misses++;
        if (state.misses >= this.config.lostAfterMisses) {
          state.found = false;
          state.misses = 0;
          this.lastPoses[i] = null;
          this.emitter.emit("targetLost", { index: i });
        }
      }
    }
  }
}
