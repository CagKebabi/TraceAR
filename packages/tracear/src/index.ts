/**
 * Tracear SDK — public API.
 *
 * M1 state: continuous *detection* (worker + WASM) with homography output.
 * Sub-pixel tracking (M2) and pose/filtering (M3) slot in behind this same
 * API; `poseAt` stays null until M3.
 */
import { Emitter } from "./events";
import { mat4FromPose, quatAngle, quatSlerp, type Quat, type Vec3 } from "./math";
import type { ReadyMessage, ResultMessage, ErrorMessage, FallbackMessage } from "./worker";

export type Homography = Float64Array;

/** Values per marker in a worker result (mirrors the wasm crate):
 * [status (0/1/2), h00..h22, nGood, nTotal, quality,
 *  poseValid, qx,qy,qz,qw, tx,ty,tz, vx,vy,vz, wx,wy,wz,
 *  posLagS, rotLagS, rqx,rqy,rqz,rqw, rtx,rty,rtz, focalRatio] */
const RESULT_STRIDE = 37;

export interface TracearConfig {
  /** Element the managed <video> is appended to; position it yourself (e.g. relative). */
  container: HTMLElement;
  /** Compiled `.tracear` targets: URLs or raw bytes. A target file may hold
   * a single marker or a multi-marker pack (see `packMarkers`); packs expand
   * in place, so marker indices follow the expanded order. */
  targets: (string | ArrayBuffer | Uint8Array)[];
  /** Physical width of each marker in meters (or any unit — poses come out
   * in the same unit), in expanded marker order. Defaults to 1 per marker. */
  targetWidthsMeters?: number[];
  /** Long-side cap for the processed frame, default 640. */
  maxProcessSize?: number;
  /** Cap on simultaneously tracked targets; 0/undefined = unlimited.
   * `1` gives an exclusive session (like MindAR's maxTrack:1): once a
   * target is acquired, all other detection pauses until it is lost —
   * the cheapest and steadiest mode when only one target should be
   * active at a time. */
  maxTracked?: number;
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
  /** Group delay (s) of the translation filter — prediction adds it back. */
  posLagS: number;
  /** Group delay (s) of the rotation filter. */
  rotLagS: number;
  /** This frame's un-filtered pose: zero-lag but noisy. Rendering blends
   * toward it with speed (motion masks noise; filter lag would show as swim). */
  rawPosition: Vec3;
  rawQuaternion: Quat;
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
  private lastPoses: ({ pose: PoseData; timestamp: number; emaSpeed: number; emaAngSpeed: number } | null)[] = [];
  /** Presentation-side rotation state for render smoothing (per target). */
  private renderQuats: (Quat | null)[] = [];
  private focalRatio = 0;
  private lastProcessW = 0;
  private lastProcessH = 0;

  private constructor(config: TracearConfig, worker: Worker, markerSizes: [number, number][]) {
    this.config = { maxProcessSize: 640, lostAfterMisses: 8, ...config };
    this.worker = worker;
    this.targets = markerSizes.map(([width, height]) => ({ found: false, misses: 0, width, height }));
    this.lastPoses = markerSizes.map(() => null);
    this.renderQuats = markerSizes.map(() => null);
    this.video = document.createElement("video");
    this.video.playsInline = true;
    this.video.muted = true;
    this.video.autoplay = true;
    this.video.style.display = "block";
    this.video.style.width = "100%";
    this.worker.onmessage = (ev: MessageEvent<ResultMessage | ErrorMessage | FallbackMessage>) => {
      if (ev.data.type === "result") {
        const { requestId } = ev.data;
        if (requestId !== undefined) {
          this.pending.get(requestId)?.(ev.data);
          this.pending.delete(requestId);
        } else {
          this.onResult(ev.data);
        }
      } else if (ev.data.type === "fallback") {
        // The VideoFrame fast path is unusable (format/API) — bitmaps next.
        this.useVideoFramePath = false;
        this.inflight = false;
      } else if (ev.data.type === "error") {
        this.inflight = false; // never let a failed frame stall the pump
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
    // Widths align with EXPANDED marker order (packs may hold several
    // markers); the worker consumes them as targets expand, padding with 1.0.
    const widths = config.targetWidthsMeters ?? [];
    worker.postMessage(
      { type: "init", markers, widths, maxTracked: config.maxTracked ?? 0 },
      markers,
    );
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
    // Speed-adaptive raw/filtered blend: at rest render the filtered pose
    // (frozen); the moment motion starts (instantaneous speed responds
    // within one frame) shift toward the RAW measurement, which has zero
    // filter lag — motion masks its noise, while filter lag would read as
    // the content swimming behind the target.
    const smoothstep = (x: number, lo: number, hi: number) => {
      const t = Math.min(Math.max((x - lo) / (hi - lo), 0), 1);
      return t * t * (3 - 2 * t);
    };
    // Thresholds sit safely above the at-rest noise floor of the raw-pose
    // speed estimate (~0.05 u/s), so alpha is EXACTLY zero on a still scene.
    const alphaSpeed = Math.max(
      smoothstep(lp.emaSpeed, 0.1, 0.5),
      smoothstep(lp.emaAngSpeed, 0.15, 0.8),
    );
    // Error-adaptive catch-up: whenever the rendered (filtered) pose has
    // visibly diverged from the raw measurement — after a fast move, a
    // re-acquire, or the marker itself moving — snap toward raw NOW instead
    // of easing back at the filter's own pace. Position error is measured
    // relative to depth (screen-proportional); both floors sit above the
    // at-rest noise level, so a still scene stays frozen.
    const dp = [
      pose.rawPosition[0] - pose.position[0],
      pose.rawPosition[1] - pose.position[1],
      pose.rawPosition[2] - pose.position[2],
    ];
    const relPosErr = Math.hypot(dp[0], dp[1], dp[2]) / Math.max(Math.abs(pose.position[2]), 0.2);
    const rotErr = quatAngle(pose.quaternion, pose.rawQuaternion);
    const alpha = Math.max(alphaSpeed, smoothstep(relPosErr, 0.004, 0.02));
    // Rotation blends far more conservatively than translation: planar-pose
    // estimation has a rotation ambiguity whose noise makes raw orientation
    // wobble visibly, while a little rotational lag is imperceptible.
    // Its own error term still pulls hard when orientation truly diverged.
    const alphaRot = Math.max(alphaSpeed * 0.5, smoothstep(rotErr, 0.025, 0.09));
    const basePos: Vec3 = [
      pose.position[0] + (pose.rawPosition[0] - pose.position[0]) * alpha,
      pose.position[1] + (pose.rawPosition[1] - pose.position[1]) * alpha,
      pose.position[2] + (pose.rawPosition[2] - pose.position[2]) * alpha,
    ];
    const baseQ = quatSlerp(pose.quaternion, pose.rawQuaternion, alphaRot);

    // Extrapolate TRANSLATION over the pipeline latency only. Filter-lag
    // compensation is deliberately NOT added: at high speed the raw blend
    // already removes the lag, and at low speed the lag distance is tiny
    // while the compensation horizon (~200 ms x a noisy velocity) launched
    // the content off the target and back — the "overshoots then settles"
    // artifact. Rotation is not extrapolated at all (the ambiguity puts
    // fake rad/s into the angular velocity; predicting with it manufactures
    // wobble far worse than the imperceptible lag it would remove).
    const latency = Math.min(Math.max((timestamp - lp.timestamp) / 1000, 0), 0.1);
    // Deadband: velocity estimates are never exactly zero — do not let their
    // noise wiggle a still scene. Clamp: never extrapolate absurdly far.
    const v = pose.velocity.map((x) => (Math.abs(x) < 0.01 ? 0 : x)) as Vec3;
    const step = Math.hypot(v[0], v[1], v[2]) * latency;
    const posScale = step > 0.2 ? 0.2 / step : 1;
    const p: Vec3 = [
      basePos[0] + v[0] * latency * posScale,
      basePos[1] + v[1] * latency * posScale,
      basePos[2] + v[2] * latency * posScale,
    ];
    // Render-side rotation glide: measurements arrive at ~25-30 Hz while the
    // display runs faster; without this, orientation visibly steps. Half-way
    // slerp per render frame turns steps into a glide at ~1 frame of lag.
    const prevQ = this.renderQuats[index];
    const q = prevQ ? quatSlerp(prevQ, baseQ, 0.5) : baseQ;
    this.renderQuats[index] = q;
    return mat4FromPose(q, p);
  }

  /** Latest camera intrinsics estimate (null before the first result). */
  intrinsics(): CameraIntrinsics | null {
    if (!this.focalRatio || !this.lastProcessW) return null;
    // focalRatio is defined against the LONG side (orientation-invariant).
    const f = this.focalRatio * Math.max(this.lastProcessW, this.lastProcessH);
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

  /** Fastest first: WebCodecs VideoFrame (Y plane read in the worker, no
   * canvas at all) -> GPU bitmap -> main-thread canvas readback. Each path
   * permanently falls back on first failure. */
  private useVideoFramePath = typeof VideoFrame === "function";
  private useBitmapPath = typeof createImageBitmap === "function" && typeof OffscreenCanvas === "function";

  private processFrame(): void {
    if (!this.running) return;
    // Drop frames while the worker is busy — never queue.
    if (!this.inflight && this.video.readyState >= 2) {
      const timestamp = performance.now();
      if (this.useVideoFramePath) {
        let frame: VideoFrame | null = null;
        try {
          frame = new VideoFrame(this.video);
        } catch {
          this.useVideoFramePath = false;
        }
        if (frame) {
          this.inflight = true;
          this.worker.postMessage(
            { type: "frame", videoFrame: frame, width: this.processW, height: this.processH, timestamp },
            [frame as unknown as Transferable],
          );
        }
      } else if (this.useBitmapPath) {
        // createImageBitmap(video, {resize}) stays on the GPU and returns in
        // ~1 ms; the expensive pixel readback happens in the worker, so the
        // frame pump is not serialized on the main thread (on phones a main-
        // thread getImageData costs tens of ms and halves the update rate).
        this.inflight = true;
        createImageBitmap(this.video, {
          resizeWidth: this.processW,
          resizeHeight: this.processH,
        }).then(
          (bitmap) => {
            if (!this.running) {
              bitmap.close();
              this.inflight = false;
              return;
            }
            this.worker.postMessage(
              { type: "frame", bitmap, width: this.processW, height: this.processH, timestamp },
              [bitmap],
            );
          },
          () => {
            // resize options or video source unsupported — permanent fallback
            this.useBitmapPath = false;
            this.inflight = false;
          },
        );
      } else if (this.ctx) {
        this.ctx.drawImage(this.video, 0, 0, this.processW, this.processH);
        const img = this.ctx.getImageData(0, 0, this.processW, this.processH);
        this.inflight = true;
        this.worker.postMessage(
          { type: "frame", buf: img.data.buffer, width: this.processW, height: this.processH, timestamp },
          [img.data.buffer],
        );
      }
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
        posLagS: d[base + 27],
        rotLagS: d[base + 28],
        rawQuaternion: [d[base + 29], d[base + 30], d[base + 31], d[base + 32]],
        rawPosition: [d[base + 33], d[base + 34], d[base + 35]],
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
        this.focalRatio = msg.data[i * RESULT_STRIDE + 36];
        if (update.pose) {
          // Instantaneous speed from consecutive RAW poses: responds within
          // one frame of motion starting — the filtered velocity estimate
          // lags and would delay the raw/filtered rendering blend.
          const prev = this.lastPoses[i];
          let instSpeed = 0;
          let instAngSpeed = 0;
          if (prev) {
            const dt = Math.max((msg.timestamp - prev.timestamp) / 1000, 1e-3);
            const dp = update.pose.rawPosition.map((v, k) => v - prev.pose.rawPosition[k]);
            instSpeed = Math.hypot(dp[0], dp[1], dp[2]) / dt;
            instAngSpeed = quatAngle(update.pose.rawQuaternion, prev.pose.rawQuaternion) / dt;
          }
          // EMA: single-frame speed spikes (pose noise at rest crosses any
          // low threshold) must not flap the raw/filtered blend around.
          const emaSpeed = prev ? prev.emaSpeed + 0.35 * (instSpeed - prev.emaSpeed) : instSpeed;
          const emaAngSpeed = prev ? prev.emaAngSpeed + 0.35 * (instAngSpeed - prev.emaAngSpeed) : instAngSpeed;
          this.lastPoses[i] = { pose: update.pose, timestamp: msg.timestamp, emaSpeed, emaAngSpeed };
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
          this.renderQuats[i] = null;
          this.emitter.emit("targetLost", { index: i });
        }
      }
    }
  }
}
