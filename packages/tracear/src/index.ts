/**
 * Tracear SDK — public API surface (types only until M1 wires the WASM core).
 *
 * The API is frozen conceptually in docs/ARCHITECTURE.md; implementation
 * lands in M1 (camera + worker + WASM bridge) and M3 (pose + filtering).
 */

/** 3x3 row-major homography, marker px -> frame px. */
export type Homography = Float64Array;

/** 4x4 column-major (WebGL/three.js convention) rigid pose, marker -> camera. */
export type PoseMatrix = Float32Array;

export interface TracearConfig {
  /** Element that will contain the managed <video> and size the coordinate space. */
  container: HTMLElement;
  /** Compiled `.tracear` targets: URLs or raw buffers. */
  targets: (string | ArrayBuffer)[];
  /** Physical width of each target in meters (defaults to 1 unit). */
  targetWidthsMeters?: number[];
  /** Cap for the processed frame's long side, default 640. */
  maxProcessSize?: number;
  /** Override the estimated camera FOV in degrees (advanced). */
  cameraFovDeg?: number;
}

export interface UpdateEvent {
  index: number;
  homography: Homography;
  pose: PoseMatrix;
  /** 0..1 track quality (surviving patches, NCC stats). */
  quality: number;
  /** Camera-frame capture timestamp (performance.now() domain). */
  timestamp: number;
}

export interface TracearEvents {
  targetFound: { index: number };
  targetLost: { index: number };
  update: UpdateEvent;
  error: { message: string };
}

export interface Tracear {
  start(): Promise<void>;
  stop(): void;
  dispose(): void;
  on<K extends keyof TracearEvents>(event: K, cb: (e: TracearEvents[K]) => void): () => void;
  /** Filtered pose predicted to the given render timestamp (see ARCHITECTURE: filtering). */
  poseAt(index: number, timestamp: number): PoseMatrix | null;
}

export const Tracear = {
  async create(_config: TracearConfig): Promise<Tracear> {
    throw new Error("tracear: not implemented yet — runtime lands in milestone M1");
  },
};
