# `Tracear`

The core tracker. Import from the package root:

```ts
import { Tracear } from "@tracear/sdk";
```

## `Tracear.create(config)`

```ts
static async create(config: TracearConfig): Promise<Tracear>
```

Fetches/compiles nothing — it loads the given `.tracear` targets, boots the
tracking worker, and resolves when the engine is ready. Does **not** touch
the camera yet.

### `TracearConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `container` | `HTMLElement` | — | Element the managed `<video>` is appended to. Position it yourself (e.g. `position: relative`). |
| `targets` | `(string \| ArrayBuffer \| Uint8Array)[]` | — | Compiled `.tracear` targets: URLs or raw bytes. A file may hold one marker or a [multi-marker pack](/guide/markers#many-targets-one-file-packs); packs expand in place. |
| `targetWidthsMeters` | `number[]` | `1` per marker | Physical width of each marker (in expanded pack order) — poses come out in the same unit. |
| `maxProcessSize` | `number` | `640` | Long-side cap for the processed frame. |
| `maxTracked` | `number` | unlimited | Cap on simultaneously tracked targets. `1` = exclusive session (like MindAR's `maxTrack: 1`): once a target is acquired, all other detection pauses until it is lost — cheapest and steadiest when only one target should be active at a time. |
| `lostAfterMisses` | `number` | `8` | Consecutive misses before `targetLost`. |
| `videoConstraints` | `MediaTrackConstraints` | — | Extra `getUserMedia` video constraints, merged over the defaults (environment camera, 1280×720 ideal). |

## Methods

### `start()`

```ts
async start(): Promise<void>
```

Requests the camera (`getUserMedia`), attaches the video to `container`, and
begins processing frames. Rejects if the user denies permission.

### `stop()`

Stops frame processing and releases the camera. The tracker can `start()`
again later.

### `dispose()`

`stop()` plus: terminates the worker, removes the video element, clears all
listeners. The instance is done after this.

### `on(event, callback)`

```ts
on<K extends keyof TracearEvents>(event: K, cb: (e: TracearEvents[K]) => void): () => void
```

Subscribes to an [event](#events); returns an unsubscribe function.

### `poseAt()` {#poseat}

```ts
poseAt(index: number, timestamp: number): Float32Array | null
```

The filtered pose blended and extrapolated to a render timestamp
(`performance.now()` domain). Returns a **column-major 4×4** marker-object →
OpenCV-camera matrix, or `null` while the target isn't tracked. This is the
method to render from — [`TracearThree`](/reference/three) calls it for you
and converts to WebGL axes. See
[Tracking & poses](/guide/tracking-and-poses#render-time-prediction-poseat).

### `intrinsics()`

```ts
intrinsics(): CameraIntrinsics | null
```

Latest self-calibrated pinhole intrinsics, in processed-frame pixels. `null`
before the first result.

```ts
interface CameraIntrinsics {
  fx: number; fy: number;
  cx: number; cy: number;
  width: number; height: number; // the frame size fx/cx are expressed in
}
```

### `detectImage()`

```ts
async detectImage(source: ImageBitmapSource): Promise<(UpdateEvent | null)[]>
```

One-shot detection on a still image — no camera involved, no events emitted.
One entry per configured target, `null` where not found. Useful for marker
validation and automated tests.

## Properties

| Property | Type | Description |
|---|---|---|
| `video` | `HTMLVideoElement` | The managed video element (created in `create()`, attached on `start()`). |

## Events

```ts
type TracearEvents = {
  targetFound: { index: number };
  targetLost:  { index: number };
  update:      UpdateEvent;
  error:       { message: string };
};
```

### `UpdateEvent` {#updateevent}

| Field | Type | Description |
|---|---|---|
| `index` | `number` | Which target. |
| `homography` | `Float64Array` | Row-major 3×3, maps marker px → processed-frame px. |
| `markerWidth` / `markerHeight` | `number` | Compiled marker size in marker px. |
| `tracking` | `boolean` | `true` when the sub-pixel tracker produced this pose; `false` for full detection (first acquire / re-acquire / `detectImage`). |
| `inliers` | `number` | Detection: RANSAC inliers. Tracking: surviving patches. |
| `matches` | `number` | Detection: total matches. Tracking: attempted patches. |
| `quality` | `number` | 0..1 confidence. |
| `pose` | `PoseData?` | Filtered 6DoF pose; `undefined` if pose estimation failed this frame. |
| `timestamp` | `number` | Frame capture time (`performance.now()` domain). |
| `processWidth` / `processHeight` | `number` | Processed-frame size the homography maps into. |
| `workerMs` | `number` | CV processing time inside the worker (ms). |

### `PoseData`

Conventions: object frame origin at the marker center, X right, Y up, Z out
of the marker; camera frame is OpenCV (X right, Y down, Z forward).

| Field | Type | Description |
|---|---|---|
| `position` | `[x, y, z]` | Filtered object → camera translation (physical units). |
| `quaternion` | `[x, y, z, w]` | Filtered object → camera rotation. |
| `velocity` | `[x, y, z]` | Filtered linear velocity, units/s. |
| `angularVelocity` | `[x, y, z]` | Body-frame angular velocity, rad/s. |
| `posLagS` / `rotLagS` | `number` | Group delay (s) of the translation / rotation filter. |
| `rawPosition` / `rawQuaternion` | — | This frame's unfiltered pose: zero lag, more noise. |
