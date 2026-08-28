# Tracking & poses

This page explains what the engine reports and in which coordinate systems.
If you only render through [`TracearThree`](/guide/three), you can skip most
of it — the adapter handles the conventions for you.

## Lifecycle & events

```ts
tracker.on("targetFound", ({ index }) => {});
tracker.on("targetLost", ({ index }) => {});
tracker.on("update", (e) => {});
tracker.on("error", ({ message }) => {});
```

- `targetFound` fires when a target is first detected (and on re-acquire
  after a loss).
- `update` fires for every processed frame where the target is visible, with
  the full [`UpdateEvent`](/reference/tracear#updateevent).
- `targetLost` fires after `lostAfterMisses` consecutive misses (default 8) —
  brief occlusions don't flicker the content.
- `on()` returns an unsubscribe function.

Internally the engine runs a detect ↔ track state machine: full detection
finds the target, then a sub-pixel patch tracker follows it frame-to-frame
(`e.tracking` tells you which mode produced the pose). Tracking is both much
faster and much less noisy than per-frame re-detection.

## The homography

Every update carries `homography`: a row-major 3×3 matrix mapping **marker
pixels → processed-frame pixels**. It's the right tool for 2D overlays:

```ts
function project(h: Float64Array, x: number, y: number) {
  const w = h[6] * x + h[7] * y + h[8];
  return [(h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w];
}

// Marker outline in frame pixels:
const corners = [
  [0, 0],
  [e.markerWidth, 0],
  [e.markerWidth, e.markerHeight],
  [0, e.markerHeight],
].map(([x, y]) => project(e.homography, x, y));
```

Frame pixels are in the processed frame (`e.processWidth` ×
`e.processHeight`); scale by `displayWidth / e.processWidth` to draw over
your video element.

## The 6DoF pose

`e.pose` (when present) is the filtered rigid transform **object → camera**:

- **Object frame**: origin at the marker center, X right, Y up, Z out of the
  marker toward the viewer. Units follow `targetWidthsMeters`.
- **Camera frame**: OpenCV convention — X right, Y down, Z forward.

`position`/`quaternion` are the filtered (smooth) pose; `rawPosition`/
`rawQuaternion` are the same frame's unfiltered measurement — zero lag but
noisy. `velocity`, `angularVelocity` and the filter lags (`posLagS`,
`rotLagS`) support prediction.

## Render-time prediction: `poseAt`

Don't render `e.pose` directly — measurements arrive at camera rate and lag
the display. Instead call:

```ts
const m = tracker.poseAt(0, performance.now()); // Float32Array | null
```

It returns a column-major 4×4 object → OpenCV-camera matrix, blended and
extrapolated for *this* render moment:

- at rest it renders the filtered pose — content stays frozen;
- during motion it shifts toward the raw measurement, whose noise the motion
  masks, so content stays glued instead of swimming behind the target;
- after a fast move or re-acquire it snaps back quickly instead of easing in.

For three.js you never call it yourself — `t3.update()` does, plus the
OpenCV → WebGL axis flip.

## Camera intrinsics

```ts
const intr = tracker.intrinsics(); // null before the first result
// { fx, fy, cx, cy, width, height } in processed-frame pixels
```

The focal length is self-calibrated online from the tracked homographies and
refines over the first seconds of tracking. Use it to build a projection
matrix if you render with something other than the three.js adapter.

## Still images: `detectImage`

One-shot detection with no camera — marker validation, thumbnails, tests:

```ts
const results = await tracker.detectImage(imageOrCanvasOrBlob);
// one UpdateEvent | null per configured target; emits no events
```
