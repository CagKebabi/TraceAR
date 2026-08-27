# @tracear/sdk

[![npm](https://img.shields.io/npm/v/%40tracear%2Fsdk?color=2a6df4)](https://www.npmjs.com/package/@tracear/sdk)
[![CI](https://github.com/CagKebabi/TraceAR/actions/workflows/ci.yml/badge.svg)](https://github.com/CagKebabi/TraceAR/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue)](https://github.com/CagKebabi/TraceAR/blob/main/LICENSE)

**High-performance, jitter-free image tracking for the mobile web.**

Tracear tracks known image targets with the phone camera, in the browser, and
gives you a filtered 6DoF pose to hang 3D content on — built as a lean
Rust→WASM(+SIMD) engine with sub-pixel frame-to-frame tracking, online camera
self-calibration, and render-time pose prediction.

Measured against MindAR on the same phone, same marker, same metric
(median jitter of the marker center, 640 px frame): **1.9 px vs 5.5 px** —
about 3× steadier — with ~2.5 ms/frame of CV time while tracking and
detection around 20 ms. Compile a 512 px marker in ~0.1 s to a ~170 KB file
(MindAR: ~1.4 s, ~420 KB). SDK weight: **~92 KB gzipped** including WASM.

> Status: **0.1 — early but real.** Android Chrome + iOS Safari 16.4+.
> APIs may still move before 1.0.

## Install

```sh
npm i @tracear/sdk
```

Use a bundler. With **Vite**, add one line so the dev server doesn't
pre-bundle the SDK (pre-bundling breaks the worker/WASM asset URLs; production
builds are unaffected either way):

```ts
// vite.config.ts
export default defineConfig({
  optimizeDeps: { exclude: ["@tracear/sdk"] },
});
```

Serving directly from a CDN does not work yet (workers must be same-origin).

## 1 · Compile a marker

Turn your target image (poster, packaging, card…) into a `.tracear` file —
once, offline:

```sh
npx tracear compile poster.png        # -> poster.tracear
```

or in the browser:

```ts
import { compileImage } from "@tracear/sdk/compiler";
const { data } = await compileImage(imageFileOrCanvas); // Uint8Array (.tracear)
```

Good markers are textured and asymmetric; avoid large flat areas and
repeating patterns.

## 2 · Track

```ts
import { Tracear } from "@tracear/sdk";

const tracker = await Tracear.create({
  container: document.querySelector("#ar")!, // gets the <video> appended
  targets: ["/markers/poster.tracear"],
  targetWidthsMeters: [0.2], // optional: physical width -> metric poses
});

tracker.on("targetFound", ({ index }) => console.log("found", index));
tracker.on("targetLost", ({ index }) => console.log("lost", index));
tracker.on("update", (e) => {
  // e.homography: marker px -> camera-frame px (Float64Array, row-major 3x3)
  // e.pose: filtered 6DoF pose + velocities (see conventions below)
  // e.tracking: false on (re)detection frames, true while tracking
});

await tracker.start();
```

For rendering, ask for the pose at *display* time — this applies the
render-time prediction that cancels pipeline latency:

```ts
const m = tracker.poseAt(0, performance.now()); // column-major 4x4 | null
```

## 3 · three.js

```ts
import * as THREE from "three";
import { TracearThree } from "@tracear/sdk/three";

const t3 = new TracearThree(tracker);
scene.add(t3.anchor(0));          // put your content inside this Group
// each frame:
t3.update();                       // best driven by video.requestVideoFrameCallback
renderer.render(scene, t3.camera); // camera projection follows the self-calibrated intrinsics
```

Anchor space: origin at the marker center, X right, Y up, Z out of the
marker; 1 unit = the target's physical width unit (1 marker-width if unset).

## Conventions

- Homographies are row-major 3×3 mapping **marker pixels → processed-frame
  pixels** (`p' = H·(x, y, 1)`, divide by w).
- Poses map the marker-centered object frame into an **OpenCV-style camera
  frame** (X right, Y down, Z forward); `tracear/three` converts for WebGL.
- Camera intrinsics are estimated online from tracked views
  (`tracker.intrinsics()`), starting from a typical phone FOV.

## How it stays smooth

Detection (FAST + rotated BRIEF + RANSAC) runs only to acquire; every other
frame is sub-pixel inverse-compositional patch alignment against the
compiled marker — coarse-to-fine over a half-resolution level so fast
handheld motion and motion blur survive. Poses go through One-Euro-on-SE(3)
filtering with a rotation-ambiguity prior in the pose solver, and rendering
blends filtered↔raw by instantaneous speed: frozen when still, glued when
moving.

## License

MIT — free for commercial use. Source: [github.com/CagKebabi/TraceAR](https://github.com/CagKebabi/TraceAR).
